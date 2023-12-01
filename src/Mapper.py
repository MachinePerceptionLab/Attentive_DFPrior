import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable

from src.common import (get_samples, random_select)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Mapper(object):
    """
    Mapper thread. 

    """

    def __init__(self, cfg, args, slam
                 ):

        self.cfg = cfg
        self.args = args

        self.idx = slam.idx
        self.c = slam.shared_c
        self.bound = slam.bound
        self.logger = slam.logger
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame
        self.scene_id = slam.scene_id
        with torch.no_grad():
            self.tsdf_volume_shared = slam.tsdf_volume_shared
            self.tsdf_bnds = slam.tsdf_bnds
        
    
        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.device = cfg['mapping']['device']
        self.fix_high = cfg['mapping']['fix_high']
        self.eval_rec = cfg['meshing']['eval_rec']
    
        
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.fix_color = cfg['mapping']['fix_color']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.num_joint_iters = cfg['mapping']['iters']
        self.clean_mesh = cfg['meshing']['clean_mesh']
        self.every_frame = cfg['mapping']['every_frame']
        self.color_refine = cfg['mapping']['color_refine']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.high_iter_ratio = cfg['mapping']['high_iter_ratio']
        self.low_iter_ratio = cfg['mapping']['low_iter_ratio']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}


        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        if 'Demo' not in self.output:  # disable this visualization in demo
            self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                         vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                         verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]))

        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = 0
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        # For ray with depth==0, fill it with maximum depth
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak-ray_o
        dist = torch.sum(dist*dist, axis=1)
        mask2 = dist < 0.5*0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=16, pixels=100):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            cam_cord_homo = w2c@homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list
    
    def eval_points(self, p, decoders, tsdf_volume, tsdf_bnds, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): point coordinates.
            decoders (nn.module decoders): decoders.
            c (dicts, optional): feature grids. Defaults to None.
            stage (str, optional): query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): device name to compute on. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, 500)
        bound = self.bound
        rets = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            pi = pi.unsqueeze(0)
            ret, _ = decoders(pi, c_grid=c, tsdf_volume=tsdf_volume, tsdf_bnds=tsdf_bnds, stage=stage)
  
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret

    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict, keyframe_list, tsdf_volume, cur_c2w):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation.

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.
            tsdf_volume (tensor): tsdf volume.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            return None
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        c = self.c
        cfg = self.cfg
        device = self.device
        tsdf_bnds = self.tsdf_bnds.to(device)

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                num = self.mapping_window_size-2
                optimize_frame = random_select(len(self.keyframe_dict)-1, num)
            elif self.keyframe_selection_method == 'overlap':
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num)

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w']
                else:
                    frame_idx = idx
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
            self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        mlp_para_list = []
        decoders_para_list = []
        low_grid_para = []
        high_grid_para = []
        color_grid_para = []
        gt_depth_np = cur_gt_depth.cpu().numpy()
        if True:
            if self.frustum_feature_selection:
                masked_c_grad = {}
                mask_c2w = cur_c2w
            for key, val in c.items():
                if not self.frustum_feature_selection:
                    val = Variable(val.to(device), requires_grad=True)
                    c[key] = val
                    if key == 'grid_low':
                        low_grid_para.append(val)
                    elif key == 'grid_high':
                        high_grid_para.append(val)
                    elif key == 'grid_color':
                        color_grid_para.append(val)

                else:
                    mask = self.get_mask_from_c2w(
                        mask_c2w, key, val.shape[2:], gt_depth_np)
                    mask = torch.from_numpy(mask).permute(2, 1, 0).unsqueeze(
                        0).unsqueeze(0).repeat(1, val.shape[1], 1, 1, 1)
                    val = val.to(device)
                    # val_grad is the optimizable part, other parameters will be fixed
                    val_grad = val[mask].clone()
                    val_grad = Variable(val_grad.to(
                        device), requires_grad=True)
                    masked_c_grad[key] = val_grad
                    masked_c_grad[key+'mask'] = mask
                    if key == 'grid_low':
                        low_grid_para.append(val_grad)
                    elif key == 'grid_high':
                        high_grid_para.append(val_grad)
                    elif key == 'grid_color':
                        color_grid_para.append(val_grad)


        if not self.fix_high:
            decoders_para_list += list(
                self.decoders.high_decoder.parameters())
        if not self.fix_color:
            decoders_para_list += list(
                self.decoders.color_decoder.parameters())
        mlp_para_list += list(
            self.decoders.mlp.parameters())
        

        optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                        {'params': mlp_para_list, 'lr': 0},
                                        {'params': low_grid_para, 'lr': 0},
                                        {'params': high_grid_para, 'lr': 0},
                                        {'params': color_grid_para, 'lr': 0}])
        

        for joint_iter in range(num_joint_iters):
            if self.frustum_feature_selection:
                for key, val in c.items():
                    val_grad = masked_c_grad[key]
                    mask = masked_c_grad[key+'mask']
                    val = val.to(device)
                    val[mask] = val_grad
                    c[key] = val

            if joint_iter <= int(num_joint_iters*self.low_iter_ratio):
                self.stage = 'low'
            elif joint_iter <= int(num_joint_iters*self.high_iter_ratio):
                self.stage = 'high'
            else:
                self.stage = 'color'

            optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['decoders_lr']*lr_factor
            optimizer.param_groups[1]['lr'] = cfg['mapping']['stage'][self.stage]['mlp_lr']*lr_factor
            optimizer.param_groups[2]['lr'] = cfg['mapping']['stage'][self.stage]['low_lr']*lr_factor
            optimizer.param_groups[3]['lr'] = cfg['mapping']['stage'][self.stage]['high_lr']*lr_factor
            optimizer.param_groups[4]['lr'] = cfg['mapping']['stage'][self.stage]['color_lr']*lr_factor
            
            if (not (idx == 0 and self.no_vis_on_first_frame)) and ('Demo' not in self.output):
                self.visualizer.vis(
                    idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, self.c, self.decoders, tsdf_volume, tsdf_bnds)

            optimizer.zero_grad()
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []

            camera_tensor_id = 0
            for frame in optimize_frame:
                if frame != -1:
                    gt_depth = keyframe_dict[frame]['depth'].to(device)
                    gt_color = keyframe_dict[frame]['color'].to(device)
                    c2w = keyframe_dict[frame]['est_c2w']

                else:
                    gt_depth = cur_gt_depth.to(device)
                    gt_color = cur_gt_color.to(device)
                    c2w = cur_c2w

                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                    0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)


            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(
                    device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

            ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
                                                 batch_rays_o, device, tsdf_volume, tsdf_bnds, self.stage,
                                                 batch_gt_depth)
            depth, uncertainty, color, weight = ret


            depth_mask = (batch_gt_depth > 0)
            
            if joint_iter > int(num_joint_iters*self.low_iter_ratio) and joint_iter <= int(num_joint_iters*self.low_iter_ratio)+5 and idx <= 1:
                loss = torch.abs(
                batch_gt_depth[depth_mask]-depth[depth_mask]).sum() + torch.abs(weight-torch.ones(weight.shape).to(device)).sum()
            else:
                loss = torch.abs(
                    batch_gt_depth[depth_mask]-depth[depth_mask]).sum()
           
            if self.stage == 'color':
                color_loss = torch.abs(batch_gt_color - color).sum()
                weighted_color_loss = self.w_color_loss*color_loss
                loss += weighted_color_loss

            loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()

            # put selected and updated features back to the grid
            if self.frustum_feature_selection:
                for key, val in c.items():
                    val_grad = masked_c_grad[key]
                    mask = masked_c_grad[key+'mask']
                    val = val.detach()
                    val[mask] = val_grad.clone().detach()
                    c[key] = val

        return None


    def run(self):
        cfg = self.cfg
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]

        self.estimate_c2w_list[0] = gt_c2w.cpu()
        init = True
        prev_idx = -1
        tsdf_volume = self.tsdf_volume_shared
    
        while (1):
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1:
                    break
                if self.sync_method == 'strict':
                    if idx % self.every_frame == 0 and idx != prev_idx:
                        break
                elif self.sync_method == 'loose':
                    if idx == 0 or idx >= prev_idx+self.every_frame//2:
                        break
                elif self.sync_method == 'free':
                    break
                time.sleep(0.1)
            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                prefix = ''
                print(prefix+"Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx]

            # valid c2w
            valid_c2w = gt_c2w.clone().cpu().numpy()
            if not np.isfinite(valid_c2w).any():
                self.mapping_idx[0] = idx
                continue


            if not init:
                lr_factor = cfg['mapping']['lr_factor']
                num_joint_iters = cfg['mapping']['iters']

                # here provides a color refinement postprocess
                if idx == self.n_img-1 and self.color_refine:
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2
                    self.low_iter_ratio = 0.0
                    self.high_iter_ratio = 0.0
                    num_joint_iters *= 5
                    self.fix_color = True
                    self.frustum_feature_selection = False
                else:
                    outer_joint_iters = 1
                    

            else:
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor']
                num_joint_iters = cfg['mapping']['iters_first']

            cur_c2w = self.estimate_c2w_list[idx].to(self.device)
            num_joint_iters = num_joint_iters//outer_joint_iters
           
            for outer_joint_iter in range(outer_joint_iters):


                _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, tsdf_volume, cur_c2w=cur_c2w)
                

                # add new frame to keyframe set
                if outer_joint_iter == outer_joint_iters-1:
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) \
                            and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                        ), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone()})

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            init = False
            # mapping of first frame is done, can begin tracking
            self.mapping_first_frame[0] = 1

            if True:
                if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) \
                        or idx == self.n_img-1 or (idx == 4640 and self.scene_id==50):
                    self.logger.log(idx, self.keyframe_dict, self.keyframe_list,
                                    selected_keyframes=self.selected_keyframes
                                    if self.save_selected_keyframes_info else None)

                self.mapping_idx[0] = idx
                self.mapping_cnt[0] += 1

                if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                    mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx, tsdf_volume, self.device,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)

                if idx == self.n_img-1 or (idx == 4640 and self.scene_id==50):
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx, tsdf_volume, self.device,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)
                    os.system(
                        f"cp {mesh_out_file} {self.output}/mesh/{idx:05d}_mesh.ply")
                    if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                             self.estimate_c2w_list, idx, tsdf_volume, self.device,
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True)
                    break

            if idx == self.n_img-1 or (idx == 4640 and self.scene_id==50):
                break
