import torch
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf, normalize_3d_coordinate
import torch.nn.functional as F


class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        self.lindisp = cfg['rendering']['lindisp']
        self.perturb = cfg['rendering']['perturb']
        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']
        self.N_importance = cfg['rendering']['N_importance']

        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        self.bound = slam.bound
        self.sample_mode = 'bilinear'
        self.tsdf_bnds = slam.vol_bnds

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

        self.resolution = cfg['meshing']['resolution']

    def eval_points(self, p, decoders, tsdf_volume, tsdf_bnds, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            tsdf_volume (tensor): tsdf volume.
            tsdf_bnds (tensor): tsdf volume bounds.
            c (dicts, optional): Feature grids. Defaults to None.
            stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)
        bound = self.bound
        rets = []
        weights = []

        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            pi = pi.unsqueeze(0)
            ret, w = decoders(pi, c_grid=c, tsdf_volume=tsdf_volume, tsdf_bnds=tsdf_bnds, stage=stage)
            ret = ret.squeeze(0)


            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100 
            rets.append(ret)
            weights.append(w)

        ret = torch.cat(rets, dim=0)
        weight = torch.cat(weights, dim=0)

        return ret, weight 

    def sample_grid_tsdf(self, p, tsdf_volume, device='cuda:0'):

        p_nor = normalize_3d_coordinate(p.clone(), self.tsdf_bnds)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
            # acutally trilinear interpolation if mode = 'bilinear'
        tsdf_value = F.grid_sample(tsdf_volume.to(device), vgrid.to(device), padding_mode='border', align_corners=True,
                            mode='bilinear').squeeze(-1).squeeze(-1)
        return tsdf_value


    def eval_points_tsdf(self, p, tsdf_volume, device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            

        Returns:
            ret (tensor): tsdf value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)
        tsdf_vals = []
        for pi in p_split:
            pi = pi.unsqueeze(0)
            tsdf_volume_tensor = tsdf_volume

            tsdf_val = self.sample_grid_tsdf(pi, tsdf_volume_tensor, device)
            tsdf_val = tsdf_val.squeeze(0)
            tsdf_vals.append(tsdf_val)

        tsdf_values = torch.cat(tsdf_vals, dim=1)
        return tsdf_values


    def render_batch_ray(self, c, decoders, rays_d, rays_o, device, tsdf_volume, tsdf_bnds, stage, gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            tsdf_volume (tensor): tsdf volume.
            tsdf_bnds (tensor): tsdf volume bounds.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
            weight (tensor): attention weight.
        """
        eval_tsdf_volume = tsdf_volume
        

        N_samples = self.N_samples
        N_surface = self.N_surface
        N_importance = self.N_importance

        N_rays = rays_o.shape[0]

        if gt_depth is None:
            N_surface = 0
            near = 0.01
        else:
            gt_depth = gt_depth.reshape(-1, 1)
            gt_depth_samples = gt_depth.repeat(1, N_samples)
            near = gt_depth_samples*0.01

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) -
                 det_rays_o)/det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        if gt_depth is not None:
            # in case the bound is too large
            far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))

        else:
            far = far_bb
        if N_surface > 0:
            if False:
                # this naive implementation downgrades performance
                gt_depth_surface = gt_depth.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).to(device)
                z_vals_surface = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
            else:
                # since we want to colorize even on regions with no depth sensor readings,
                # meaning colorize on interpolated geometry region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample near the surface,
                # since it is not a good idea to sample 16 points near (half even behind) camera,
                # for pixels with zero depth value, we sample uniformly from camera to max_depth.
                gt_none_zero_mask = gt_depth > 0
                gt_none_zero = gt_depth[gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).double().to(device)
                # emperical range 0.05*depth
                z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
                z_vals_surface = torch.zeros(
                    gt_depth.shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask,
                               :] = z_vals_surface_depth_none_zero
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                z_vals_surface_depth_zero = near_surface * \
                    (1.-t_vals_surface) + far_surface * (t_vals_surface)
                z_vals_surface_depth_zero.unsqueeze(
                    0).repeat((~gt_none_zero_mask).sum(), 1)
                z_vals_surface[~gt_none_zero_mask,
                               :] = z_vals_surface_depth_zero

        t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        if N_surface > 0:
            z_vals, _ = torch.sort(
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
        pointsf = pts.reshape(-1, 3)
        
        raw, weight = self.eval_points(pointsf, decoders, tsdf_volume, tsdf_bnds, c, stage, device)
        raw = raw.reshape(N_rays, N_samples+N_surface, -1)
        weight = weight.reshape(N_rays, N_samples+N_surface, -1)


        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
        
        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + \
                rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            
            raw, weight = self.eval_points(pointsf, decoders, tsdf_volume, tsdf_bnds, c, stage, device)
            raw = raw.reshape(N_rays, N_samples+N_surface, -1)
            weight = weight.reshape(N_rays, N_samples+N_surface, -1)

            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
            return depth, uncertainty, color, weight


        return depth, uncertainty, color, weight


    def render_img(self, c, decoders, c2w, device, tsdf_volume, tsdf_bnds, stage, gt_depth=None):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            tsdf_volume (tensor): tsdf volume.
            tsdf_bnds (tensor): tsdf volume bounds.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(
                H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            uncertainty_list = []
            color_list = []


            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]

                iter = 10

                if gt_depth is None:
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, tsdf_volume, tsdf_bnds, stage, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, tsdf_volume, tsdf_bnds, stage, gt_depth=gt_depth_batch)

                depth, uncertainty, color, _= ret

                
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)
         
        



            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)
     
            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)

            return depth, uncertainty, color 


 