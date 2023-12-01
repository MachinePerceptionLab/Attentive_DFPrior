import argparse
import os
import numpy as np
import torch

from src import config
import src.fusion as fusion
import open3d as o3d
from src.utils.datasets import get_dataset


def update_cam(cfg):
    """
    Update the camera intrinsics according to pre-processing config, 
    such as resize or edge crop.
    """
    H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
    # resize the input images to crop_size (variable name used in lietorch)
    if 'crop_size' in cfg['cam']:
        crop_size = cfg['cam']['crop_size']
        H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        sx = crop_size[1] / W
        sy = crop_size[0] / H
        fx = sx*fx
        fy = sy*fy
        cx = sx*cx
        cy = sy*cy
        W = crop_size[1]
        H = crop_size[0]

        
    # croping will change H, W, cx, cy, so need to change here
    if cfg['cam']['crop_edge'] > 0:
        H -= cfg['cam']['crop_edge']*2
        W -= cfg['cam']['crop_edge']*2
        cx -= cfg['cam']['crop_edge']
        cy -= cfg['cam']['crop_edge']
    
    return H, W, fx, fy, cx, cy


def init_tsdf_volume(cfg, args, space=10):
    """
    Initialize the TSDF volume.
    Get the TSDF volume and bounds.

    space: the space between frames to integrate into the TSDF volume.

    """
    # scale the bound if there is a global scaling factor
    scale = cfg['scale']
    bound = torch.from_numpy(
        np.array(cfg['mapping']['bound'])*scale)
    bound_divisible = cfg['grid_len']['bound_divisible']
    # enlarge the bound a bit to allow it divisible by bound_divisible
    bound[:, 1] = (((bound[:, 1]-bound[:, 0]) /
                        bound_divisible).int()+1)*bound_divisible+bound[:, 0]

    # TSDF volume
    H, W, fx, fy, cx, cy = update_cam(cfg)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy).intrinsic_matrix # (3, 3)

    print("Initializing voxel volume...")
    vol_bnds = np.array(bound)
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=4.0/256) 

    frame_reader = get_dataset(cfg, args, scale)

    for idx in range(len(frame_reader)):
        if idx % space != 0: continue
        print(f'frame: {idx}')
        _, gt_color, gt_depth, gt_c2w = frame_reader[idx]

        # convert to open3d camera pose
        c2w = gt_c2w.cpu().numpy()

        if np.isfinite(c2w).any():            
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0

            depth = gt_depth.cpu().numpy() # (368, 496, 3)
            color = gt_color.cpu().numpy()
            depth = depth.astype(np.float32)
            color = np.array((color * 255).astype(np.uint8))
            tsdf_vol.integrate(color, depth, intrinsic, c2w, obs_weight=1.)

    print('Getting TSDF volume')
    tsdf_volume, _, bounds = tsdf_vol.get_volume()

    print("Getting mesh")
    verts, faces, norms, colors = tsdf_vol.get_mesh()

    tsdf_volume = torch.tensor(tsdf_volume)
    tsdf_volume = tsdf_volume.reshape(1, 1, tsdf_volume.shape[0], tsdf_volume.shape[1], tsdf_volume.shape[2])
    tsdf_volume = tsdf_volume.permute(0, 1, 4, 3, 2)
    
    return tsdf_volume, bounds, verts, faces, norms, colors

def get_tsdf():
    """
    Save the TSDF volume and bounds to a file.
    
    """
    parser = argparse.ArgumentParser(
        description='Arguments for running the code.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--space', type=int, default=10, help='the space between frames to integrate into the TSDF volume.')

    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/df_prior.yaml')

    dataset = cfg['data']['dataset']
    scene_id = cfg['data']['id']

    
    path = f'{dataset}_tsdf_volume'
    os.makedirs(path, exist_ok=True)

    tsdf_volume, bounds, verts, faces, norms, colors = init_tsdf_volume(cfg, args, space=args.space)

    if dataset == 'scannet':
        tsdf_volume_path = os.path.join(path, f'scene{scene_id}_tsdf_volume.pt')
        bounds_path = os.path.join(path, f'scene{scene_id}_bounds.pt')

    elif dataset == 'replica':
        tsdf_volume_path = os.path.join(path, f'{scene_id}_tsdf_volume.pt')
        bounds_path = os.path.join(path, f'{scene_id}_bounds.pt')


    torch.save(tsdf_volume, tsdf_volume_path)
    torch.save(bounds, bounds_path)



if __name__ == '__main__':
    get_tsdf()
