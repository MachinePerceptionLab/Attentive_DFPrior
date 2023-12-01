# adapted from https://github.com/zju3dv/manhattan_sdf
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import torch
import glob
import os
import pyrender
import os
from tqdm import tqdm
from pathlib import Path
import sys 
sys.path.append(".") 
from src import config
from src.utils.datasets import get_dataset
import argparse

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Chamfer': (np.mean(dist1) + np.mean(dist2))/2,
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics



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

# load pose
def get_pose(cfg, args):
    scale = cfg['scale']
    H, W, fx, fy, cx, cy = update_cam(cfg)
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy).intrinsic_matrix # (3, 3)

    frame_reader = get_dataset(cfg, args, scale)

    pose_ls = []
    for idx in range(len(frame_reader)):
        if idx % 10 != 0: continue
        _, gt_color, gt_depth, gt_c2w = frame_reader[idx]

        c2w = gt_c2w.cpu().numpy()

        if np.isfinite(c2w).any():

            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            pose_ls.append(c2w)

    return pose_ls, K, H, W


class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])

        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]]) # [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


def refuse(mesh, poses, K, H, W, cfg):
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3*0.01, 
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    idx = 0
    H, W, fx, fy, cx, cy = update_cam(cfg)
    
    for pose in tqdm(poses):

        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)

        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

def evaluate_mesh():
    """
    Evaluate the scannet mesh.

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

    scene_id = cfg['data']['id']

    input_file = f"output/scannet/scans/scene{scene_id:04d}_00/mesh/final_mesh.ply"  
    mesh = trimesh.load_mesh(input_file)
    mesh.invert() # change noraml of mesh

    poses, K, H, W = get_pose(cfg, args)

    # refuse mesh
    mesh = refuse(mesh, poses, K, H, W, cfg)

    # save mesh
    out_mesh_path = f"output/scannet/scans/scene{scene_id:04d}_00/mesh/final_mesh_refused.ply"
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)  

    mesh = trimesh.load(out_mesh_path)
    gt_mesh = os.path.join("./Datasets/scannet/GTmesh_lowres", f"{scene_id:04d}_00.obj")
    gt_mesh = trimesh.load(gt_mesh)
    
    metrics = evaluate(mesh, gt_mesh)
    print(metrics)      


if __name__ == "__main__":
    evaluate_mesh()