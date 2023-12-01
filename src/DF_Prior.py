import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

# import src.fusion as fusion
# import open3d as o3d

torch.multiprocessing.set_sharing_strategy('file_system')


class DF_Prior():
    """
    DF_Prior main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args

        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        model = config.get_model(cfg)
        self.shared_decoders = model
     
        self.scale = cfg['scale']

        self.load_bound(cfg)
        self.load_pretrain(cfg)
        self.grid_init(cfg)

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()

        dataset = self.cfg['data']['dataset']
        scene_id = self.cfg['data']['id']
        self.scene_id = scene_id
        print(scene_id)
        # load tsdf grid
        if dataset == 'scannet':
            self.tsdf_volume_shared = torch.load(f'scannet_tsdf_volume/scene{scene_id}_tsdf_volume.pt') 
        elif dataset == 'replica':
            self.tsdf_volume_shared = torch.load(f'replica_tsdf_volume/{scene_id}_tsdf_volume.pt')
        self.tsdf_volume_shared = self.tsdf_volume_shared.to(self.cfg['mapping']['device'])
        self.tsdf_volume_shared.share_memory_()

        # load tsdf grid bound
        if dataset == 'scannet':
            self.tsdf_bnds = torch.load(f'scannet_tsdf_volume/scene{scene_id}_bounds.pt')
        elif dataset == 'replica':
            self.tsdf_bnds = torch.load(f'replica_tsdf_volume/{scene_id}_bounds.pt')
        self.tsdf_bnds = torch.tensor(self.tsdf_bnds).to(self.cfg['mapping']['device'])
        self.tsdf_bnds.share_memory_()

        self.vol_bnds = self.tsdf_bnds
        self.vol_bnds.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()
        for key, val in self.shared_c.items():
            val = val.to(self.cfg['mapping']['device'])
            val.share_memory_()
            self.shared_c[key] = val
        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()
        self.renderer = Renderer(cfg, args, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()


    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")


    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']


    # def init_tsdf_volume(self, cfg, args):
    #     # scale the bound if there is a global scaling factor
    #     scale = cfg['scale']
    #     bound = torch.from_numpy(
    #         np.array(cfg['mapping']['bound'])*scale)
    #     bound_divisible = cfg['grid_len']['bound_divisible']
    #     # enlarge the bound a bit to allow it divisible by bound_divisible
    #     bound[:, 1] = (((bound[:, 1]-bound[:, 0]) /
    #                     bound_divisible).int()+1)*bound_divisible+bound[:, 0]
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(self.W, self.H, self.fx, self.fy, self.cx, self.cy).intrinsic_matrix # (3, 3)

    #     print("Initializing voxel volume...")
    #     vol_bnds = np.array(bound)
    #     tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=4/256) 


    #     return tsdf_vol, intrinsic, vol_bnds


    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound'])*self.scale)
        bound_divisible = cfg['grid_len']['bound_divisible']
        # enlarge the bound a bit to allow it divisible by bound_divisible
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_divisible).int()+1)*bound_divisible+self.bound[:, 0]
        self.shared_decoders.bound = self.bound
        self.shared_decoders.low_decoder.bound = self.bound
        self.shared_decoders.high_decoder.bound = self.bound
        self.shared_decoders.color_decoder.bound = self.bound
            

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        ckpt = torch.load(cfg['pretrained_decoders']['low_high'],
                          map_location=cfg['mapping']['device'])
        low_dict = {}
        high_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    low_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    high_dict[key] = val
        self.shared_decoders.low_decoder.load_state_dict(low_dict)
        self.shared_decoders.high_decoder.load_state_dict(high_dict)


    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids.

        Args:
            cfg (dict): parsed config dict.
        """
        
        low_grid_len = cfg['grid_len']['low']
        self.low_grid_len = low_grid_len
        high_grid_len = cfg['grid_len']['high']
        self.high_grid_len = high_grid_len
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        c = {}
        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1]-self.bound[:, 0]


    
        low_key = 'grid_low'
        low_val_shape = list(map(int, (xyz_len/low_grid_len).tolist()))
        low_val_shape[0], low_val_shape[2] = low_val_shape[2], low_val_shape[0]
        self.low_val_shape = low_val_shape
        val_shape = [1, c_dim, *low_val_shape]
        low_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[low_key] = low_val

        high_key = 'grid_high'
        high_val_shape = list(map(int, (xyz_len/high_grid_len).tolist()))
        high_val_shape[0], high_val_shape[2] = high_val_shape[2], high_val_shape[0]
        self.high_val_shape = high_val_shape
        val_shape = [1, c_dim, *high_val_shape]
        high_val = torch.zeros(val_shape).normal_(mean=0, std=0.0001)
        c[high_key] = high_val

        color_key = 'grid_color'
        color_val_shape = list(map(int, (xyz_len/color_grid_len).tolist()))
        color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        self.color_val_shape = color_val_shape
        val_shape = [1, c_dim, *color_val_shape]
        color_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[color_key] = color_val

        self.shared_c = c


    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while (1):
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()


    def mapping(self, rank):
        """
        Mapping Thread. (updates low, high, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()


    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
