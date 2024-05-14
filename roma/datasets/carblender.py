from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as tvf
from tqdm import tqdm
import pyexr
from pathlib import Path
import warnings
from argparse import ArgumentParser
import json
from skimage import metrics
from itertools import combinations
from roma.utils import get_depth_tuple_transform_ops, get_tuple_transform_ops
from roma.utils import *
from torch.utils.data import Dataset

class CarBlenderScene(Dataset):
    def __init__(
        self,
        data_root,
        ht=1456,
        wt=1932,
        min_overlap=0.0,
        max_overlap=1.0,
        shake_t=0,
        compute_overlaps=False,
        normalize=True,
        max_num_pairs = 100_000,
        scene_name = None,
        use_horizontal_flip_aug = False,
        colorjiggle_params = None,
        random_eraser = None,
    ) -> None:
        
        self.data_root = Path(data_root)
        self.scene_path = self.data_root / scene_name
        
        parser = ArgumentParser(description='Render a scene with multiple cameras')
        parser.add_argument('--num_cameras', type=int, default=36, help='Number of cameras to render per level')
        parser.add_argument('--num_levels', type=int, default=5, help='Number of levels to render')
        parser.add_argument('--start_radius', type=float, default=8, help='Radius of the circle around the object')
        parser.add_argument('--base_path', type=str, default='./', help='Base path for output files')
        parser.add_argument('--obj_name', type=str, help='Name of the target object')
        parser.add_argument('--radius_decay', type=float, default=1.4, help='Decay rate of the radius for each level')
        parser.add_argument('--intrinsics_path', type=str, default='intrinsics.json', help='Path to save camera intrinsics file')
        parser.add_argument('--extrinsics_path', type=str, default='cameras.json', help='Path to save camera extrinsics file')
        parser.add_argument('--jitter', action='store_true', help='Add jitter to camera positions')
        parser.add_argument('--z_offset', type=float, default=0.8, help='Offset of the camera from the object')
        parser.add_argument('--z_scale_multiplier', type=float, default=1.4, help='Multiplier for the z offset')
        parser.add_argument('--camera_rotation', type=float, default=180, help='Rotation of the camera around the object in degrees')
        parser.add_argument('--test', action='store_true', help='Run in test mode')
        parser.add_argument('--layer_name', type=str, default='ViewLayer', help='Name of the view layer to use for rendering')
        parser.add_argument('--resolution_x', type=int, default=1920, help='Resolution of the rendered image in x')
        parser.add_argument('--resolution_y', type=int, default=1440, help='Resolution of the rendered image in y')
        
        settings_path = self.scene_path / 'settings.json'
        settings = json.load(open(settings_path))
        scene_args = parser.parse_args(settings['settings'])
        
        self.num_cameras = scene_args.num_cameras
        self.num_levels = scene_args.num_levels
        
        self.image_paths = list((self.scene_path / 'image').rglob('*.png')) + list((self.scene_path / 'image').rglob('*.jpg'))
        self.depth_paths = list((self.scene_path / 'depth').rglob('*.exr'))
        
        self.image_paths = sorted(self.image_paths)
        self.depth_paths = sorted(self.depth_paths)
        
        self.cam_info = np.load(self.scene_path / 'cameras.npz')
        self.second_cam_info = np.load(self.scene_path / 'cams.npz')
        self.poses = self.load_poses()
        self.intrinsics = self.second_cam_info['intrinsics']
        self.pairs = self.create_pairs()
        self.clip_start = self.cam_info['clip_start']
        self.clip_end = self.cam_info['clip_end']
        
        if compute_overlaps:
            self.overlaps = np.array(
                [self.compute_overlap(*pair) for pair in self.pairs]
            )
        else:
            self.overlaps = np.ones(len(self.pairs))
        
        threshold = (self.overlaps >= min_overlap) & (self.overlaps <= max_overlap)
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]
    
        
        if len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), max_num_pairs, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        
        
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt), normalize=normalize, colorjiggle_params = colorjiggle_params,
        )
        
        self.depth_transform_ops = get_depth_tuple_transform_ops(
                resize=(ht, wt)
            )
        self.wt, self.ht = wt, ht
        self.shake_t = shake_t
        self.random_eraser = random_eraser
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        

    def load_im(self, im_path):
        im = Image.open(im_path)
        ## remove alpha channel
        if im.mode == 'RGBA':
            im = im.convert('RGB')
        return im
    
    def compute_overlap(self, idx1, idx2):
        ## compute overlap between two images via ssim
        im1 = self.load_im(self.image_paths[idx1])
        im2 = self.load_im(self.image_paths[idx2])
        im1 = np.array(im1)
        im2 = np.array(im2)
        sim =  metrics.structural_similarity(im1, im2,full=True)
        return sim[0]
    
    def create_pairs(self):
        pairs = list(combinations(range(len(self.image_paths)), 2))
        return np.array(pairs)
        
    
    def load_poses(self):
        extrinsics = self.second_cam_info['extrinsics']
        ## current shape is B x 3 x 4
        ## convert to B x 4 x 4
        
        extrinsics = np.concatenate([extrinsics, np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(extrinsics.shape[0], axis=0)], axis=1)
        
        return extrinsics
    
    def load_intrinsics(self):
        fx, fy, ux, uy = self.cam_info['intrinsics']
        length = self.num_cameras * self.num_levels
        intrinsics = np.zeros((length, 3, 3))
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = ux
        intrinsics[:, 1, 2] = uy
        intrinsics[:, 2, 2] = 1

        return intrinsics
       
    
    def horizontal_flip(self, im_A, im_B, depth_A, depth_B,  K_A, K_B):
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        depth_A, depth_B = depth_A.flip(-1), depth_B.flip(-1) 
        flip_mat = torch.tensor([[-1, 0, self.wt],[0,1,0],[0,0,1.]]).to(K_A.device)
        K_A = flip_mat@K_A  
        K_B = flip_mat@K_B  
        
        return im_A, im_B, depth_A, depth_B, K_A, K_B
    
    def load_depth(self, depth_ref, crop=None):
        depth = pyexr.read(depth_ref).squeeze()
        ## clip depth from 
        depth = np.clip(depth, self.clip_start, self.clip_end)
        return torch.from_numpy(depth)

    def __len__(self):
        return len(self.pairs)

    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=2)
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def __getitem__(self, pair_idx):
        # read intrinsics of original size
        idx1, idx2 = self.pairs[pair_idx]
        
        K1 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(3, 3)
        K2 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T1 = self.poses[idx1]
        T2 = self.poses[idx2]
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[
            :4, :4
        ]  # (4, 4)

        # Load positive pair data
        im_A, im_B = self.image_paths[idx1], self.image_paths[idx2]
        depth1, depth2 = self.depth_paths[idx1], self.depth_paths[idx2]
        
        im_A = self.load_im(im_A)
        im_B = self.load_im(im_B)
        K1 = self.scale_intrinsic(K1, im_A.width, im_A.height)
        K2 = self.scale_intrinsic(K2, im_B.width, im_B.height)

        depth_A = self.load_depth(depth1)
        depth_B = self.load_depth(depth2)
        # Process images
        im_A, im_B = self.im_transform_ops((im_A, im_B))
        depth_A, depth_B = self.depth_transform_ops(
            (depth_A[None, None], depth_B[None, None])
        )
        
        [im_A, im_B, depth_A, depth_B], t = self.rand_shake(im_A, im_B, depth_A, depth_B)
        K1[:2, 2] += t
        K2[:2, 2] += t
        
        im_A, im_B = im_A[None], im_B[None]
        if self.random_eraser is not None:
            im_A, depth_A = self.random_eraser(im_A, depth_A)
            im_B, depth_B = self.random_eraser(im_B, depth_B)
                
        if self.use_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_A, im_B, depth_A, depth_B, K1, K2 = self.horizontal_flip(im_A, im_B, depth_A, depth_B, K1, K2)
        
        data_dict = {
            "im_A": im_A[0],
            "im_A_identifier": self.image_paths[idx1].stem,
            "im_B": im_B[0],
            "im_B_identifier": self.image_paths[idx2].stem,
            "im_A_depth": depth_A[0, 0],
            "im_B_depth": depth_B[0, 0],
            "K1": K1,
            "K2": K2,
            "T_1to2": T_1to2,
            "im_A_path": im_A,
            "im_B_path": im_B,
            
        }
        return data_dict
    
    def __repr__(self):
        return f"CarBlenderScene {self.scene_path.stem} with {len(self)} pairs"
    
    def __str__(self):
        return self.__repr__()

class CarBlenderBuilder:
    def __init__(self, data_root='/home/azureuser/cloudfiles/code/Users/tosin/blender_data/data', ht=728, wt=966, min_overlap=0.0, max_overlap=1.0, shake_t=0, compute_overlaps=False, normalize=True, max_num_pairs = 100_000, use_horizontal_flip_aug = False, colorjiggle_params = None, random_eraser = None):
        self.data_root = Path(data_root)
        self.scene_names = [x.name for x in self.data_root.iterdir() if x.is_dir()]
        self.scenes = []
        self.ht = ht
        self.wt = wt
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.shake_t = shake_t
        self.compute_overlaps = compute_overlaps
        self.normalize = normalize
        self.max_num_pairs = max_num_pairs
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.colorjiggle_params = colorjiggle_params
        self.random_eraser = random_eraser
        
    def build(self, start=0, end=None):
        end = end or len(self.scene_names)
        for scene_name in tqdm(self.scene_names[start:end], desc="Building Scenes"):
            self.scenes.append(self.build_scene(scene_name))
        return self.scenes

    def build_scene(self, scene_name):
        return CarBlenderScene(
            data_root=self.data_root,
            ht=self.ht,
            wt=self.wt,
            min_overlap=self.min_overlap,
            max_overlap=self.max_overlap,
            shake_t=self.shake_t,
            compute_overlaps=self.compute_overlaps,
            normalize=self.normalize,
            max_num_pairs = self.max_num_pairs,
            scene_name = scene_name,
            use_horizontal_flip_aug = self.use_horizontal_flip_aug,
            colorjiggle_params = self.colorjiggle_params,
            random_eraser = self.random_eraser
        )
        
    def __len__(self):
        if len(self.scenes) == 0:
            warnings.warn("No scenes have been loaded, returning 0")
        return len(self.scenes)
    
    def __getitem__(self, idx):
        return self.scenes[idx]
    
    def __iter__(self):
        return iter(self.scenes)
    
    def __repr__(self):
        return f"SceneBuilder with {len(self)} scenes"
    
    def __str__(self):
        return self.__repr__()
    
    @property
    def scene_items_length(self):
        return sum([len(scene) for scene in self.scenes])
    
    