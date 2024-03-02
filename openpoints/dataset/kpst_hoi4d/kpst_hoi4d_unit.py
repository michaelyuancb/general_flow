
import numpy as np
import torch
import glob
import json
import pdb
import sys
import os
import pickle
import logging
import random
import open3d as o3d
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial import distance
from util import save_pickle, load_pickle
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPModel


from ..data_util import crop_pc
from ..build import DATASETS
from ...transforms.point_transform_cpu import PointsToTensor


def preprocess_scale_trajectory(traj, scale_method='NULL', eps=1e-6):
    # traj: (Q, T, 3), pos[t], numpy.array
    traj = traj[:, 1:, :] - traj[:, :-1, :]  # (Q, T-1, 3), pos[t] - pos[t-1]
    n_q = traj.shape[0]
    if scale_method == 'TLN':       # total length normalization 
        scale = np.sum(np.linalg.norm(traj, axis=-1), axis=-1, keepdims=True)           # (Q, 1)
        div_scale = scale[..., np.newaxis]
    elif scale_method == 'TDN':     # total distance normalization
        scale = np.linalg.norm(traj.cumsum(axis=-2)[:, -1, :], axis=-1, keepdims=True)  # (Q, 1)
        div_scale = scale[..., np.newaxis]
    elif scale_method == 'SDN':
        scale = np.linalg.norm(traj, axis=-1)                                           # (Q, length)
        div_scale = scale[..., np.newaxis]
    else:
        raise ValueError(f"scale_method need to be in ['TLN', 'TDN', 'SDN'],"
                         f"but get scale_method={scale_method}")
    static_idx = div_scale < eps                                  # (Q, length, 1)
    div_scale[static_idx] = 1             
    static_idx = np.repeat(static_idx, traj.shape[1] // static_idx.shape[1], axis=1)
    static_idx = np.repeat(static_idx, traj.shape[2] // static_idx.shape[2], axis=2)                                                             
    traj[static_idx] = 0                                          # (Q, length, 3), we set traj < esp to 0 directly. 
    sstep = (traj / div_scale).reshape(n_q, -1)                   # (Q, length*3)
    return sstep, scale


@DATASETS.register_module()
class KPST_HOI4D_Unit(Dataset):

    category_class = ["Toy Car", "Mug", "Laptop", "Storage Furniture", "Bottle", "Safe",
                      "Bowl", "Bucket", "Scissors", "Pliers", "Kettle", "Knife", "Trash Can",
                      "Lamp", "Stapler", "Chair"]
    
    color_mean = [0.46259782, 0.46253258, 0.46253258]
    color_std =  [0.693565  , 0.6852543 , 0.68061745]
    voxel_downsample_bar = 0.02
   
    def __init__(self,
                 data_root='datasets/HOI4D_KPST',
                 processed_root='datasets/HOI4D_KPST_processed',
                 split='train',
                 choose_class=None,
                 n_query=64, 
                 voxel_size=0.02,
                 voxel_max=None,
                 transform=None,
                 variable=False,  
                 scale_method=None,
                 unit_r=0.4,
                 aug_cluster=[1.0, 0, 0],               # random, nearest, not used.
                 aug_hand_mask_probe=[1.0, 0.0, 0.0],   # without, full, random-mask (with hand_radius)
                 aug_hand_mask_radius=0.1,
                 eval_with_hand=False,
                 balance_n_cluster=0,
                 balance_temperature=1.0,
                 ):
        # We load all data before experiment since 512G CPU Mem is large enough.

        super().__init__()
        print(f"Eval_MODE: WITH_HAND={eval_with_hand}")
        self.data_root = data_root
        self.processed_root = processed_root
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.variable = variable
        self.pipe_transform = PointsToTensor() 
        self.eval_with_hand = eval_with_hand

        if np.abs(sum(aug_hand_mask_probe)-1.0) > 1e-5:
            raise ValueError(f"aug_cluster must sum to 1.0, but get {aug_hand_mask_probe}, sum={sum(aug_hand_mask_probe)}")
        if np.abs(sum(aug_cluster)-1.0) > 1e-5:
            raise ValueError(f"aug_cluster must sum to 1.0, but get {aug_cluster}, sum={sum(aug_cluster)}")
        self.aug_cluster = []
        self.aug_hand_mask_probe = []
        self.n_aug = 0 
        aug_acc, aug_mask_acc = 0, 0 
        for i in range(len(aug_cluster)):
            aug_acc = aug_acc + aug_cluster[i]
            aug_mask_acc = aug_mask_acc + aug_hand_mask_probe[i]
            self.aug_cluster.append(aug_acc)
            self.aug_hand_mask_probe.append(aug_mask_acc)
            if aug_acc == 1.0 and aug_mask_acc == 1.0:
                break 
        self.n_aug = len(self.aug_cluster)
        self.aug_cluster = [0] + self.aug_cluster
        self.aug_hand_mask_probe = [0] + self.aug_hand_mask_probe
        self.aug_hand_mask_radius = aug_hand_mask_radius

        if not os.path.exists(processed_root):
            os.makedirs(processed_root)
        self.choose_class = choose_class

        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be in ['train', 'val', 'test'], but split = {split}")

        with open(os.path.join(data_root, 'metadata.json'), "r") as fp:
            self.metadata = json.load(fp)[split]
    
        self.n_data = len(self.metadata)
        logging.info("Totally {} samples in {} set.".format(self.n_data, split))

        self.dtraj_list = []
        self.part_list = []
        self.n_query = n_query
        n_less_query = 0
        print(f"N-Query={self.n_query} for {self.split}")

        ############################################# Original Trajectory Label #########################################
        # pdb.set_trace()
        filename_traj = os.path.join(processed_root, f'kpst_hoi4d_{split}_traj.pkl')

        if not os.path.exists(filename_traj):
            for meta in tqdm(self.metadata, desc=f'Loading HOI4D_KPST {split} split - Trajectory Label'):
                data_fp = os.path.join(data_root, 'data', str(meta['id']))
                traj = np.load(os.path.join(data_fp, 'kpst_traj.npy'))                  # (N, T=5, 3)
                part = np.load(os.path.join(data_fp, 'kpst_part_id.npy')).reshape(-1)   # (N, )
                self.dtraj_list.append(traj)                                            # (N, T=5, 3)
                self.part_list.append(part)                                             # (N, )
                # pdb.set_trace()
            
            traj_pkl = {"dtraj": self.dtraj_list, "part": self.part_list}
            save_pickle(filename_traj, traj_pkl)
        else:
            traj_pkl = load_pickle(filename_traj)
            self.dtraj_list = traj_pkl['dtraj']
            self.part_list = traj_pkl['part']
            logging.info(f"Load HOI4D_KPST {split} split trajectory successfully.")

        assert self.n_data == len(self.dtraj_list)
        assert len(self.dtraj_list) == len(self.part_list)

        for part in self.part_list:
            if part.shape[0] < self.n_query:
                n_less_query = n_less_query + 1

        logging.warning(f"[Total: {self.n_data} samples], [Less-n_query: {n_less_query} samples]")
        
        self.balance_n_cluster = balance_n_cluster
        self.balance_temperature = balance_temperature
        filename_traj_balance = os.path.join(processed_root, f'kpst_hoi4d_{split}_traj_balance_{balance_n_cluster}_{balance_temperature}.pkl')
        self.balance_idx_list = []
        if not os.path.exists(filename_traj_balance):
            for i in tqdm(range(len(self.dtraj_list))):
                if balance_n_cluster > 0:
                    dtraj, part = self.dtraj_list[i], self.part_list[i]  # (Q, T, 3)
                    dist = np.sum(np.linalg.norm(dtraj[:, 1:] - dtraj[:, :-1], axis=-1), axis=-1)   # (Q, )
                    balance_idx = self._balance_via_clustering(dist, n_cluster=balance_n_cluster)
                else:
                    balance_idx = np.arange(self.dtraj_list[i].shape[0])
                self.balance_idx_list.append(balance_idx)
            save_pickle(filename_traj_balance, self.balance_idx_list)
            logging.info(f"Save HOI4D_KPST {split} split balance_idx_list successfully.")
        else: 
            self.balance_idx_list = load_pickle(filename_traj_balance)
            logging.info(f"Load HOI4D_KPST {split} split balance_idx_list successfully.")

        ############################################# Input PointCloud #########################################
        self.unit_r = unit_r
        data_fp_name = f'kpst_hoi4d_{split}_data_unit_{self.unit_r}_{voxel_size}_{voxel_max}.pkl'
        filename_data = os.path.join(processed_root, data_fp_name)
        self.kps_center = []
        if not os.path.exists(filename_data):
            np.random.seed(0)
            self.data = []
            for idx, meta in tqdm(enumerate(self.metadata), desc=f'Loading HOI4D_KPST {split} split for unit_r={unit_r}, voxel_size={voxel_size}, voxel_max={voxel_max}'):
                data_fp = os.path.join(data_root, 'data', str(meta['id']))
                spcd = np.load(os.path.join(data_fp, 'pcd.npy'))   
                coord = spcd[:, :3]   # (N, 3)  XYZ
                feat = spcd[:, 3:6]   # (N, 3)  RGB
                label = spcd[:, 6]    # (N, 1)  Mask-ID
                dtraj = self.dtraj_list[idx]
                coord, feat, label = self.kps_center_crop(coord, feat, label, dtraj) 
                cdata = np.hstack((coord, feat, np.expand_dims(label, -1))).astype(np.float32)
                self.data.append(cdata)
                
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename_data, 'wb') as f:
                pickle.dump(self.data, f)
                logging.info(f"{filename_data} saved successfully")
        else:
            with open(filename_data, 'rb') as f:
                self.data = pickle.load(f)
                logging.info(f"{filename_data} load successfully")

        self.traj_len = self.dtraj_list[0].shape[1]
        self.num_points = self.data[0].shape[0]
        
        ############################################# CLIP Text Features #########################################
        text_filename_data = os.path.join(processed_root, f'kpst_hoi4d_{split}_text.npy')

        if not os.path.exists(text_filename_data):
            clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_feats = []

            for meta in tqdm(self.metadata, desc=f'Get HOI4D_KPST CLIP-Features {split} split'):
                text = meta['action'] + ' ' + meta['object']
                texts = [text.lower()]
                
                inputs = clip_tokenizer(texts, padding=True, return_tensors="pt")
                text_features = clip_model.get_text_features(**inputs)              # (1, 512)
                text_features = text_features.detach().numpy().reshape(-1)

                clip_feats.append(text_features)

            self.clip_feats = np.concatenate(clip_feats, axis=0)
            np.save(text_filename_data, clip_feats)
        else:
            self.clip_feats = np.load(text_filename_data, allow_pickle=True)
            logging.info(f"{text_filename_data} load successfully")

        ############################################# Scaled Trajectory Label #########################################
        self.scale_method = scale_method
        if (scale_method is not None) and (scale_method not in ['TLN', 'TDN', 'SDN']):
            raise ValueError(f"scale_method need to be in ['TLN', 'TDN', 'SDN'],"
                             f"but get scale_method={scale_method}")
        if self.scale_method is not None:
            logging.info(f"Scale Trajectory with {scale_method}")
            scale_fp = os.path.join(processed_root, f'kpst_hoi4d_{split}_{scale_method}_ScaleTraj.pkl')
            self.sstep_list = []
            self.scale_list = []
            if not os.path.exists(scale_fp):
                for dtraj in tqdm(self.dtraj_list, desc=f'Get HOI4D_KPST Scale Trajectory {split} split'):
                    sstep, scale = preprocess_scale_trajectory(dtraj, scale_method=scale_method)
                    # pos[t] - pos[t-1], (Q, 3*(T-1)), (Q, F_scale)
                    self.sstep_list.append(sstep)
                    self.scale_list.append(scale)
                scale_info = {
                    'sstep': self.sstep_list,
                    'scale': self.scale_list
                }
                save_pickle(scale_fp, scale_info)
            else:
                scale_info = pickle.load(open(scale_fp, 'rb'))
                self.sstep_list = scale_info['sstep']
                self.scale_list = scale_info['scale']
                logging.info(f"{scale_fp} load successfully")

        ############################## Choose Specific Class ################################
        if self.choose_class is not None:
            cc_list = self.choose_class
            cc_list = ''.join(cc_list).split(',')
            self.choose_class = [' '.join(c.split('_')) for c in cc_list]
    
        pick_idx_list = []
        class_idx_info = dict()
        for i in range(len(self.metadata)):
            meta = self.metadata[i]
            object_ = meta['object']
            action_ = meta['action']
            if (self.choose_class is None) or (object_ in self.choose_class):
                if object_ not in class_idx_info.keys():
                    class_idx_info[object_] = {'idx': list(), 'action_idx': dict()}
                if action_ not in class_idx_info[object_]['action_idx'].keys():
                    class_idx_info[object_]['action_idx'][action_] = list()
                        
                picked_idx = len(pick_idx_list)
                class_idx_info[object_]['idx'].append(picked_idx)
                class_idx_info[object_]['action_idx'][action_].append(picked_idx)

                pick_idx_list.append(i)
        self.class_idx_info = class_idx_info

        def get_list_idx(l, idxs):
            return [l[i] for i in idxs]
            
        self.metadata = get_list_idx(self.metadata, pick_idx_list)
        self.n_data = len(self.metadata)
        self.data = get_list_idx(self.data, pick_idx_list)
        self.part_list = get_list_idx(self.part_list, pick_idx_list)
        self.dtraj_list = get_list_idx(self.dtraj_list, pick_idx_list)
        self.clip_feats = get_list_idx(self.clip_feats, pick_idx_list)
        self.balance_idx_list = get_list_idx(self.balance_idx_list, pick_idx_list)
        if self.scale_method is not None:
            self.sstep_list = get_list_idx(self.sstep_list, pick_idx_list)
            self.scale_list = get_list_idx(self.scale_list, pick_idx_list)
        print(f"Total n_sample_choose={self.n_data}.")


    def _balance_via_clustering(self, values, n_cluster):   # Scale Rebalance
        if values.shape[0] <= n_cluster:
            return np.arange(values.shape[0])
        
        values = values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(values)
        clusters = kmeans.labels_

        target_per_cluster_list = []
        for cluster in range(n_cluster):
            cnt = len(np.where(clusters == cluster)[0])
            if cnt > 0: target_per_cluster_list.append(cnt)
        tpc = np.array(target_per_cluster_list)
        max_target_per_cluster = tpc.max()
        max_cluster_idx = np.where(tpc == max_target_per_cluster)[0][0]
        max_cluster_size = tpc[max_cluster_idx]
        tpc_ratio = np.exp(tpc/np.sum(tpc)/self.balance_temperature) / np.sum(np.exp(tpc/np.sum(tpc)/self.balance_temperature))
        n_cluster = len(target_per_cluster_list)
        target_per_cluster_list = []
        for i in range(n_cluster):
            tpc_size = (tpc_ratio[i] / tpc_ratio[max_cluster_idx]) * max_cluster_size
            tpc_size = np.max([1, int(tpc_size)])
            target_per_cluster_list.append(tpc_size)

        balanced_idx = []
        for cluster in range(n_cluster):
            cluster_idx = np.where(clusters == cluster)[0]
            target_per_cluster = target_per_cluster_list[cluster]
            
            try:
                if len(cluster_idx) < target_per_cluster:
                    balanced_idx.extend(np.random.choice(cluster_idx, size=target_per_cluster, replace=True))
                else:
                    balanced_idx.extend(np.random.choice(cluster_idx, size=target_per_cluster, replace=False))
            except: 
                pdb.set_trace()

        return np.array(balanced_idx)


    def kps_center_crop(self, coord, feat, label, dtraj):       
        center = dtraj[:, 0, :].mean(0)   # (3, )
        is_available = (coord[:, 0] > center[0] - self.unit_r) & \
                       (coord[:, 0] < center[0] + self.unit_r) & \
                       (coord[:, 1] > center[1] - self.unit_r) & \
                       (coord[:, 1] < center[1] + self.unit_r) & \
                       (coord[:, 2] > center[2] - self.unit_r) & \
                       (coord[:, 2] < center[2] + self.unit_r)
        coord = coord[is_available]
        feat = feat[is_available]
        label = label[is_available]
        
        coord, feat, label = crop_pc(
            coord, feat, label, self.split, self.voxel_size, self.voxel_max, 
            variable=self.variable,
            voxel_downsample_bar=self.voxel_downsample_bar)

        return coord, feat, label


    def __len__(self):
        return self.n_data

    def _sample_query_cluster(self, n_cluster_aug, pos):     # QPS Augmentation
        if n_cluster_aug == 1:
            # Nearest Cluster Mode
            n_points = len(pos)
            if self.n_query > len(pos):
                sampled_indices = np.random.choice(n_points, size=self.n_query, replace=True).tolist()
            else:
                anchor = np.random.randint(0, len(pos))
                dist = np.linalg.norm(pos - pos[anchor], axis=1)
                idx = np.argsort(dist)
                sampled_indices = idx[:self.n_query].tolist()
        elif n_cluster_aug == 2:
            raise ValueError("cluster_aug_idx = 2 is not used.")
        
        return sampled_indices


    def _get_probe_int(self, acc_probe_list):
        p = random.random()
        ps = None
        for i in range(self.n_aug):
            if acc_probe_list[i] <= p and p <= acc_probe_list[i+1]:
                ps = i
                break
        if ps is None:
            raise ValueError(f"p={p}, aug_cluster={acc_probe_list}, but get None.")
        return ps
    

    def _sample_query(self, part_list, balance_idx, balance_pos):   # (Q, )
        sampled_indices = []
        
        if self.split == 'train':
            n_points = len(balance_idx)   
            # For Training, we sample query points randomly.
            aug_n = self._get_probe_int(self.aug_cluster)
            if aug_n == 0:
                if self.n_query >= n_points:
                    sampled_indices = np.random.choice(n_points, size=self.n_query, replace=True)
                else:
                    sampled_indices = np.random.choice(n_points, size=self.n_query, replace=False)
            else:
                sampled_indices = self._sample_query_cluster(aug_n, balance_pos)
            sampled_indices = balance_idx[sampled_indices]

        elif self.split == 'val':
            n_points = len(balance_idx)  
            # For Validation, we sample query points with fixed interval.
            n_loop = self.n_query // n_points
            sampled_indices = np.tile(np.arange(n_points), n_loop).tolist()
            extra_samples = self.n_query - n_points * n_loop
            if extra_samples > 0:
                interval = n_points // extra_samples
                extra_indices = np.arange(n_points)[::interval][:extra_samples]
                sampled_indices.extend(extra_indices)
            sampled_indices = balance_idx[sampled_indices]

        else:
            n_points = len(part_list)  
            # For Testing, we sample query points with interval according to ground truth part label.
            unique_classes, _ = np.unique(part_list, return_counts=True)
            samples_per_class = self.n_query // len(unique_classes)
            for cls_ in unique_classes:
                # pdb.set_trace()
                indices = np.where(part_list == cls_)[0]
                indices_len = len(indices)
                if indices_len < samples_per_class:
                    interval = 1
                else:
                    interval = indices_len // samples_per_class
                sampled_indices.extend(indices[::interval][:samples_per_class])

            extra_samples = self.n_query - len(sampled_indices)

            total_indices = np.arange(len(part_list))
            total_indices_len = len(total_indices)
            n_loop = extra_samples // total_indices_len
            sampled_indices.extend(total_indices.repeat(n_loop))
            extra_samples = extra_samples - total_indices_len * n_loop

            if extra_samples > 0:
                interval = total_indices_len // extra_samples
                extra_indices = total_indices[::interval][:extra_samples]
                sampled_indices.extend(extra_indices)
                
        return np.array(sampled_indices)
    
    
    def kpst_hand_mask_augmentation(self, coord, feat, label):       # HM Augmentation
        hand_idx = (label == 2).reshape(-1)
        n_points = coord.shape[0]
        if np.sum(hand_idx) == 0: return coord, feat, label
        if self.split == 'train':
            aug_hand_type = self._get_probe_int(self.aug_hand_mask_probe)
        else:
            if self.eval_with_hand is True:
                aug_hand_type = 1
            else:
                aug_hand_type = 0 
        if aug_hand_type == 0:   # without hand
            coord, feat, label = coord[~hand_idx], feat[~hand_idx], label[~hand_idx]
        elif aug_hand_type == 1:   # with full hand
            pass
        else:
            coord_h, feat_h, label_h = coord[hand_idx], feat[hand_idx], label[hand_idx]
            coord, feat, label = coord[~hand_idx], feat[~hand_idx], label[~hand_idx]
            anchor = np.random.randint(0, len(coord_h))
            dist = np.linalg.norm(coord_h-coord_h[anchor], axis=-1)
            hand_del_idx = dist < self.aug_hand_mask_radius
            if np.sum(hand_del_idx) < coord_h.shape[0]:
                coord_h, feat_h, label_h = coord_h[~hand_del_idx], feat_h[~hand_del_idx], label_h[~hand_del_idx]
                coord, feat = np.concatenate([coord, coord_h], axis=0), np.concatenate([feat, feat_h], axis=0)
                label = np.concatenate([label, label_h], axis=0)
            else:
                pass
        if coord.shape[0] < n_points:
            fill_idx = np.random.choice(coord.shape[0], size=n_points-coord.shape[0], replace=True).tolist()
            coord = np.concatenate([coord, coord[fill_idx]], axis=0)
            feat = np.concatenate([feat, feat[fill_idx]], axis=0)
            label = np.concatenate([label, label[fill_idx]], axis=0)
        return coord, feat, label


    def get_all_query(self, idx, kps_norm=True, origin=False):          # for inference & visualization (all_query_pont together)
        if idx > self.n_data:
            raise ValueError(f"idx must be less than n_data={self.n_data}, but idx = {idx}")
        if origin is False:
            coord, feat, label = np.split(self.data[idx], [3, 6], axis=1)
        else:
            meta = self.metadata[idx]
            data_fp = os.path.join(self.data_root, 'data', str(meta['id']))
            spcd = np.load(os.path.join(data_fp, 'pcd.npy'))                        # (N, 7)
            coord = spcd[:, :3]   # (N, 3)  XYZ 
            feat = spcd[:, 3:6]   # (N, 3)  RGB
            label = spcd[:, 6:7]
        
        dtraj = self.dtraj_list[idx]   # (Q, T=5, 3)
        coord, feat, label = self.kpst_hand_mask_augmentation(coord, feat, label)

        data = {'pos': coord.astype(np.float32),                           # (N, 3)  XYZ
                'x': feat.astype(np.float32),                              # (N, 3)  RGB
                'dtraj': dtraj.astype(np.float32),                         # (Q, T=5, 3)
                'text_feat': self.clip_feats[idx].astype(np.float32)       # (512)
                }
        if kps_norm is True:
            coord_norm = data['dtraj'][:, 0, :].mean(0)
        else:
            coord_norm = data['pos'].mean(0)

        if self.transform is not None:
            data = self.transform(data)

        desc = self.metadata[idx]['action'] + ' ' + self.metadata[idx]['object']
        data_fp = '_'.join(self.metadata[idx]['index'].split(' '))
        img_idx = self.metadata[idx]['img']
        if data_fp.startswith('recorded_rgbd'):
            data_fp = os.path.join('/home/ycb/kpst_aff/datasets/EgoSoft_DEMO', data_fp)
            rgb_fp = os.path.join(data_fp, f'rgb_{img_idx}.png')
            dep_fp = os.path.join(data_fp, f'depth_{img_idx}.png')
            camera_in = np.load(os.path.join(data_fp, 'camera_in.npy'))
            camera = o3d.camera.PinholeCameraIntrinsic()
            camera.set_intrinsics(1280, 720, camera_in[0,0], camera_in[1,1], camera_in[0,2], camera_in[1,2])
        else:
            data_fp = os.path.join('/public/datasets_yl/HOI4D', data_fp)
            rgb_fp = os.path.join(data_fp, 'align_rgb', str(img_idx).zfill(5)+'.jpg')
            dep_fp = os.path.join(data_fp, 'align_depth', str(img_idx).zfill(5)+'.png')
            camera = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault)

        depth = o3d.io.read_image(dep_fp)
        color_raw = o3d.io.read_image(rgb_fp)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth,convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera ,extrinsic=np.eye(4))
        pos = np.asarray(pcd.points)
        col = np.asarray(pcd.colors)
        spcd = np.concatenate([pos-coord_norm, col, np.zeros((pos.shape[0], 1))], axis=1)

        return data, spcd[:, :6], self.part_list[idx], desc


    def __getitem__(self, idx):
        data_idx = idx % self.n_data
        coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        balance_idx = self.balance_idx_list[idx]   # (Q, )
        query_idx = self._sample_query(self.part_list[idx], balance_idx, 
                                       self.dtraj_list[idx][balance_idx, 0, :])
        dtraj = self.dtraj_list[idx][query_idx]   # (Q, T=5, 3)

        coord, feat, label = self.kpst_hand_mask_augmentation(coord, feat, label)
        data = {'pos': coord.astype(np.float32),                           # (N, 3)  XYZ
                'x': feat.astype(np.float32),                              # (N, 3)  RGB
                'dtraj': dtraj.astype(np.float32),                         # (Q, T=5, 3)
                'text_feat': self.clip_feats[idx].astype(np.float32)       # (512)
                }
        
        if self.transform is not None:
            data = self.transform(data)  

        pack = dict()
        data['pack'] = pack

        return data