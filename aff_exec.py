import argparse
import os
import pdb
import time
import cv2
import copy
import torch
import numpy as np
import open3d as o3d
import scipy.ndimage
import matplotlib.pyplot as plt 
from transformers import CLIPTokenizer, CLIPModel

from openpoints.dataset.data_util import crop_pc
from openpoints.transforms import build_transforms_from_cfg
from inference import load_model, get_prediction
from util import save_pickle, load_pickle, load_easyconfig_from_yaml

from tool_repos.FastSAM.fastsam import FastSAM
from fastsam_prompt import FastSAMPrompt
from PIL import Image

# Input data-format:
# data = {
#     'kps_2d': np.array([[0, 0], [4, 2], [6, 6], [8, 12], [18, 25]]),  # (Q, 2)
#     'rgb': np.array(H, W, C)
#     'depth': np.array(H, W, 1)
# }


class KPSTExecutor(object):

    def __init__(self, args, cfg):
        if torch.cuda.is_available() is False:
            raise ValueError("Please use GPU for KPST Executor.")
        device = 'cuda'
        self.device = device

        # clip_model = None
        clip_model = {
            "tokenizer": CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32"),
            "model": CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        }
        self.clip_model = clip_model
        print("Finish Loading CLIP Model.")

        self.sam_model = FastSAM('./tool_repos/FastSAM/weights/FastSAM-X.pt')
        self.sam_prompt_model = FastSAMPrompt(device=device)

        self.camera = None
        self.args = args
        self.cfg = cfg
        self.device = device
        self.desc = 'None'
        self.desc_feat = None    # (1, 512)

        args.not_load = False  
        self.kpst_model = None
        self.unit_crop_r = None
        self.data_transform = build_transforms_from_cfg('test', cfg.datatransforms)
        self.set_kpst_model(args.pretrained_path)
        print("Finish Loading KPST Model.")
    
    def set_kpst_model(self, pretrained_path):
        self.args.cfg = '/'.join(pretrained_path.split('/')[:-2] + ['cfg.yaml'])
        cfg = load_easyconfig_from_yaml(self.args.cfg)
        self.print_kpst_model()
        if cfg.seed is None: cfg.seed = 0
        self.args.pretrained_path = pretrained_path
        self.kpst_model = load_model(self.args, cfg)
        self.kpst_model.to(self.device)
        self.kpst_model.eval()
        self.unit_crop_r = cfg.dataset.common.get('unit_r', None)
        self.args.voxel_max = cfg.dataset.test.voxel_max
        self.cfg = cfg 
        # Notice: voxel_size is not set by cfg, you need to run self.set_pcd_voxel_size() mannually.  

    def print_kpst_model(self):
        print(f"cfg_file_path={self.args.cfg}")
    
    def set_desc(self, desc):

        # self.desc = 'None'
        # self.desc_feat = np.random.randn(512)

        clip_open_safe_fp = 'results/exec_save/clip_feat'
        clip_open_safe_fp = os.path.join(clip_open_safe_fp, desc+'.npy')
        if not os.path.exists(clip_open_safe_fp):
            os.makedirs('results/exec_save/clip_feat', exist_ok=True)
            desc = ' '.join(desc.split('_'))
            self.desc = desc
            inputs = self.clip_model['tokenizer'](desc, padding=True, return_tensors="pt")
            text_features = self.clip_model['model'].get_text_features(**inputs)             # (1, 512)
            text_features = text_features.detach().numpy().reshape(-1)                       # (512)
            self.desc_feat = text_features
            np.save(clip_open_safe_fp, self.desc_feat)
        else:
            self.desc = desc
            print("load pre_clip_feature!")
            self.desc_feat = np.load(clip_open_safe_fp)

        print(f"Set Desc: {desc}.")

    
    def set_camera(self, camera, H=0, W=0):
        if isinstance(camera, o3d.camera.PinholeCameraIntrinsic):
            self.camera = camera
        else:
            if (H == 0) or (W == 0):
                raise ValueError(f"When set camera mannually, H != 0 and W != 0, but get (H,W)=({H},{W}).")
            self.camera = o3d.camera.PinholeCameraIntrinsic()
            self.camera.set_intrinsics(W, H, camera[0,0], camera[1,1], camera[0,2], camera[1,2])

            
    @staticmethod
    def display_and_capture_points(image, title='Default'):
        image_pil = Image.fromarray(np.uint8(image))
        points = []

        def onclick(event):
            ix, iy = event.xdata, event.ydata
            print(f'Point: ({ix}, {iy})')
            ax.plot(ix, iy, 'ro')
            fig.canvas.draw()
            points.append((ix, iy))
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.imshow(image_pil)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        points = np.array(points)        # (N, 2), (w, h)-format
        return points
    

    def segment_robot_body(self, image, robot_anchor, robot_anchor_label, vis_dir=None, commit=''):
        # robot_anchor: (N, 2), (h,w)-format, numpy

        # robot_anchor = np.array([[935, 1042]])
        # robot_anchor_label = np.ones(robot_anchor.shape[0]).astype(np.uint32)

        if robot_anchor is None:
            robot_anchor = KPSTExecutor.display_and_capture_points(image, title='robot mask')
            robot_anchor = [[int(x[1]), int(x[0])] for x in robot_anchor]   # (w, h) --> (h, w)
            robot_anchor = np.array(robot_anchor)
            robot_anchor_label = np.ones(robot_anchor.shape[0]).astype(np.uint32)

        ra = np.concatenate([robot_anchor[:, 1:2].copy(), robot_anchor[:, 0:1].copy()], axis=-1) # (h, w) --> (w, h)

        if self.args.seg_downsample_ratio > 1:
            img_pil = Image.fromarray(image)
            new_height = img_pil.height // self.args.seg_downsample_ratio
            new_width = img_pil.width // self.args.seg_downsample_ratio
            resized_img = np.asarray(img_pil.resize((new_width, new_height)))
            ra = ra // self.args.seg_downsample_ratio
        else:
            resized_img = image

        everything_results = self.sam_model(resized_img, device=self.device, retina_masks=True, imgsz=1024, 
                                            conf=0.4, iou=0.9,)
        self.sam_prompt_model.set_image_result(resized_img, everything_results)
        mask = self.sam_prompt_model.point_prompt(ra, robot_anchor_label)
        
        if self.args.seg_downsample_ratio > 1:  
            mask = scipy.ndimage.zoom(mask, (1, self.args.seg_downsample_ratio, self.args.seg_downsample_ratio), order=0)

        if vis_dir is not None:
            if len(commit) > 0: commit = '_' + commit
            desc_id = self.desc.replace(' ', '_')
            self.sam_prompt_model.img = image
            save_fp = os.path.join(vis_dir, f'{desc_id}'+commit+'_'+f'mask.png')
            self.sam_prompt_model.plot(annotations=mask, output_path=save_fp)

        return mask[0, :, :]         # (H, W)


    def _get_intrinsic_parameter(self):
        cx = self.camera.intrinsic_matrix[0, 2]
        cy = self.camera.intrinsic_matrix[1, 2]
        fx = self.camera.intrinsic_matrix[0, 0]
        fy = self.camera.intrinsic_matrix[1, 1]
        return cx, cy, fx, fy


    def find_corresponding_3d_point(self, gripper_2d_pos, pcd_scene, depth_image):
        # pdb.set_trace()
        # gripper_2d_pos: (h,w)-format
        depth_value = depth_image[gripper_2d_pos[0], gripper_2d_pos[1]]

        y, x = gripper_2d_pos
        if depth_value == 0:
            return None
        z = depth_value / 1000.0
        cx, cy, fx, fy = self._get_intrinsic_parameter()
        point_3d = np.array([(x - cx) * z / fx, (y - cy) * z / fy, z])
        pcd_points = np.asarray(pcd_scene.points)
        distances = np.linalg.norm(pcd_points - point_3d, axis=1)
        closest_point_index = np.argmin(distances)
        closest_point = pcd_points[closest_point_index]

        return closest_point


    def set_pcd_voxel_size(self, voxel_size):
        self.args.voxel_size = voxel_size


    def pcd_cut(self, pcd, area_bound):
        # pdb.set_trace()
        pcd_pos = np.asarray(pcd.points)
        is_available = (pcd_pos[:, 0] > area_bound[0]) & (pcd_pos[:, 0] < area_bound[1]) & \
                        (pcd_pos[:, 1] > area_bound[2]) & (pcd_pos[:, 1] < area_bound[3]) & \
                        (pcd_pos[:, 2] > area_bound[4]) & (pcd_pos[:, 2] < area_bound[5]) 
        pcd_cut = o3d.geometry.PointCloud()
        pcd_cut.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[is_available])
        pcd_cut.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[is_available])
        return pcd_cut


    def geometric_generation(self, rgb_image, depth_image, gripper_pos, mask=None, area_bound=None):
        # pdb.set_trace()
        color_raw = o3d.geometry.Image(rgb_image[:, :, :3])      # (H, W, C)
        depth_scene = depth_image.copy()                          
        if mask is not None: 
            depth_scene[mask] = 0
        depth_scene = o3d.geometry.Image(depth_scene)             # (H, W)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_scene, convert_rgb_to_intensity=False)
        pcd_scene = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera)
        if len(gripper_pos) == 3:
            gripper_3d_pos = gripper_pos
        else:
            gripper_3d_pos = self.find_corresponding_3d_point(gripper_pos, pcd_scene, depth_image)
            if gripper_3d_pos is None:
                raise ValueError("gripper_2d_pos={gripper_pos}, but depth is broken, can not find gripper_3d_pos")
        
        if area_bound is not None:
            pcd_scene = self.pcd_cut(pcd_scene, area_bound)

        voxel_down_pcd = pcd_scene.voxel_down_sample(voxel_size=self.args.voxel_size)

        if self.unit_crop_r is not None:
            # pdb.set_trace()
            coord, feat = np.asarray(voxel_down_pcd.points), np.asarray(voxel_down_pcd.colors)
            is_available = (coord[:, 0] > gripper_3d_pos[0] - self.unit_crop_r) & \
                           (coord[:, 0] < gripper_3d_pos[0] + self.unit_crop_r) & \
                           (coord[:, 1] > gripper_3d_pos[1] - self.unit_crop_r) & \
                           (coord[:, 1] < gripper_3d_pos[1] + self.unit_crop_r) & \
                           (coord[:, 2] > gripper_3d_pos[2] - self.unit_crop_r) & \
                           (coord[:, 2] < gripper_3d_pos[2] + self.unit_crop_r)
            coord, feat = coord[is_available], feat[is_available]
            voxel_down_pcd = o3d.geometry.PointCloud()
            voxel_down_pcd.points = o3d.utility.Vector3dVector(coord)
            voxel_down_pcd.colors = o3d.utility.Vector3dVector(feat)

        return voxel_down_pcd, gripper_3d_pos, pcd_scene
    

    def get_kps_3d(self, pcd, gripper_3d_pos, radius=0.08, kps_max=256):
        # pdb.set_trace()
        kps = np.array(pcd.points)
        dist = np.linalg.norm(kps - gripper_3d_pos, axis=1)
        idx = dist < radius
        kps, dist = kps[idx], dist[idx]
        if kps.shape[0]> kps_max:
            idx = np.argsort(dist)[:kps_max]
            kps, dist = kps[idx], dist[idx]

        weights = 1 / (dist + self.args.weight_beta)

        if kps.shape[0] < kps_max:
            # pdb.set_trace()
            repeat_idx = np.random.choice(kps.shape[0], size=kps_max - kps.shape[0], replace=True)
            kps = np.concatenate([kps, kps[repeat_idx]], axis=0)
            weights = np.concatenate([weights, weights[repeat_idx]])

        return kps, weights
    

    def get_kpst_model_prediction(self, data, return_np=False, inference_num=20):
        # pdb.set_trace()
        # data = {'pos': coord, 'x': feat, 'dtraj': qry_pos, 'text_feat': text_features}, Tensor
        pos, feat = data['pos'], data['x']                         
        feat = torch.concat([feat, pos], axis=-1)                     
        dtraj, text_feat = data['dtraj'], data['text_feat']                       # (Q, T=5, 3)        
        query_np = dtraj[:, 0, :]  
                 
        pos = pos.unsqueeze(0).to(self.device).float()                      # (1, N, 3)
        feat = feat.unsqueeze(0).to(self.device).float()                    # (1, N, 6)
        query = query_np.unsqueeze(0).to(self.device).float()               # (1, Q, 3)
        text_feat = text_feat.unsqueeze(0).to(self.device).float()          # (1, Ft)

        traj_prediction = self.kpst_model.inference(pos, feat, text_feat, query, num_sample=inference_num).squeeze(0)   # (Q, M, T-1, 3)
      
        traj_prediction = traj_prediction.transpose(0, 1)                     # (M, Q, T-1=3, 3)
        qry = query.unsqueeze(-2).repeat(traj_prediction.shape[0], 1, 1, 1)   # (M, Q, 1, 3)
        kpst = torch.cat([qry, traj_prediction], -2)                          # (M, Q, T=5, 3)

        # pdb.set_trace()
        dist = torch.mean(torch.sum(torch.norm(kpst[:, :, :-1] - kpst[:, :, 1:], dim=-1), dim=-1), dim=-1) 
        uid = torch.argmax(dist, dim=0)
        kpst = kpst[uid: uid+1]
        print(f"kpst_average_length = {dist[uid]}")

        if return_np is True:
            kpst = kpst.detach().cpu().numpy()
        return kpst
    

    def get_kpst_prediction(self, pcd, kps_3d, return_np=False):
        # pdb.set_trace()
        qry_pos = kps_3d[:, np.newaxis, :]   # (Q, 1, 3), = (Q, T, 3)
        pcd_coord = np.array(pcd.points)         # (N, 3)
        pcd_feat = np.array(pcd.colors)          # (N, 3)
        coord, feat, _ = crop_pc(
            pcd_coord, pcd_feat, None, 'test', self.args.voxel_size, self.args.voxel_max, 
            variable=False, voxel_downsample_bar=0.02, 
            mask=None, mask_ratio=None)
        data = {
            'pos': coord,
            'x': feat,
            'dtraj': qry_pos,
            'text_feat': self.desc_feat
        }
        data = self.data_transform(data)
        # norm_coord = coord.mean(0)               # (3, ), numpy
        norm_coord = qry_pos.mean(0).mean(0)

        kpst = self.get_kpst_model_prediction(data, return_np=return_np)    # (M, Q, T, 3), Tensor
        
        if return_np is False:
            kpst = kpst + torch.from_numpy(norm_coord).to(self.device)      # (M, Q, T, 3)
            kpst = kpst.float()
        else:
            kpst = kpst + norm_coord[np.newaxis, np.newaxis, np.newaxis, :]  # (M, Q, T, 3)
        
        return kpst


    def rigid_transform_3d(self, A, B, weights=None, weight_threshold=0):
        # pdb.set_trace()
        """ 
        CodeBase: https://github.com/zhongcl-thu/3D-Implicit-Transporter
        Input:
            - A:       [bs, num_corr, 3], source point cloud
            - B:       [bs, num_corr, 3], target point cloud
            - weights: [bs, num_corr]     weight for each correspondence 
            - weight_threshold: float,    clips points with weight below threshold
            all is Tensor
        Output:
            - R, t 
        """
        # pdb.set_trace()
        bs = A.shape[0]
        if weights is None:
            weights = torch.ones_like(A[:, :, 0])
        weights[weights < weight_threshold] = 0
        # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

        # find mean of point cloud
        centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
        centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        # construct weight covariance matrix
        Weight = torch.diag_embed(weights)
        H = Am.permute(0, 2, 1) @ Weight @ Bm

        # find rotation
        try:
            U, S, Vt = torch.svd(H.cpu())
            U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
            delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
            eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
            eye[:, -1, -1] = delta_UV
            R = Vt @ eye @ U.permute(0, 2, 1)
            t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
            # warp_A = transform(A, integrate_trans(R,t))
            # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
            return R, t, True
        except:
            print("Fail to Generation.")
            return torch.eye(3).unsqueeze(0).repeat(A.shape[0]).to(self.device), torch.zeros(A.shape[0], 3, 1).to(self.device), False


    def get_motion_planning(self, kpst, weights, plan_step=1, commit=''):
        # pdb.set_trace()
        # desc: str
        # data & weights: numpy dict & numpy.
        # pdb.set_trace()
        if plan_step > kpst.shape[2]:
            raise ValueError(f"plan_step={plan_step} should be smaller than KPST.Length={kpst.shape[2]}")
        motion_plan = []
        if weights.ndim == 1:
            weights = weights[None, :].repeat(kpst.shape[0], 1)    # (M, Q)
        for i in range(plan_step):
            pcd_A = kpst[:, :, i]    # (M, Q, 3)
            pcd_B = kpst[:, :, i+1]    # (M, Q, 3)
            R, t, success = self.rigid_transform_3d(pcd_A, pcd_B, weights=weights)
            R = R.detach().cpu().numpy()
            t = t.detach().cpu().numpy()
            motion_plan.append((R, t, success))
        return motion_plan
    

    def kpst_motion_execusion(self, rgb_image, depth_image, gripper_pos, 
                              area_bound=None,
                              policy_radius=0.08,
                              policy_env='voxel',
                              policy_kps_max=256,
                              robot_anchor=None, 
                              robot_anchor_label=None,
                              plan_step=1, 
                              commit='',
                              vis_dir=None):
        """
            rgb_image: (H, W, 3), RGB, uint8.
            depth_image: (H, W), Depth, float32. (scale: meter)
            gripper_pos: (3, ), the 3d position of gripper. or (2, ) with (h,w)-format, the 2d position of the gripper.
            policy_radius: float, the radius for kps sampling & policy generation.
            policy_env: 'voxel' or 'origin', get kps from voxel-downsampling or origin point cloud.
            robot_anchor: (N, 2), (h,w)-format, numpy, Point or BBox SAM prompt for robot-body segmentation.
            plan_step: int, the step for close-loop motion planning.
            commit: str, the commit for saving the results.
        """
        # pdb.set_trace()
        print(f"KPST-MOTION EXECUSION, Description=[{self.desc}].")
        mask = self.segment_robot_body(rgb_image, robot_anchor, robot_anchor_label, commit=commit, vis_dir=vis_dir)
        pcd, gripper_3d_pos, pcd_org = self.geometric_generation(rgb_image, depth_image, gripper_pos, mask=mask, area_bound=area_bound)

        if policy_env == 'voxel':
            kps_3d, weights = self.get_kps_3d(pcd, gripper_3d_pos, radius=policy_radius, kps_max=policy_kps_max)
        else:
            kps_3d, weights = self.get_kps_3d(pcd_org, gripper_3d_pos, radius=policy_radius, kps_max=policy_kps_max)
        kpst = self.get_kpst_prediction(pcd, kps_3d, return_np=False)   # (M, Q, T, 3), cuda
        weights = torch.Tensor(weights / np.sum(weights)).to(self.device)
        motion_plan = self.get_motion_planning(kpst, weights, plan_step=plan_step, commit=commit)
        if vis_dir is not None:
            vis_kpst = kpst.detach().cpu().numpy()
            vis_result = self.save_exec_data_to_dir(vis_kpst, gripper_3d_pos, motion_plan, pcd_org, vis_dir=vis_dir, commit=commit)
        else:
            vis_result = None
        return motion_plan, vis_result
    
    
    def save_exec_data_to_dir(self, vis_kpst, gripper_3d_pos, motion_plan, pcd_org, vis_dir, commit=''):
        pcd_numpy = np.concatenate([pcd_org.points, pcd_org.colors], axis=-1)
        print(f"Number of Points: {pcd_numpy.shape[0]}")
        result = {
            'description': self.desc,
            'model': self.args.pretrained_path,
            'inference_num': 1,
            'traj_prediction': vis_kpst,                   # (M, Q, T, 3)
            'pcd': pcd_numpy,                              # (N, 6)
            'gripper_3d_pos': gripper_3d_pos,              # (3, )
            'motion_plan': motion_plan,                    # [(R, t, success), ...]
        }
        if len(commit) > 0:
            commit = '_' + commit
        # desc_id = self.desc.replace(' ', '_')
        save_fp = os.path.join(vis_dir, f'robot_exec.pkl')
        save_pickle(save_fp, result)
        return result


def exec_kpst_affordance_from_input_dir(exec_model, input_dir, desc, 
                                        policy_radius=0.1,
                                        gripper_2d_pos=None,
                                        policy_env='voxel',
                                        policy_kps_max=128,
                                        robot_anchor=None, 
                                        plan_step=3, 
                                        vis_dir=None,
                                        area_bound=None,
                                        commit=''):
    if os.path.exists(os.path.join(input_dir, 'rgb.jpg')):
        numpy_image = cv2.imread(os.path.join(input_dir, 'rgb.jpg'), cv2.IMREAD_COLOR)
    else:
        numpy_image = cv2.imread(os.path.join(input_dir, 'rgb.png'), cv2.IMREAD_COLOR)
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    if numpy_image.dtype != np.uint8:
        numpy_image = numpy_image.astype(np.uint8)
    numpy_depth = cv2.imread(os.path.join(input_dir, 'dep.png'), cv2.IMREAD_ANYDEPTH)
    numpy_depth = numpy_depth.astype(np.float32)
        
    exec_model.set_desc(desc)
    print(f"Task: {desc}")
    print(f"Scene: {input_dir}")
    print(f"Weight: {args.pretrained_path}")
    if len(commit) > 0:
        commit = commit + '_' + input_dir.split('/')[-1]
    else:
        commit = input_dir.split('/')[-1]

    depth_image = numpy_depth       # (H, W), original, scale: mm
    rgb_image = numpy_image

    # gripper_2d_pos = np.array([650, 959])
    if gripper_2d_pos is None:
        gripper_2d_pos_wh = KPSTExecutor.display_and_capture_points(rgb_image, title='gripper 2d position')
        gripper_2d_pos = np.array([gripper_2d_pos_wh[0, 1], gripper_2d_pos_wh[0, 0]]) # (w, h) --> (h, w)

    exec_model.kpst_motion_execusion(rgb_image, depth_image, gripper_2d_pos, policy_radius=policy_radius, policy_env=policy_env, policy_kps_max=policy_kps_max,
                                     robot_anchor=robot_anchor, plan_step=plan_step, commit=commit, vis_dir=vis_dir, area_bound=area_bound)
    

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python aff_exec.py input_dir=demo/input/safe_0_hand

    parser = argparse.ArgumentParser('KPST Model Execusion')
    parser.add_argument('--input_dir', type=str, default='demo/input/safe_0_hand')
    parser.add_argument('--desc', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.01, help='the voxel size for downsampling the input point cloud')
    parser.add_argument('--voxel_max', type=int, default=2048)
    parser.add_argument('--weight_beta', type=float, default=0.1, help='the weight beta for the KPST model')
    parser.add_argument('--seg_downsample_ratio', type=int, default=1, help='the downsample ratio for robot-body segmentation')
    parser.add_argument('-p', '--pretrained_path', type=str, 
                        default='log/kpst_hoi4d/ScaleGFlow-B/checkpoint/ckpt_best_train_scalegflow_b.pth')
    
    os.makedirs('demo', exist_ok=True)
    os.makedirs('demo/output', exist_ok=True)
    args, opts = parser.parse_known_args()
    args.cfg = '/'.join(args.pretrained_path.split('/')[:-2] + ['cfg.yaml'])
    args.save_dir = 'demo/output'
    cfg = load_easyconfig_from_yaml(args.cfg)
    print(f"cfg_file_path={args.cfg}")
    cfg.update(opts)
    if cfg.seed is None: cfg.seed = 0

    os.makedirs(args.save_dir, exist_ok=True)
    exec_model = KPSTExecutor(args, cfg)
    
    # TODO: Make sure the camera parameters are correct before you try your own demo !!!!!!!!!!!!!
    # Kinect V2
    # camera = o3d.camera.PinholeCameraIntrinsic(
    #     o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault
    # )
    # exec_model.set_camera(camera)


    # pdb.set_trace()
    input_dir = args.input_dir
    rgb = cv2.imread(os.path.join(input_dir, 'rgb.jpg'), cv2.IMREAD_COLOR)
    H, W = rgb.shape[0], rgb.shape[1]
    camera_param = np.load(input_dir + '/' + 'camera_in.npy')
    exec_model.set_camera(camera_param, H=H, W=W)

    ############################# Model Loading ##################################

    desc = args.desc
    commit = ''
    vis_dir = args.save_dir
    exec_kpst_affordance_from_input_dir(exec_model, input_dir, desc, policy_radius=0.1, gripper_2d_pos=None, robot_anchor=None, vis_dir=vis_dir)
    pdb.set_trace()


# CUDA_VISIBLE_DEVICES=0 python aff_exec.py --input_dir demo/input/safe_0_hand --desc open_Safe --pretrained_path log/kpst_hoi4d/ScaleGFlow-B/checkpoint/ckpt_best_train_scalegflow_b.pth