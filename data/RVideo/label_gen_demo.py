import os
import torch
import pdb
import numpy as np 
import pandas as pd
from tqdm import tqdm

from PIL import Image
import cv2
import json
import time
import open3d as o3d
import argparse
import scipy.ndimage
from base64 import b64encode

from plyfile import PlyData

from tool_repos.FastSAM.fastsam import FastSAM
from fastsam_prompt import FastSAMPrompt
from ego_hoi_detector import EgoHOIDetector
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor


def save_video(video_save_fp, video):
    video_np = video.squeeze(0).permute(0, 2, 3, 1).numpy()
    # pdb.set_trace()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_save_fp, fourcc, 30, (1920, 1080))
    for frame in video_np:
        frame_bgr = cv2.cvtColor(np.uint8(frame), cv2.COLOR_RGB2BGR) 
        video_writer.write(frame_bgr)
    video_writer.release()


class EgoHOIAnalysts(object):

    def __init__(self, 
                 fastsam_fp='./tool_repos/FastSAM/weights/FastSAM-X.pt', 
                 ego_hoi_det_cfg='tool_repos/ego_hand_detector/cfgs/res101.yml',
                 ego_hoi_det_pth='tool_repos/ego_hand_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth',
                 tracker_pth = 'tool_repos/co_tracker/checkpoints/cotracker_stride_8_wind_16.pth',
                 downsample_ratio=1,     # for faster inference
                 device='cuda',
                 ):
        
        self.device = device
        self.fastsam_fp = fastsam_fp
        self.ego_hoi_det_cfg = ego_hoi_det_cfg
        self.ego_hoi_det_pth = ego_hoi_det_pth
        self.downsample_ratio = downsample_ratio

        self.sam_model = FastSAM('./tool_repos/FastSAM/weights/FastSAM-X.pt')
        self.sam_prompt_model = FastSAMPrompt(device=device)
        self.hoi_det_model = EgoHOIDetector(cfg_file=self.ego_hoi_det_cfg,
                                            pretrained_path=self.ego_hoi_det_pth)
        kps_tracker = CoTrackerPredictor(checkpoint=os.path.join(tracker_pth))
        self.kps_tracker = kps_tracker.to(device)
        
    def segment_hoi_object(self, image, vis_dir=None):
        """
            Get the Segmentation Mask of Active HOI Objects. 

            img: (H, W, C) & RGB & Numpy-Image
            Return: (1, H, W), Human-Interaction Object Segmentation Mask.
        """
        start_time = time.time()
        vis = vis_dir is not None

        img_pil = Image.fromarray(image)
        new_height = img_pil.height // self.downsample_ratio
        new_width = img_pil.width // self.downsample_ratio
        resized_img_pil = img_pil.resize((new_width, new_height))
        resized_img = np.asarray(resized_img_pil)
        obj_det, hand_det = self.hoi_det_model.detect(resized_img, vis=vis)  # <uw, uh, dw, dh>
        
        bboxes_obj  = []    
        bboxes_hand = []                  
        # only use bbox with contact confident > 0.5
        if hand_det[0, 4] > 0.5: bboxes_obj.append(obj_det[0, :4]), bboxes_hand.append(hand_det[0, :4])
        if hand_det[1, 4] > 0.5: bboxes_obj.append(obj_det[1, :4]), bboxes_hand.append(hand_det[1, :4])

        # pdb.set_trace()
        if len(bboxes_obj) > 0:
            everything_results = self.sam_model(resized_img, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
            self.sam_prompt_model.set_image_result(resized_img, everything_results)

            ann_hand, mask_index_hand = self.sam_prompt_model.box_prompt(bboxes=bboxes_hand)
            ann_hand = np.max(ann_hand, axis=0, keepdims=True)
            
            if vis:
                self.sam_prompt_model.plot(annotations=ann_hand, output_path=vis_dir+'/mask_hand.jpg')
            ann_obj, mask_index_obj = self.sam_prompt_model.box_prompt(bboxes=bboxes_obj, limit_mask_index=mask_index_hand)
            if len(ann_obj) < len(bboxes_obj):
                ann_obj = ann_obj.repeat(len(bboxes_obj)//len(ann_obj), axis=0)
            # ann: (H, W, C)

            ann = np.max(ann_obj, axis=0, keepdims=True)
            ann = scipy.ndimage.zoom(ann, (1, self.downsample_ratio, self.downsample_ratio), order=0)
            if np.sum(ann) == 0:
                return None, ann_hand
            if vis:
                print(f"Execusion Time: {time.time() - start_time} secs")
                self.sam_prompt_model.img = image
                self.sam_prompt_model.plot(annotations=ann, output_path=vis_dir+'/mask.jpg')
            return ann, ann_hand
        else:
            return None, ann_hand
        
    def get_kps(self, mask, n_sample_max=1024):
        if mask.ndim == 2:
            mask = mask[None]
        H, W = mask.shape[1], mask.shape[2]  
        mask_flattened = mask.reshape(-1) 
        ones_indices = np.where(mask_flattened == 1)[0]
        if len(ones_indices) > n_sample_max:
            selected_indices = np.random.choice(ones_indices, size=n_sample_max, replace=False)
        else:
            selected_indices = ones_indices
        selected_2d_indices = np.array([np.unravel_index(idx, (H, W)) for idx in selected_indices])
        return selected_2d_indices
    
    def get_kpst_track(self, video, kps_2d, vis_dir=None):
        # kps_2d: (h, w) --> (w, h) format
        query = np.concatenate([np.zeros((kps_2d.shape[0], 1)), kps_2d[:, 1:2], kps_2d[:, 0:1]], axis=1)
        query = torch.tensor(query).cuda()

        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()  # torch.Size([1, 113, 3, 1080, 1920])
        video = video.to(self.device)
        pred_tracks, pred_visibility = self.kps_tracker(video.float(), queries=query[None].float())

        if vis_dir is not None:
            vis = Visualizer(save_dir=vis_dir, linewidth=3, mode='cool', tracks_leave_trace=-1)
            vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='queries')
        
        pred_tracks = pred_tracks[0].cpu().numpy()
        pred_visibility = pred_visibility[0].cpu().numpy()   # (w, h) - format
        return pred_tracks, pred_visibility

HEIGHT = 720
WIDTH = 1280
FPS = 15

def get_video(base_fp, prefix='none'):
    n_file = len(os.listdir(base_fp))
    n_file = (n_file - 1) // 3
    video = []
    for fl_id in range(n_file):
        image = Image.open(os.path.join(base_fp, f'{prefix}_{fl_id}.png'))
        image = np.asarray(image)
        video.append(image)
    video = np.stack(video, axis=0)          # (T, H, W, C)
    return video


def get_mark_image(image, mark_coordinates):
    im = image.copy()
    for x, y in mark_coordinates:
        cv2.circle(im, (y, x), radius=5, color=(0, 255, 0), thickness=-1)
    return im


def kpst_label_gen_demo_2d(analysts, args):

    task_list = os.listdir(args.raw_data_root)
    clip_list = []
    fps = 15    
    os.makedirs(os.path.join(args.save_root, 'step1_2d'), exist_ok=True)

    for task in task_list:
        action, object = task.split(' ')[0], task.split(' ')[1]
        base_task_read_fp = os.path.join(args.raw_data_root, task)
        exec_fp_list = os.listdir(base_task_read_fp)
        
        for exec_fp in tqdm(exec_fp_list):
            base_fp = os.path.join(base_task_read_fp, exec_fp)
            video_rgb = get_video(base_fp, 'rgb')               # (T, H, W, C)
            # video_dep = get_video(base_fp, 'depth')             # (T, H, W)
            n_frame = (len(os.listdir(base_fp)) - 1) // 3
            traj_st, traj_ed = 0, n_frame - 1


            clip_duration = int(args.clip_duration * fps)
            clip_interval = int(args.aft_clip_interval * fps)
            traj_clip_idx = []
            if traj_ed - traj_st < clip_duration:
                traj_clip_idx.append((traj_st, traj_ed))
            else:
                current = traj_st
                while current + clip_duration <= traj_ed:
                    traj_clip_idx.append((current, current + clip_duration))
                    current = current + clip_interval
            if traj_clip_idx[-1][0] + clip_duration != traj_ed:
                last_st = traj_ed - clip_duration
                traj_clip_idx.append((last_st, traj_ed))

            for traj_idx in traj_clip_idx:

                traj_st_real, traj_ed_real = traj_idx
                if traj_ed_real - traj_st_real < args.traj_len:
                    continue
                # pdb.set_trace()
                traj_idx = np.linspace(traj_st_real, traj_ed_real-1, args.traj_len + 1).tolist()
                traj_idx = [int(idx) for idx in traj_idx]

                clip = {
                    'id': len(clip_list),
                    'index': base_fp,
                    'st': traj_st_real,
                    'ed': traj_ed_real, 
                    'action': action,
                    'object': object,
                    'seq_index': traj_idx
                }
                save_fp = os.path.join(args.save_root, 'step1_2d', str(len(clip_list)))
                os.makedirs(save_fp, exist_ok=True)

                rgb_video_clip = video_rgb[traj_st_real: traj_ed_real]
                # dep_video_clip = video_dep[traj_st_real: traj_ed_real]
                rgb_image = rgb_video_clip[0]
                mask_obj, mask_hand = analysts.segment_hoi_object(rgb_image, vis_dir=save_fp)       # (1, H, W), (1, H, W)
                if mask_obj is None: 
                    print("Segment Fail, Continue.")
                kps = analysts.get_kps(mask_obj, n_sample_max=args.n_sample_max)     # (h, w)
                mark_image = get_mark_image(rgb_image, kps)
                pred_tracks, pred_visibility = analysts.get_kpst_track(rgb_video_clip, kps, vis_dir=save_fp)  # (37, 512, 2), (37, 512)
                
                cv2.imwrite(os.path.join(save_fp, 'mark.jpg'), cv2.cvtColor(mark_image, cv2.COLOR_RGB2BGR))
                np.save(os.path.join(save_fp, 'mask_obj.npy'), mask_obj[0])
                np.save(os.path.join(save_fp, 'mask_hand.npy'), mask_hand[0])
                np.save(os.path.join(save_fp, 'kps_tracks.npy'), pred_tracks)            # (w, h) - format
                np.save(os.path.join(save_fp, 'kps_visibility.npy'), pred_visibility)    # (w, h) - format

                clip_list.append(clip)

                # break
            break
        break

    fp = os.path.join(args.save_root, 'step1_kpst_2d_info.json')
    with open(fp, 'w') as f:
        json.dump(clip_list, f, indent=4)


def get_camera(camera_in):
    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(WIDTH, HEIGHT, camera_in[0,0], camera_in[1,1], camera_in[0,2], camera_in[1,2])
    return camera

def get_intrinsic_parameter(camera_in):
    cx = camera_in[0, 2]
    cy = camera_in[1, 2]
    fx = camera_in[0, 0]
    fy = camera_in[1, 1]
    return cx, cy, fx, fy


def kpst_label_gen_demo_3d(args):
    with open(os.path.join(args.save_root, 'step1_kpst_2d_info.json')) as fp:
        org_clip_list = json.load(fp)
    print("Total number of step1_kpst_label_gen_2d clips: ", len(org_clip_list))
    os.makedirs(os.path.join(args.save_root, 'data'), exist_ok=True)


    clip_list = []
    for _, clip_org in tqdm(enumerate(org_clip_list)):
        
        base_fp = clip_org['index']
        step_fp = os.path.join(args.save_root, 'step1_2d', str(clip_org['id']))
        save_fp = os.path.join(args.save_root, 'data', str(clip_org['id']))
        os.makedirs(save_fp, exist_ok=True)
        camera_in = np.load(os.path.join(base_fp, f"camera_in.npy"))
        camera = get_camera(camera_in)

        seq_idx_list = clip_org['seq_index']
        kps_track = np.load(os.path.join(step_fp, 'kps_tracks.npy'))                # (T, N, 2), wh-format
        kps_visibility = np.load(os.path.join(step_fp, 'kps_visibility.npy'))       # (T, N), wh-format
        kps_pos, kps_vis = [], []
        st = clip_org['st']

        for sidx in seq_idx_list:
            kps_pos.append(kps_track[sidx-st])
            kps_vis.append(kps_visibility[sidx-st])
        kps_pos, kps_vis = np.stack(kps_pos, axis=1), np.stack(kps_vis, axis=1)  # (N, T, 2), (N, T)

        kps_vis_sum = np.sum(kps_vis, axis=1)
        is_available = (kps_vis_sum == kps_vis.shape[1])
        if np.sum(is_available) == 0: continue
        kps_pos = kps_pos[is_available]
        kps_vis = kps_vis[is_available]
        
        point_3d_list = []
        for img_idx, t in zip(seq_idx_list, range(kps_vis.shape[1])):
            # pdb.set_trace()
            # pdb.set_trace()
            pos_list = kps_pos[:, t]  # (N, 2), wh-format from co-tracker
            dep = np.asarray(Image.open(os.path.join(base_fp, f"depth_{img_idx}.png")))
            pos_list = np.round(pos_list).astype(np.uint16)
            pos_list[:, 0] = np.clip(pos_list[:, 0], 0, WIDTH-1)                   
            pos_list[:, 1] = np.clip(pos_list[:, 1], 0, HEIGHT-1)
            pos_dep = dep[pos_list[:, 1], pos_list[:, 0]]

            z = pos_dep / 1000.0
            cx, cy, fx, fy = get_intrinsic_parameter(camera_in)
            point_3d = np.stack([(pos_list[:, 0] - cx) * z / fx, (pos_list[:, 1] - cy) * z / fy, z], axis=-1)
            point_3d_list.append(point_3d)
        
        kpst_3d = np.stack(point_3d_list, axis=1)  # (N, T, 3)
        is_available = (kpst_3d[:, :, -1] > 0).sum(axis=1) == kpst_3d.shape[1]
        if np.sum(is_available) == 0: continue
        kpst_3d = kpst_3d[is_available]
        kpst_dist = np.sum(np.linalg.norm(kpst_3d[:, 1:]-kpst_3d[:, :-1], axis=-1), axis=-1)
        is_available = kpst_dist > args.static_threshold
        if np.sum(is_available) == 0: continue

        np.save(os.path.join(save_fp, 'kpst_traj.npy'), kpst_3d)   # (N, T, 3)
        np.save(os.path.join(save_fp, 'kpst_part_id'), np.ones((kpst_3d.shape[0], 1)))

        dep = np.asarray(Image.open(os.path.join(base_fp, f"depth_{st}.png")))
        mask_hand = np.load(os.path.join(step_fp, "mask_hand.npy"))
        mask_obj = np.load(os.path.join(step_fp, "mask_obj.npy"))

        mask = np.zeros_like(mask_obj)
        mask = mask + mask_obj + 2 * mask_hand
        mask_bg = (mask==0)
        mask_list = [mask_bg, mask_obj, mask_hand]

        geo_list, color_list, label_list = [], [], []
        for i, mk in enumerate(mask_list):
            
            color_raw = o3d.io.read_image(os.path.join(base_fp, f"rgb_{st}.png"))
            depth_t = dep.copy()
            mk = mk > 0
            depth_t[~mk] = 0 
            cv2.imwrite(os.path.join(step_fp, f"dep_part_{i}.png"), depth_t.astype(np.uint16))
            depth = o3d.io.read_image(os.path.join(step_fp, f"dep_part_{i}.png"))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth,convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera ,extrinsic=np.eye(4))

            voxel_down_pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
            # voxel_down_pcd = pcd
            # pdb.set_trace()
            o3d.io.write_point_cloud(os.path.join(save_fp, f'bg_{i}.ply'), voxel_down_pcd)
            plydata_0 = PlyData.read(os.path.join(save_fp, f'bg_{i}.ply'))
            data_0 = plydata_0.elements[0].data
            data_pd_0 = pd.DataFrame(data_0)
            data_np_0 = np.zeros((data_pd_0.shape[0],data_pd_0.shape[1]+1), dtype=np.float64) 
            property_names_0 = data_0[0].dtype.names
            for ii, name in enumerate(property_names_0): 
                data_np_0[:, ii] = data_pd_0[name]
            geo = data_np_0[:,:3]
            color = data_np_0[:, 3:6]
            label = np.ones((data_np_0.shape[0], 1)) * i
            geo_list.append(geo)
            color_list.append(color)
            label_list.append(label)

        spcd = np.concatenate([np.concatenate(geo_list, axis=0), np.concatenate(color_list, axis=0),
                               np.concatenate(label_list, axis=0)], axis=1)
        np.save(os.path.join(save_fp, 'pcd.npy'), spcd)
        
        clip_list.append(clip_org)

    fp = os.path.join(args.save_root, 'metadata_egosoft_demo.json')
    with open(fp, 'w') as f:
        json.dump(clip_list, f, indent=4)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser('TAP-KPST Label Extraction')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--downsample_ratio', type=int, default=1)
    parser.add_argument('--n_sample_max', type=int, default=1024)

    parser.add_argument('--aft_clip_interval', type=float, default=0.15)    
    parser.add_argument('--clip_duration', type=float, default=1.5)
    parser.add_argument('--traj_len', type=int, default=3)
    parser.add_argument('--voxel_size', default=0.02, type=float)
    parser.add_argument('--static_threshold', default=0.02, type=float)

    parser.add_argument('--raw_data_root', type=str, default='recorded_rgbd')
    parser.add_argument('--save_root', type=str, default='Soft_KPST')
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    analysts = EgoHOIAnalysts(downsample_ratio=args.downsample_ratio, device=args.device)
    kpst_label_gen_demo_2d(analysts, args)
    kpst_label_gen_demo_3d(args)
