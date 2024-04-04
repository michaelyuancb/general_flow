import argparse
import json
import sys
import os
import pdb 
from PIL import Image
import random

import numpy as np
from tqdm import tqdm
import cv2
import multiprocessing as mlp
from pcd_hoi4d.pixel2category import get_mask_and_label
from pcd_hoi4d.scene2frame import get_foreground_depths_and_labels
import random
from scipy.spatial.transform import Rotation as Rt

import open3d as o3d

from hoi4d_tool import read_rtd, num2d, transform_vec2mat, objpose2mseg, get_class
from pcd_hoi4d.scene2frame import scene2pcd

# from utils.util import save_pickle

# Accepted: activate actions
ac_actions = [
    "open", "carry", "close", "smash", "push", "pull", "dump",
    "pick up", "pickup", "put down", "putdown", 
    "turn on", "turnon",  "cut", "paper-cut",
    "carry both hands", "carrybothhands"
]

# Ignore: un-chosen actions
uc_actions = [
    "binding", "dig", "decap", "stop", "press", 
    "grasp"  # eg. bucket
]

# Flag: un-activate actions
ua_actions = [
    "rest", "look around", "read", "go", "sit", "squat",
    "reach out", "reachout"
]

cls_name = {
    'C1': 'Toy Car', 'C2': 'Mug', 'C3': 'Laptop', 'C4': 'Storage Furniture',
    'C5': 'Bottle', 'C6': 'Safe', 'C7': 'Bowl', 'C8': 'Bucket', 'C9': 'Scissors',
    'C11': 'Pliers', 'C12': 'Kettle', 'C13': 'Knife', 'C14': 'Trash Can',
    'C17': 'Lamp', 'C18': 'Stapler', 'C20': 'Chair'
}
large_static_multipart = ['Storage Furniture', 'Laptop', 'Safe', 'Trash Can', 'Bucket']

def main_step1_clips_gen(args):

    def get_st_time(action, mark_type):
        if mark_type == 0:
            return action['startTime']
        elif mark_type == 1:
            return action['hdTimeStart']
        
    def get_ed_time(action, mark_type):
        if mark_type == 0:
            return action['endTime']
        elif mark_type == 1:
            return action['hdTimeEnd']

    with open(args.idx_file, "r") as fp:
        idx_list = fp.readlines()
    idx_list = [idx.strip() for idx in idx_list]

    clip_list = []

    for idx in tqdm(idx_list):
        fp = os.path.join(args.anno_root, idx, "action/color.json")
        with open(fp, "r") as fp_json:
            actions = json.load(fp_json)

        # There are 104 video's fps=30 & duration=10.0 , while others fps=15 & duration=20.0

        if 'duration' in actions['info'].keys():
            duration_t = actions['info']['duration']
        elif 'Duration' in actions['info'].keys():
            duration_t = actions['info']['Duration']
        else:
            raise ValueError(f"duration not in clip, idx={idx}: {actions['info']}")
        
        if duration_t != 20:
            fps = 30
        else:
            fps = 15

        if 'events' in actions.keys():
            actions = actions['events']
            mark_type = 0   # {'filePath', 'info', 'events'}
        elif 'markResult' in actions.keys():
            actions = actions['markResult']['marks']
            mark_type = 1   # {'markResult', 'info', 'workload'}
        else:
            raise ValueError(f"Unknown mark type. Keys={actions.keys()}")

        for idx_action, action in enumerate(actions):
            clip = {'index': idx}
            event = action['event'].lower()
            if event not in ac_actions:
                continue
            clip['action'] = event
            clip['object'] = cls_name[get_class(idx)]
            clip['trj_start'] = int(np.max([0, int(get_st_time(action, mark_type) * fps)]))
            clip['trj_end'] = int(np.min([int(get_ed_time(action, mark_type) * fps), 299]))    # each sequence has 300 frames in HOI4D.
            clip['img_pos'] = clip['trj_start']
            clip['fps'] = fps

            idx_pre = idx_action
            pre_act_list = []
            while idx_pre > 0 and (actions[idx_pre-1]['event'].lower() in ua_actions):
                idx_pre = idx_pre - 1
                pre_act_list.append(actions[idx_pre]['event'].lower())
            clip['img_pre'] = int(np.max([0, int(get_st_time(actions[idx_pre], mark_type) * fps)]))
            clip['pre_act_list'] = pre_act_list

            clip_list.append(clip)
    
    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)

    fp = os.path.join(args.output_root, 'step1_clips.json')
    with open(fp, 'w') as f:
        json.dump(clip_list, f, indent=4)
    

def main_step2_trajs_gen(args):

    with open(args.output_root + "/step1_clips.json", "r") as fp:
        org_clip_list = json.load(fp)
    print("Total number of step1_clips: ", len(org_clip_list))

    zero_rot = np.array([0,0,0])

    clip_list = []

    for idx_clip, org_clip in tqdm(enumerate(org_clip_list)):

        ################################ Init ##############################
        obj_pose_root = os.path.join(args.anno_root, org_clip['index'], 'objpose')
        
        camera_fp = os.path.join(args.anno_root, org_clip['index'], '3Dseg', 'output.log')
        outCam = o3d.io.read_pinhole_camera_trajectory(camera_fp).parameters

        traj_st = org_clip['trj_start']
        traj_ed = org_clip['trj_end']
        img_pos = org_clip['img_pos']
        img_pre = org_clip['img_pre']
        fps = org_clip['fps']

        # trajectory clip sampling
        clip_duration = int(args.clip_duration * fps)
        clip_interval = int(args.aft_clip_interval * fps)
        # pre sampling
        if img_pos - img_pre < args.pre_sample:  
            if img_pos == img_pre: 
                pre_idx_st = []
            else:
                pre_idx_st = [iii for iii in range(img_pre, img_pos)]
        else:
            pre_idx_st = np.linspace(0, 1.0, args.pre_sample + 1).tolist()[:-1]
            pre_idx_st = [img_pre + int((img_pos - img_pre) * idx) for idx in pre_idx_st]
        if traj_ed - traj_st < clip_duration:
            traj_clip_idx = [(pre_idx_st[i], traj_st, traj_ed) for i in range(len(pre_idx_st))]
        else:
            traj_clip_idx = [(pre_idx_st[i], traj_st, traj_st + clip_duration) for i in range(len(pre_idx_st))]
        # aft sampling
        if traj_ed - traj_st < clip_duration:
            traj_clip_idx.append((img_pos, traj_st, traj_ed))
        else:
            current = traj_st
            while current + clip_duration <= traj_ed:
                traj_clip_idx.append((current, current, current + clip_duration))
                current = current + clip_interval

            if traj_clip_idx[-1][0] + clip_duration != traj_ed:
                last_st = traj_ed - clip_duration
                traj_clip_idx.append((last_st, last_st, traj_ed))

        for st, traj_st_real, traj_ed_real in traj_clip_idx:
            if traj_ed_real - traj_st_real < args.traj_len:
                continue
            traj_idx = np.linspace(traj_st_real, traj_ed_real, args.traj_len + 1).tolist()
            traj_idx = [int(idx) for idx in traj_idx]

            # Coordination Adjustment (Base_Prepare)
            camera_ex_base = outCam[st].extrinsic
            obj_pose_list = []

            for iiii, traj in enumerate(traj_idx):
                fp = os.path.join(obj_pose_root, f"{traj}.json")
                if os.path.exists(fp):
                    obj_pose = json.load(open(fp, 'r'))
                else:
                    fp = os.path.join(obj_pose_root, num2d(traj)+".json")
                    obj_pose = json.load(open(fp, 'r'))
                if "dataList" in obj_pose.keys():
                    obj_pose = obj_pose["dataList"]
                else:
                    obj_pose = obj_pose["objects"]

                # Coordination Adjustment (Matrix_Prepare)
                camera_ex = outCam[traj].extrinsic
                coord_adjustment = camera_ex_base @ np.linalg.inv(camera_ex)
                    
                n_object = len(obj_pose)
                obj_pose_timestamp = []
                for obj_i in range(n_object):
                    anno = obj_pose[obj_i]
                    rot, trans, _ = read_rtd(anno)

                    # Coordination Adjustment (Final Adjust, from Future to Current)
                    trans = (coord_adjustment @ 
                            np.concatenate([trans, np.ones([1])], axis=0).reshape(4, 1))[:3, 0]

                    obj_pose_timestamp.append({
                        "id": anno['id'],
                        "label": anno['label'],
                        "rot": rot, "trans": trans
                    })
                obj_pose_list.append(obj_pose_timestamp)
                
            final_obj_trans_list = []
            
            for i in range(n_object):
                
                trans_pose_list = []
                rot_current, trans_current = obj_pose_list[0][i]["rot"], obj_pose_list[0][i]["trans"]
                trans_pose_list = [(zero_rot, trans_current)]
                
                for idx in range(len(obj_pose_list) - 1):
                    rot_next, trans_next = obj_pose_list[idx+1][i]["rot"], obj_pose_list[idx+1][i]["trans"]
                    rot_diff = Rt.from_rotvec(rot_next) * (Rt.from_rotvec(rot_current).inv())
                    rot_diff = rot_diff.as_rotvec()

                    trans_pose_list.append((rot_diff, trans_next))


                final_obj_trans_list.append({
                    "id": obj_pose_list[0][i]["id"],
                    "label": obj_pose_list[0][i]["label"],
                    "camera_coord_rot_diff": [transf[0].tolist() for transf in trans_pose_list],
                    "camera_coord_trans": [transf[1].tolist() for transf in trans_pose_list]
                })

            clip = {
                'id': len(clip_list),
                'index': org_clip['index'],
                'img': st,
                'action': org_clip['action'],
                'object': org_clip['object'],
                'traj_idx': traj_idx,
                'camera_coord_transformation': final_obj_trans_list
            }
            clip_list.append(clip)

    fp = os.path.join(args.output_root, 'step2_clips_transformation.json')

    with open(fp, 'w') as f:
        json.dump(clip_list, f, indent=4)


def proc_step3_masks_gen(start_idx, end_idx, org_clip_list, args):

    def save_mask(mask, output_dir_fp, obj_id, traj_id):
        mask = mask
        save_mask_fp = os.path.join(output_dir_fp, "mask_obj"+str(obj_id)+"traj"+str(traj_id)+".png")
        mask_image = Image.fromarray(np.uint8(mask * 255), 'L')
        mask_image.save(save_mask_fp)
    
    os.makedirs(os.path.join(args.output_root, "data"), exist_ok=True)
    print("Total number of step2_clips_trans: ", len(org_clip_list))

    camera_in = o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(camera_in)
    camera_in = camera_intrinsic.intrinsic_matrix
    camera_in_inv = np.linalg.inv(camera_in)

    for idx_clip_info, org_clip in tqdm(enumerate(org_clip_list[start_idx:end_idx])):

        idx_clip = start_idx + idx_clip_info
        
        if org_clip['index'] == 'ZY20210800004/H4/C5/N15/S56/s02/T1':
            # seems that the segmentation label is lack of this clip.
            continue
        
        img_idx = org_clip['img']
        msk_fp = os.path.join(args.anno_root, org_clip['index'], '2Dseg')
        folder_name = os.listdir(msk_fp)[0]
        msk_fp = os.path.join(msk_fp, folder_name, f'{num2d(img_idx)}.png')
        output_dir_fp = os.path.join(args.output_root, "data", str(idx_clip))
        cls = get_class(org_clip['index'])

        mx_img_idx = len(os.listdir(os.path.join(args.data_root, org_clip['index'], 'align_rgb'))) - 1

        if not os.path.exists(output_dir_fp):
            os.mkdir(output_dir_fp)
            
        camera_fp = os.path.join(args.anno_root, org_clip['index'], '3Dseg', 'output.log')
        outCam = o3d.io.read_pinhole_camera_trajectory(camera_fp).parameters

        # traj_idx = org_clip['traj_idx']
        n_traj_len = int(np.min([args.mask_filter_len, mx_img_idx-img_idx]))
        traj_idx = [img_idx + ia for ia in range(n_traj_len+1)]
        n_traj_len = int(np.min([args.mask_filter_len, img_idx]))
        traj_idx = traj_idx + [img_idx - ia for ia in range(1, n_traj_len+1)]

        traj_ok = True

        traj_dict = dict()
        for traj_id in traj_idx:
            msk_fp = os.path.join(args.anno_root, org_clip['index'], '2Dseg')
            folder_name = os.listdir(msk_fp)[0]
            msk_fp = os.path.join(msk_fp, folder_name, f'{num2d(traj_id)}.png')
            dpt_fp = os.path.join(args.data_root, org_clip['index'], 'align_depth', f'{num2d(traj_id)}.png')
            msk, label = get_mask_and_label(msk_fp)
            dpt = get_foreground_depths_and_labels(dpt_fp, msk_fp, msk)

            if len(traj_dict) == 0:
                for idx, lb in enumerate(label):
                    traj_dict[lb] = [(msk[idx], dpt[idx])]
                traj_dict['camera_ex'] = [outCam[traj_id].extrinsic]
            else:
                for idx, lb in enumerate(label):
                    if lb not in traj_dict.keys():
                        traj_ok = False
                        break
                    traj_dict[lb].append((msk[idx], dpt[idx]))
                traj_dict['camera_ex'].append(outCam[traj_id].extrinsic)
        
        if traj_ok is not True:
            continue

        transformation = org_clip['camera_coord_transformation']
        obj_has = []
        for obj in transformation:
            obj_name = obj['label']
            if obj_name in obj_has:      # only pick the first active object.
                continue
            mseg_info = objpose2mseg(cls, obj_name, num=0)
            if not mseg_info:
                continue
            obj_mseg_idx, obj_mseg_part = mseg_info
            if obj_mseg_idx not in traj_dict.keys():
                continue
            
            mask_obj = traj_dict[obj_mseg_idx][0][0]
            # save_mask(mask_obj, output_dir_fp, obj_mseg_idx, 0)

            camera_ex_base = traj_dict['camera_ex'][0]
            for idx_mask, (mask, depth) in enumerate(traj_dict[obj_mseg_idx][1:]):
                mask_indices = np.where(mask == 1)
                v_coords, u_coords = mask_indices[0], mask_indices[1]

                depth_values = depth[v_coords, u_coords] / args.depth_metric

                # Build 3D Points
                ones = np.ones(depth_values.shape).reshape(-1, 1)
                uv_depth = np.stack((u_coords * depth_values, v_coords * depth_values, depth_values), axis=-1)
                points_3d_camera1 = np.dot(camera_in_inv, uv_depth.T).T
                points_3d_camera1 = np.concatenate([points_3d_camera1, ones], axis=-1)
                points_3d_camera2 = np.dot(camera_ex_base, np.dot(np.linalg.inv(traj_dict['camera_ex'][idx_mask+1]), points_3d_camera1.T)).T
                points_3d_camera2 = points_3d_camera2[:, :3]

                # Project Back to the original 
                projected_points = np.dot(camera_in, points_3d_camera2.T).T
                projected_points /= projected_points[:, 2:3]

                new_mask = np.zeros(depth.shape, dtype=np.uint8)
                projected_u = projected_points[:, 0].astype(int)
                projected_v = projected_points[:, 1].astype(int)
                valid_indices = (projected_u >= 0) & (projected_u < new_mask.shape[1]) & \
                                (projected_v >= 0) & (projected_v < new_mask.shape[0])
                new_mask[projected_v[valid_indices], projected_u[valid_indices]] = 1

                # save_mask(mask_obj, output_dir_fp, obj_mseg_idx, idx_mask+1)

                mask_obj =  np.logical_and(mask_obj, new_mask)
            
            if np.sum(mask_obj) == 0:
                continue 
                
            save_obj_mask_fp = os.path.join(output_dir_fp, "confident_mask_"+str(obj_mseg_idx)+".npy")
            np.save(save_obj_mask_fp, mask_obj)
            save_mask(mask_obj, output_dir_fp, obj_mseg_idx, -1)


def proc_step4_kpsts_gen(start_idx, end_idx, org_clip_list, json_fp, args):

    clip_list = []
    camera_in = o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault
    zero_trans = np.array([0,0,0])

    for idx_clip_info, org_clip in tqdm(enumerate(org_clip_list[start_idx:end_idx])):
        idx_clip = int(idx_clip_info + start_idx)

        if org_clip['index'] == 'ZY20210800004/H4/C5/N15/S56/s02/T1':
            # seems that the segmentation label is lack of this clip.
            continue
        
        img_idx = org_clip['img']
        rgb_fp = os.path.join(args.data_root, org_clip['index'], 'align_rgb', f'{num2d(img_idx)}.jpg')
        dpt_fp = os.path.join(args.data_root, org_clip['index'], 'align_depth', f'{num2d(img_idx)}.png')
        msk_fp = os.path.join(args.anno_root, org_clip['index'], '2Dseg')
        folder_name = os.listdir(msk_fp)[0]
        msk_fp = os.path.join(msk_fp, folder_name, f'{num2d(img_idx)}.png')
        output_dir_fp = os.path.join(args.output_root, "data", str(idx_clip))
        
        cls = get_class(org_clip['index'])
        spcd = scene2pcd(args, cls, rgb_fp, dpt_fp, msk_fp, camera_in, output_dir_fp)  # semantic point_cloud.
        # spcd = np.load(os.path.join(output_dir_fp, 'pcd.npy'))
        if spcd is None:
            continue

        clip = {
            'id': idx_clip,
            'index': org_clip['index'],
            'action': org_clip['action'],
            'object': org_clip['object'],
            'img': img_idx,
            'kpst_part': [],
        }
        kpst_traj = []
        kpst_part_id = []
        cls = get_class(org_clip['index'])

        transformation = org_clip['camera_coord_transformation']
        obj_has = []
        for obj in transformation:
            obj_name = obj['label']
            if obj_name in obj_has:      # only pick the first active object.
                continue
            mseg_info = objpose2mseg(cls, obj_name, num=0)
            if not mseg_info:
                continue
            obj_mseg_idx, obj_mseg_part = mseg_info
            
            filtered_rows = spcd[np.abs(spcd[:, 6] - obj_mseg_idx) <= 0.01]
            if filtered_rows.shape[0] <= 0:
                continue

            rot_diff_l = obj['camera_coord_rot_diff']
            trans_l = obj['camera_coord_trans']

            kps = filtered_rows[:, :3]
            kpst = [kps]
            kps_h = np.concatenate([kps - trans_l[0], np.ones((kps.shape[0], 1))], axis=1)  # (N, 1)
            obj_has.append(obj_name)
            kpst_part_id.append(np.ones((kps.shape[0], 1))*obj_mseg_idx)
            clip['kpst_part'].append(obj_mseg_part)

            for it in range(len(rot_diff_l)-1):
                rot_diff = np.array(rot_diff_l[it+1])
                trans_pose = np.array(trans_l[it+1])

                transformation_matrix = transform_vec2mat(rot_diff, zero_trans)
                kps_future = (transformation_matrix @ kps_h.T).T

                kps_future = kps_future[:, :3] + trans_pose.reshape(1, 3)
                kpst.append(kps_future)
            
            kpst = np.stack(kpst, axis=1)   # (N, T, 3)
            kpst_traj.append(kpst)
        
        if len(obj_has) == 0:
            continue
        
        kpst_traj = np.concatenate(kpst_traj, axis=0)        # (N, T, 3)
        kpst_part_id = np.concatenate(kpst_part_id, axis=0)  # (N, 1)

        ############### Camera Shaking Filter #################
        kpst_dis = np.linalg.norm(kpst_traj[:, 0, :] - kpst_traj[:, -1, :], axis=1)  # (N, )
        
        if org_clip['object'] in large_static_multipart:
            # use 4 cm as threshold for large multi-part objects.
            shake_idx = np.where(kpst_dis < args.static_threshold)[0]
        else:
            # use 0.5 cm as threshold for small & rigid objects.
            shake_idx = np.where(kpst_dis < args.static_threshold / 8.0)[0]

        if shake_idx.shape[0] > 0:
            relative_kpst = kpst_traj - kpst_traj[:, 0:1, :]
            shaking_fixed = relative_kpst[shake_idx, :, :].mean(axis=0, keepdims=True)   # (1, T, 3)
            kpst_traj = kpst_traj - shaking_fixed

        np.save(os.path.join(output_dir_fp, 'kpst_traj.npy'), kpst_traj)
        np.save(os.path.join(output_dir_fp, 'kpst_part_id.npy'), kpst_part_id)
        # break
        clip_list.append(clip)

    fp = os.path.join(json_fp, 'step3_clips_kpst_'+str(start_idx)+'_'+str(end_idx-1)+'.json')
    with open(fp, 'w') as f:
        json.dump(clip_list, f, indent=4)


def main_step34_kpst_gen(args):

    def step34_gather(start_idx, end_idx, org_clip_list, json_fp, args):
        proc_step3_masks_gen(start_idx, end_idx, org_clip_list, args)  
        proc_step4_kpsts_gen(start_idx, end_idx, org_clip_list, json_fp, args)

    os.makedirs(args.output_root, exist_ok=True)
    with open(args.output_root + "/step2_clips_transformation.json", "r") as fp:
        org_clip_list = json.load(fp)
    print("Total number of step2_clips_trans: ", len(org_clip_list))

    json_fp = os.path.join(args.output_root, "json_cache")
    if not os.path.exists(json_fp):
        os.mkdir(json_fp)

    n_clip = len(org_clip_list)
    numFramesPerThread = np.ceil(n_clip / args.num_threads).astype(np.uint32)
    # numFramesPerThread = np.ceil(args.num_threads).astype(np.uint32)

    if args.num_threads > 1:
        procs = []

        for proc_index in range(args.num_threads):
            startIdx = proc_index * numFramesPerThread
            endIdx = min(startIdx + numFramesPerThread, n_clip)
            print(f"proc-{proc_index}: [startIdx={startIdx}] [endIdx={endIdx}]")
            proc_args = (startIdx, endIdx, org_clip_list, json_fp, args)
            proc = mlp.Process(target=step34_gather, args=proc_args)

            proc.start()
            procs.append(proc)
            # break

        for i in range(len(procs)):
            procs[i].join()

    else:
        step34_gather(0, len(org_clip_list), org_clip_list, json_fp, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_threads', type=int, default=10)

    # data paths
    parser.add_argument('--data_root', type=str, default='/public/datasets_yl/HOI4D/')
    parser.add_argument('--anno_root', type=str, default='/public/datasets_yl/HOI4D_annotations/')
    parser.add_argument('--idx_file', type=str, default='HOI4D-Instructions/release.txt')
    parser.add_argument('--depth_metric', type=float, default=1000.0)

    # image_position sampling & duration
    parser.add_argument('--pre_sample', type=int, default=4)
    parser.add_argument('--aft_clip_interval', type=float, default=0.15)
    parser.add_argument('--clip_duration', type=float, default=1.5)

    # data setting
    parser.add_argument('--traj_len', type=int, default=3)
    parser.add_argument('--voxel_size', type=float, default=0.02)
    parser.add_argument('--obj_erode_iter', type=int, default=2, help='erode inside for object-mask.')
    parser.add_argument('--hand_erode_iter', type=int, default=8, help='erode outside for hand-mask.')
    parser.add_argument('--mask_filter_len', type=int, default=1)
    parser.add_argument('--static_threshold', type=float, default=0.02, help='unit: meter')

    # parser.add_argument('--output_root', type=str, default='HOI4D_KPST')
    parser.add_argument('--output_root', type=str, default='/home/ycb/HOI4D_KPST')
    # parser.add_argument('--output_root', type=str, default='/cache0/ycb/kpst/HOI4D_KPST')

    args = parser.parse_args()

    main_step1_clips_gen(args)
    main_step2_trajs_gen(args)
    main_step34_kpst_gen(args)