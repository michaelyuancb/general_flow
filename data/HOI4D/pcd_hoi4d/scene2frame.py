import os
import cv2
import open3d as o3d
from plyfile import PlyData
import pandas as pd
import pdb
import numpy as np
from skimage import feature as sk_ft

from .pixel2category import get_mask_and_label


# object_motion label that is not the hand
obj_not_hand_dict = {
    'C1': [1],
    'C2': [1, 3],
    'C3': [1, 3],
    'C4': [1, 3, 4, 5],
    'C5': [1, 3],
    'C6': [1, 3, 4],
    'C7': [1, 3],
    'C8': [1, 3, 4, 5],
    'C9': [1, 3],
    'C11': [1, 3],
    'C12': [1, 3],
    'C13': [1, 3],
    'C14': [1, 3, 4],
    'C17': [1, 3, 4, 5],
    'C18': [1, 3, 4, 5],
    'C20': [1]
}

obj_hand_dict = {
    'C1': [2],
    'C2': [2],
    'C3': [2],
    'C4': [2],
    'C5': [2],
    'C6': [2],
    'C7': [2],
    'C8': [2, 6],
    'C9': [2],
    'C11': [2],
    'C12': [2],
    'C13': [2],
    'C14': [2],
    'C17': [2, 6],
    'C18': [2],
    'C20': [2, 3]
}


def mseg_not_hand(cls, mseg_id):
    if cls not in obj_not_hand_dict:
        raise ValueError("Unknown Object Class: {}".format(cls))
    if mseg_id > 6:
        raise ValueError("Unknown Motion Segmentation ID: {}".format(mseg_id))
    else:
        return mseg_id in obj_not_hand_dict[cls]


def get_foreground_depths_and_labels(depth_path, mask_path, ds):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    assert mask.shape == (1080, 1920, 3)

    ans = []
    for d in ds:
        x = depth.copy()
        x[~d] = 0
        ans.append(x)
    return ans


erode_kernel = np.ones((3,3), np.uint8)


def convert(args, obj_cls, depth_path, mask_path, output_root):

    # background
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    assert mask.shape == (1080, 1920, 3)

    s = np.sum(mask, axis=2)
    depth_b = depth.copy()
    depth_b[s > 0] = 0

    # foreground
    depth_fs = []
    ds, ls_motion = get_mask_and_label(mask_path)
    depths = get_foreground_depths_and_labels(depth_path, mask_path, ds)

    for i in range(len(depths)):
        dpt = depths[i]
        motion_label = ls_motion[i]
        if motion_label not in obj_hand_dict[obj_cls]:    # conduct hand extend first.
            continue
        mask_org_hand = (dpt > 0)
        mask_org = ~ mask_org_hand
        mask_erode = cv2.erode(mask_org.astype(np.uint8), erode_kernel, iterations=args.hand_erode_iter).astype(bool)
        mask_hand = ~ mask_erode
        edges = mask_hand ^ mask_org_hand
        depths[i][edges] = depth[edges]
        for j in range(len(depths)):
            if i == j: continue
            depths[j][edges] = 0
        depth_b[edges] = 0


    for i in range(len(depths)):
        d = depths[i]
        motion_label = ls_motion[i]

        if (motion_label not in obj_not_hand_dict[obj_cls]) and (motion_label not in obj_hand_dict[obj_cls]):
            continue

        if motion_label in obj_hand_dict[obj_cls]:
            motion_label = 2
            # we use 2 to represent hand, since 2 is the typical motion label of hand in HOI4D dataset.

        if (motion_label != 2) and os.path.exists(os.path.join(output_root, "confident_mask_"+str(motion_label)+".npy")):
            # load confident object mask
            mask_confident = np.load(os.path.join(output_root, "confident_mask_"+str(motion_label)+".npy"))
            mask_org = (d > 0)
            mask_hand_erode = (mask_org == False) & (mask_confident == True)
            mask_confident[mask_hand_erode] = False
            edges = mask_org ^ mask_confident 
            depth_b[edges > 0] = d[edges > 0]
            d[edges > 0] = 0 
            if args.obj_erode_iter > 1 and (motion_label != 2):
                mask_org = (d > 0).astype(np.uint8)
                mask_erode = cv2.erode(mask_org, erode_kernel, iterations=args.obj_erode_iter)
                edges = mask_org - mask_erode 
                depth_b[edges > 0] = d[edges > 0]
                d[edges > 0] = 0
        elif (motion_label != 2):
            obj_mask = ds[i].astype(np.uint8) > 0
            depth_b[obj_mask] = d[obj_mask]
            continue
        else:
            pass

        output_foreground_p = os.path.join(output_root, "fg_depth_"+str(i+1)+".png")
        cv2.imwrite(output_foreground_p, d)
        d_f = o3d.io.read_image(output_foreground_p)
        depth_fs.append([d_f, ls_motion[i]])

    
    output_background_p = os.path.join(output_root, "bg_depth.png")
    cv2.imwrite(output_background_p, depth_b)
    depth_b = o3d.io.read_image(output_background_p)
    
    return depth_fs, depth_b    # LIST([o3d.io.Image, label, labels_instanceseg]), o3d.io.Image


def scene2pcd(args, obj_cls, rgb_path, depth_path, mask_path, camera_intrinsic,
              output_root,
              camera_extrinsic=np.eye(4)):

    os.makedirs(output_root, exist_ok=True)

    color_raw = o3d.io.read_image(rgb_path)

    # # (TODO) can be deleted, only for conviniently checking
    # o3d.io.write_image(os.path.join(output_root, "rgb.png"), color_raw)  
    
    depth_foregrounds, depth_background = convert(args, obj_cls, depth_path, mask_path, output_root)

    # background
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_background,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault),extrinsic=camera_extrinsic)
    
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    # voxel_down_pcd = pcd
    o3d.io.write_point_cloud(os.path.join(output_root, 'bg.ply'), voxel_down_pcd)

    plydata_0 = PlyData.read(os.path.join(output_root, 'bg.ply'))
    data_0 = plydata_0.elements[0].data
    data_pd_0 = pd.DataFrame(data_0)
    data_np_0 = np.zeros((data_pd_0.shape[0],data_pd_0.shape[1]+1), dtype=np.float64) 
    property_names_0 = data_0[0].dtype.names
    for i, name in enumerate(property_names_0): 
        data_np_0[:, i] = data_pd_0[name]
    
    geo = data_np_0[:,:3]
    color = data_np_0[:, 3:6]
    label = data_np_0[:, 6:]
    ans = np.concatenate([geo,color,label],axis=1)
    
    has_part = False

    for i in range(len(depth_foregrounds)):
        depth_f = depth_foregrounds[i][0]
        if np.sum(depth_f) == 0:
            continue
        label_motion = depth_foregrounds[i][1]
        has_part = True

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_f,convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault),extrinsic=camera_extrinsic)

        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        points = np.array(voxel_down_pcd.points)
        if points.shape[0] == 0:
            continue
        o3d.io.write_point_cloud(os.path.join(output_root, 'fg_'+str(i+1)+'.ply'), voxel_down_pcd)
        plydata_0 = PlyData.read(os.path.join(output_root, 'fg_'+str(i+1)+'.ply'))
        data_0 = plydata_0.elements[0].data
        data_pd_0 = pd.DataFrame(data_0)
        data_np_0 = np.zeros((data_pd_0.shape[0],data_pd_0.shape[1]+1), dtype=np.float64)  
        property_names_0 = data_0[0].dtype.names  
        for ii, name in enumerate(property_names_0): 
            data_np_0[:, ii] = data_pd_0[name]

        geo = data_np_0[:,:3]
        color = data_np_0[:, 3:6]
        label = np.ones_like(data_np_0[:, 6:]) * label_motion
     
        # pcd_points.append(np.asarray(voxel_down_pcd.points))
        # pcd_colors.append(np.asarray(voxel_down_pcd.colors))
        out = np.concatenate([geo,color,label],axis=1)
        ans = np.concatenate([ans,out],axis=0)

    if not has_part:
        return None
    
    np.save(os.path.join(output_root, 'pcd.npy'), ans)
    # pcd_points = np.concatenate(pcd_points, axis=0)
    # pcd_colors = np.concatenate(pcd_colors, axis=0)
    # pcd_all = o3d.geometry.PointCloud()
    # pcd_all.points = o3d.utility.Vector3dVector(np.array(pcd_points))
    # pcd_all.colors = o3d.utility.Vector3dVector(np.array(pcd_colors))
    # o3d.io.write_point_cloud(os.path.join(output_root, 'pcd.ply'), pcd_all)

    # ans: a (N, 7) numpy, with:
    # (1) ans[:, :3] as (x,y,z)
    # (2) ans[:, 3:6] as (r,g,b)
    # (3) ans[:, 6] as motion segmentation label.
    return ans


'''
if __name__ == '__main__':
    app.run(main)
'''
