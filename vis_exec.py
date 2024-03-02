import argparse
import open3d as o3d
import numpy as np
import pdb
import random
import pdb

import argparse
import json
import sys
import os
import pdb 
import pickle
import random

import numpy as np
from tqdm import tqdm
import cv2
from scipy.spatial.transform import Rotation as Rt

import open3d as o3d

# from datasets.visulization import get_arrow

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_pkl(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

 
def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat
 
 
def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
 
    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
 
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
 
    qTrans_Mat *= scale
    return qTrans_Mat
 
def get_arrow(begin=[0,0,0],end=[0,0,1]):
    vec = end - begin
    z_unit_Arr = np.array([0, 0, 1])
    begin = begin
    end = np.add(begin,vec)
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)
 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])
 
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * 1 ,
        cone_radius=0.06 * 1,
        cylinder_height=0.8 * 1,
        cylinder_radius=0.04 * 1
    )
    mesh_arrow.paint_uniform_color([0, 1, 0])
    mesh_arrow.compute_vertex_normals()
 
    mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=0.001, resolution=20)
    mesh_sphere_begin.translate(begin)
    mesh_sphere_begin.paint_uniform_color([0, 1, 1])
    mesh_sphere_begin.compute_vertex_normals()
 
    mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=0.001, resolution=20)
    mesh_sphere_end.translate(end)
    mesh_sphere_end.paint_uniform_color([0, 1, 1])
    mesh_sphere_end.compute_vertex_normals()
 
 
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end
 

def visualization_exec(args, result):
    org_pcd = result['pcd']         


    print("#"*20 + " SCENE INFO " + "#"*20)
    print(f"N_Points={org_pcd.shape[0]}")
    print(f"N_KPS={result['traj_prediction'].shape[1]}")

    print("#"*20 + " MODEL INFO " + "#"*20)
    print(f"model: {result['model']}")
    print(f"description: {result['description']}")
    print(f"inference_num: {result['inference_num']}")
    print("#"*20 + " VISUALIZATION START " + "#"*20)

    color_motion = [0, 0, 1]    # green for motion
    color_kpst = [0, 1, 0]      # red for kpst

    use_traj_idx = 0 

    traj_prediction = result['traj_prediction'][use_traj_idx]  # (Q, T, 3)
    nQ = traj_prediction.shape[0]
    if nQ > args.max_traj:
        traj_prediction = traj_prediction[random.sample(range(nQ), args.max_traj)]
    gripper_3d_pos = result['gripper_3d_pos']
    motion_R = [x[0] for x in result['motion_plan']]
    motion_t = [x[1] for x in result['motion_plan']]
    motion_S = [x[2] for x in result['motion_plan']]


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(org_pcd[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(org_pcd[:, 3:6] / 255.0)
    vis.add_geometry(pcd)

    for idx_t, pred_vec  in enumerate(traj_prediction):
        for ii in range(pred_vec.shape[0]-1):
            point2, point1 = pred_vec[ii+1], pred_vec[ii]
            mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end = get_arrow(point1, point2)
            mesh_arrow.paint_uniform_color(color_kpst) 
            vis.add_geometry(mesh_arrow)
            if ii == 0:  vis.add_geometry(mesh_sphere_begin)

    pos = gripper_3d_pos
    mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=0.002, resolution=20)
    mesh_sphere_begin.translate(pos)
    mesh_sphere_begin.paint_uniform_color(color_motion)
    mesh_sphere_begin.compute_vertex_normals()
    vis.add_geometry(mesh_sphere_begin)

    # pdb.set_trace()
    
    pos_org = gripper_3d_pos

    for idx, (R, t, S) in enumerate(zip(motion_R, motion_t, motion_S)):
        print(f"step{idx}")
        if S is False:
            print(f"Fail at {idx+1} Step. Break.")
            break
        # pos: (3,)
        # R: (1, 3, 3)
        # t: (1, 3, 1)
        nxt = (np.matmul(R[0], pos[:, np.newaxis]) + t[0]).squeeze(-1)
        mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end = get_arrow(pos, nxt)
        mesh_arrow.paint_uniform_color(color_motion) 
        vis.add_geometry(mesh_arrow)
        mesh_sphere_end.paint_uniform_color(color_motion)
        vis.add_geometry(mesh_sphere_end)
        
        # mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end = get_arrow(pos-pos_org, nxt-pos_org)
        # mesh_arrow.paint_uniform_color(color_motion) 
        # vis.add_geometry(mesh_arrow)
        # mesh_sphere_end.paint_uniform_color(color_motion)
        # vis.add_geometry(mesh_sphere_end)
        # pos = nxt


    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame)

    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])  
    view_ctl.set_lookat([0, 0, 0])  
    view_ctl.set_up([0, -1, 0]) 
    view_ctl.set_zoom(0.5)  
    vis.update_renderer()
    vis.poll_events()
    vis.run()
    vis.destroy_window()


    print("#"*20 + " VISUALIZATION END " + "#"*20)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('KPST training')
    parser.add_argument('-r', '--result_path', type=str, default='franka_exec.pkl')
    parser.add_argument('-n', '--max_traj', type=int, default=48)

    args, opts = parser.parse_known_args()

    data = load_pickle(args.result_path)
    data['pcd'][:, 3:] = data['pcd'][:, 3:] * 255.0
    visualization_exec(args, data)


# python vis.py -r C:\Users\86151\Desktop\KPST_Test\exp
