import argparse
import open3d as o3d
import numpy as np
import random
import pdb
import pickle
# from vis_pip import get_arrow


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
 

def visualization_hoi4d(args):

    result = load_pickle(args.result_path)
    org_pcd = result['pcd']         
    traj_target = result['traj_target']         # (Q, T, 3)
    part = result['part_list'].reshape(-1)      # (N, )
    part_l = list(set(part.reshape(-1).astype(np.int32).tolist()))
    part_n = len(part_l)
    part_num_pt = []
    for pt in part_l:
        part_num_pt.append(np.sum(part == pt))

    print("#"*20 + " SCENE INFO " + "#"*20)
    print(f"N_Points={org_pcd.shape[0]}")
    print(f"N_Parts={part_n}")
    print(f"Parts_Num_Points={part_num_pt}")

    print("#"*20 + " MODEL INFO " + "#"*20)
    print(f"model: {result['model']}")
    print(f"description: {result['description']}")
    print(f"inference_num: {result['inference_num']}")
    print(f"MIN_ADE: {result['min_ade']}  |  MIN_ADE_idx: {result['min_ade_idx']}")
    print(f"MIN_FDE: {result['min_fde']}  |  MIN_FDE_jdx: {result['min_fde_idx']}")
    print("#"*20 + " VISUALIZATION START " + "#"*20)

    if args.num_sample > 0: 
        num_sample = args.num_sample
    else:
        num_sample = min(part_num_pt)

    color_gt = [0, 1, 0]    # green for gt
    color_part = [[1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1]]  # red, cyan, yellow, magenta, blue

    geometry_list = []
    # pdb.set_trace()

    while True: 

        vidx = input(f"please input the index of trajectory (0~{result['inference_num']-1}) to visulization: ")
        vidx = int(vidx)
        if vidx >= result['inference_num']:
            print(f"vidx must be less than {result['inference_num']}, but you input {vidx}. retry again.")
            continue

        traj_prediction = result['traj_prediction'][vidx]  # (Q, T, 3)
        ade = np.mean(np.mean(np.sqrt(np.sum((traj_prediction - traj_target)**2, axis=-1)), axis=-1)) 
        fde = np.mean(np.sqrt(np.sum((traj_prediction[:, -1] - traj_target[:, -1])**2, axis=-1)))
        print("Visualize the {}-th trajectory".format(vidx))
        print("ADE: {:.6f}  |  FDE: {:.6f}".format(ade, fde))
        print(f"Average_len = {np.mean(np.sum(np.linalg.norm(traj_prediction[:, 1:] - traj_prediction[:, :-1], axis=-1), axis=-1), axis=0)}")
        obs_shift = np.array([0.0, 0, 0])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(org_pcd[:, :3] - obs_shift)
        pcd.colors = o3d.utility.Vector3dVector(org_pcd[:, 3:6])
        traj_prediction = traj_prediction - obs_shift
        vis.add_geometry(pcd)
        geometry_list.append(pcd)

        for i, pt in enumerate(part_l):
            part_pred = traj_prediction[part == pt]
            if part_pred.shape[0] <= num_sample:
                selected_trajectories_indices = np.random.choice(len(part_pred), size=num_sample, replace=True) 
            else:
                selected_trajectories_indices = np.random.choice(len(part_pred), size=num_sample, replace=False)

            part_pred = part_pred[selected_trajectories_indices]  # [M, T, 3]
            part_gt = traj_target[part == pt][selected_trajectories_indices]


            for idx_t, (pred_vec, gt_vec)  in enumerate(zip(part_pred, part_gt)):
                for ii in range(pred_vec.shape[0]-1):
                    point2, point1 = pred_vec[ii+1], pred_vec[ii]
                    mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end = get_arrow(point1, point2)
                    # print(color_part[i])
                    mesh_arrow.paint_uniform_color([0, 1, 0]) 
                    vis.add_geometry(mesh_arrow)
                    geometry_list.append(mesh_arrow)
                    if args.only_pred is False:
                        point2, point1 = gt_vec[ii+1], gt_vec[ii]
                        mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end = get_arrow(point1, point2)
                        mesh_arrow.paint_uniform_color([0, 1, 0]) 
                        vis.add_geometry(mesh_arrow)
                        geometry_list.append(mesh_arrow)

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
    parser.add_argument('-r', '--result_path', type=str)
    parser.add_argument('-n', '--num_sample', type=int, default=0)
    parser.add_argument('-v', '--vis_target', type=str, default='HOI4D', help='the datasets to visualization')
    parser.add_argument('-o', '--only_pred', action='store_true', default=False)

    args, opts = parser.parse_known_args()

    if args.vis_target == "HOI4D":
        visualization_hoi4d(args)


# python vis.py -r C:\Users\86151\Desktop\KPST_Test\exp