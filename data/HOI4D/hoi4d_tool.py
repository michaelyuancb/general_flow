from scipy.spatial.transform import Rotation as Rt
from tqdm import tqdm
import numpy as np
import argparse
import pdb
import json
import os
from plyfile import PlyData, PlyElement

def read_rtd(anno):
    trans, rot, dim = anno["center"], anno["rotation"], anno["dimensions"]
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    rot = np.array([rot['x'], rot['y'], rot['z']])
    dim = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
    rot = Rt.from_euler('XYZ', rot).as_rotvec()

    return np.array(rot, dtype=np.float32), trans, dim


def transform_vec2mat(rot, trans):
    rotation_matrix = Rt.from_rotvec(rot).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = trans
    return transformation_matrix

def transform_mat2vec(transformation_matrix):
    trans = transformation_matrix[:3, 3]
    rot = Rt.from_matrix(transformation_matrix[:3, :3]).as_rotvec()
    return rot, trans


def num2d(x):
    if x > 99999:
        raise ValueError("x should be less than 99999")
    else:
        return "0" * (5 - len(str(x))) + str(x)


def get_class(index):
    index = index.split("/")
    return index[2]

# from object_pose label to object_motion label
obj2mseg_dict = {
        'C1': {'Toycar': [(1, 'Toy Car')]},
        'C2': {'Watercup': [(1, 'Mug'), (3, 'Mug')]},
        'C3': {'Laptopdisplay': [(1, 'Screen')], 
               'Laptopkeyboard': [(3, 'Keyboard')]},

        'C4': {'Lockerdrawer': [(4, 'Drawer')], 
               'Lockersldingdoor': [(1, 'Door')],
               'Lockerbody': [(3, 'Body of the Cabinet')]
               },       # (TODO) Need to Check Later. 

        'C5': {'bottleddrinks': [(1, 'Bottle')]},
        'C6': {'Safebox': [(1, 'Body of the Safe')],
               'Safedoor': [(3, 'Door')]},
        'C7': {'bowl': [(1, 'Bowl')]},
        'C8': {'bucket': [(1, 'Body of the Bucket'), (4, 'Body of the Bucket')],
               'Buckethandle': [(3, 'Handle'), (5, 'Handle')]},
        'C9': {},       # Scissors is not included.  (not know how to project between 1-2 & left-right)
        'C10': {},      # C10 is not released.
        'C11': {'Pliersleft': [(1, 'Left Part of the Pliers')],
                'Pliersright': [(3, 'Right Part of the Pliers')]},
        'C12': {'kettle': [(1, 'Kettle')]},
        'C13': {'knife': [(1, 'Knife')]},
        'C14': {"Dustbinbase": [(1, 'Body of the Trash Can')],
                "Dustbincover": [(3, 'Lid')]},
        'C15': {},      # C15 is not released.
        'C16': {},      # C16 is not released.
        'C17': {},      # Lamp is not included. (complex joint, not know how to project)
        'C18': {"Staplercover": [(4, 'Lid')],
                "Staplerbase": [(1, 'Base')]},
        'C20': {'chair': [(1, 'Chair')]}
    }


def objpose2mseg(cls, objpose_label, num):
    if cls not in obj2mseg_dict:
        raise ValueError("Unknown Object Class: {}".format(cls))
    obj_projection = obj2mseg_dict[cls]
    if objpose_label not in obj_projection.keys():
        return None 
    else:
        obj_projection = obj_projection[objpose_label]
        if num > len(obj_projection):
            return None 
        else:
            return obj_projection[num][0], obj_projection[num][1]

def get_all_obj_pose_label(args):
    with open(args.idx_file, "r") as fp:
        idx_list = fp.readlines()
    idx_list = [idx.strip() for idx in idx_list]

    obj_label = {}
    for i in range(20):
        obj_label["C"+str(i+1)] = []

    for idx in tqdm(idx_list):
        cls = get_class(idx)
        try:
            if os.path.exists(args.anno_root + '/' + idx + "/objpose/0.json"):
                fp = os.path.join(args.anno_root, idx, "objpose/0.json")
            else:
                fp = os.path.join(args.anno_root, idx, "objpose/00000.json")
            obj_pose = json.load(open(fp, "r"))
        except Exception as e:
            print(f"An Error Occured: {e}")
        
        if "dataList" in obj_pose.keys():
            obj_pose = obj_pose["dataList"]
        else:
            obj_pose = obj_pose["objects"]
        for obj in obj_pose:
            obj_label[cls].append(obj["label"])

    for key in obj_label.keys():
        obj_label[key] = list(set(obj_label[key]))

    print(obj_label)
    json.dump(obj_label, open(os.path.join(args.output_root, "objpose_label.json"), "w"), indent=4)


def test_vec_mat_transform():
    rot_vec = [0.00830034430548275, 0.024529269327448047, -0.014291375220518421]
    trans_vec = [0.004453085362911224, -0.002241566777229309, -0.009249687194824219]
    
    mat = transform_vec2mat(rot_vec, trans_vec)
    print(mat @ mat.T)
    print(mat.T @ mat)

    rot, trans = transform_mat2vec(mat)

    print(rot)
    print(trans)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/public/datasets_yl/HOI4D/')
    parser.add_argument('--anno_root', type=str, default='/public/datasets_yl/HOI4D_annotations/')
    parser.add_argument('--idx_file', type=str, default='HOI4D-Instructions/release.txt')
    parser.add_argument('--output_root', type=str, default='HOI4D_KPST')
    args = parser.parse_args()

    # get_all_obj_pose_label(args)
    # test_vec_mat_trasnform()

    