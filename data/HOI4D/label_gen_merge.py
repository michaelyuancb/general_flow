import argparse
import json
import sys
import os
import copy
import pdb 
import random

import numpy as np
from tqdm import tqdm
from pcd_hoi4d.pixel2category import get_mask_and_label
from pcd_hoi4d.scene2frame import get_foreground_depths_and_labels
import random
from scipy.spatial.transform import Rotation as Rt


def main_step5_metad_gen(args):
    
    json_fp = os.path.join(args.output_root, 'json_cache')
    filelist = os.listdir(json_fp)
    json_files = []
    json_ids = []
    for file in filelist:
        with open(os.path.join(json_fp, file), "r") as fp:
            json_files.append(json.load(fp))
        json_ids.append(file.split('_')[3])

    combined = zip(json_ids, json_files)
    sorted_combined = sorted(combined)
    sorted_json_files = [item[1] for item in sorted_combined]
    merge_json = sorted_json_files[0]
    for i in range(1, len(sorted_json_files)):
        merge_json.extend(sorted_json_files[i])
    
    with open(os.path.join(args.output_root, 'metadata_all.json'), 'w') as f:
        json.dump(merge_json, f, indent=4)

    valid_ids = [item['id'] for item in merge_json]

    print("finish meta generation.")
    
    cache_fp = os.path.join(args.output_root, 'cache')
    os.makedirs(cache_fp, exist_ok=True)
    if os.path.exists(os.path.join(args.output_root, 'step1_clips.json')):
        os.system(f'mv {os.path.join(args.output_root, "step1_clips.json")} {cache_fp}')
    if os.path.exists(os.path.join(args.output_root, 'step2_clips_transformation.json')):
        os.system(f'mv {os.path.join(args.output_root, "step2_clips_transformation.json")} {cache_fp}')
    if os.path.exists(os.path.join(args.output_root, 'json_cache')):
        os.system(f'mv {os.path.join(args.output_root, "json_cache")} {cache_fp}')
    
    print("Total Number of KPST-Clips: {}".format(len(valid_ids)))

    # os.system(f'rm -rf {cache_fp}')


def main_step6_train_test_split(args):

    def get_action_stat(meta):
        action_set = set([clip['action'] for clip in meta])
        action_stat = dict()
        for action in action_set:
            action_stat[action] = len([clip for clip in meta if clip['action'] == action])
        return action_stat
    
    with open(os.path.join(args.output_root, 'metadata_all.json'), 'r') as fp:
        metadata = json.load(fp)

    def get_obj_ins(index):
        return index.split('/')[3]

    random.seed(args.seed)
    np.random.seed(args.seed)

    all_train_meta, all_val_meta, all_test_meta = [], [], []
    meta_stat_obj = dict()

    obj_set = set([clip['object'] for clip in metadata])
    print(f"OBJECT SET: {obj_set}")
    for obj in tqdm(obj_set):
        obj_meta = [clip for clip in metadata if clip['object'] == obj]
        obj_index = list(set([clip['index'] for clip in obj_meta]))  

        ins_dict = {}
        for index in obj_index:
            ins = get_obj_ins(index)
            if ins not in ins_dict.keys():
                ins_dict[ins] = [index]
            else:
                ins_dict[ins].append(index)
        ins_list = list(ins_dict.keys())
        random.shuffle(ins_list)
        n_ins = len(ins_list)
        if n_ins < 10:
            print(f"[{obj}] is abandoned, the number of instance is {n_ins} < 10.")
            continue
        else:
            print(f"[{obj}] has {n_ins} instances to split.")
            obj_index = []
            for ins in ins_list[:int(n_ins*0.8)]: obj_index = obj_index + ins_dict[ins]
            n_train_index = len(obj_index)
            for ins in ins_list[int(n_ins*0.8):int(n_ins*0.9)]: obj_index = obj_index + ins_dict[ins]
            n_val_index = len(obj_index) - n_train_index
            for ins in ins_list[int(n_ins*0.9):]: obj_index = obj_index + ins_dict[ins]
            n_test_index = len(obj_index) - n_train_index - n_val_index

        ######### Meta Split Generation #########
        train_index = obj_index[:n_train_index]
        val_index = obj_index[n_train_index:n_train_index + n_val_index]
        test_index = obj_index[n_train_index + n_val_index:]
        train_meta = [clip for clip in obj_meta if clip['index'] in train_index]
        val_meta = [clip for clip in obj_meta if clip['index'] in val_index]
        test_meta = [clip for clip in obj_meta if clip['index'] in test_index]
        # random.shuffle(train_meta)
        # random.shuffle(val_meta)
        # random.shuffle(test_meta)

        obj_stat = dict()
        obj_stat['train'] = {
            'n_index': n_train_index,
            'n_clips': len(train_meta),
            'actions': get_action_stat(train_meta)
        }
        obj_stat['val'] = {
            'n_index': n_val_index,
            'n_clips': len(val_meta),
            'actions': get_action_stat(val_meta)
        }
        obj_stat['test'] = {
            'n_index': n_test_index,
            'n_clips': len(test_meta),
            'actions': get_action_stat(test_meta)
        }
        meta_stat_obj[obj] = obj_stat

        all_train_meta = all_train_meta + train_meta
        all_val_meta = all_val_meta + val_meta
        all_test_meta = all_test_meta + test_meta

    meta_stat = dict()
    meta_data = dict()
    meta_stat['n_train'] = len(all_train_meta)
    meta_stat['n_val'] = len(all_val_meta)
    meta_stat['n_test'] = len(all_test_meta)
    meta_stat['obj_stat'] = meta_stat_obj
    meta_data['train'] = all_train_meta
    meta_data['val'] = all_val_meta
    meta_data['test'] = all_test_meta

    with open(os.path.join(args.output_root, 'metadata.json'), 'w') as f:
        json.dump(meta_data, f, indent=4)
    with open(os.path.join(args.output_root, 'metadata_stat.json'), 'w') as f:
        json.dump(meta_stat, f, indent=4)

    cache_fp = os.path.join(args.output_root, 'cache')
    os.makedirs(cache_fp, exist_ok=True)
    # os.system(f'mv {os.path.join(args.output_root, "metadata_all.json")} {cache_fp}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)

    # parser.add_argument('--output_root', type=str, default='HOI4D_KPST')
    parser.add_argument('--output_root', type=str, default='/home/ycb/HOI4D_KPST')

    args = parser.parse_args()

    print("Start Try to Merge Jsons...")

    main_step5_metad_gen(args)
    main_step6_train_test_split(args)