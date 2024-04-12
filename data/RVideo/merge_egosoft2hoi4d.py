import argparse
import json
import copy
import os
import pdb
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('TAP-KPST Label Extraction')
    parser.add_argument('--hoi4d_dir', type=str, default='/home/ycb/HOI4D_KPST')
    parser.add_argument('--egosoft_dir', type=str, default='./Soft_KPST')
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.hoi4d_dir, 'metadata_hoi4d_org.json')):
        hoi4d_metadata = json.load(open(os.path.join(args.hoi4d_dir, 'metadata_hoi4d_org.json'), "rb"))
        hoi4d_metadata_stat = json.load(open(os.path.join(args.hoi4d_dir, 'metadata_stat_hoi4d_org.json'), "rb"))
    else:
        hoi4d_metadata = json.load(open(os.path.join(args.hoi4d_dir, 'metadata.json'), "rb"))
        hoi4d_metadata_stat = json.load(open(os.path.join(args.hoi4d_dir, 'metadata_stat.json'), "rb"))
    if os.path.exists(os.path.join(args.hoi4d_dir, 'max_idx_hoi4d_org.npy')):
        hoi4d_fold_idx_max = int(np.load(os.path.join(args.hoi4d_dir, 'max_idx_hoi4d_org.npy'))[0])
    else:
        hoi4d_fold_idx = os.listdir(os.path.join(args.hoi4d_dir, 'data'))
        hoi4d_fold_idx = [int(x) for x in hoi4d_fold_idx]
        hoi4d_fold_idx_max = max(hoi4d_fold_idx)
        np.save(os.path.join(args.hoi4d_dir, 'max_idx_hoi4d_org.npy'), np.array([hoi4d_fold_idx_max]))

    soft_metadata = json.load(open(os.path.join(args.egosoft_dir, 'metadata_egosoft_demo.json'), "rb"))
    st_soft_idx = hoi4d_fold_idx_max + 1
    soft_ins_split = {
        'train': ['N1', 'N2', 'N3', 'N4'],
        'val': ['N5'],
        'test': ['N6']
    }

    def get_ins_id(index):  # "recorded_rgbd/fold Clothes/N3_3"
        index = index.split('/')[-1]
        index = index.split('_')[0]
        return index

    # if you have more data, you may split it automatically just like what we do in HOI4D.
    metadata = copy.deepcopy(hoi4d_metadata)
    metadata_stat = copy.deepcopy(hoi4d_metadata_stat)
    stat_soft_data = {'train': 0, 'val': 0, 'test': 0}
    for sclip in soft_metadata:
        clip = {
            'id': sclip['id'] + st_soft_idx,
            'index': sclip['index'],
            'action': sclip['action'],
            'object': sclip['object'],
            'img': sclip['st'],
            'kpst_part': ['body']
        }
        ins_id = get_ins_id(sclip['index'])
        for split in ['train', 'val', 'test']:
            if ins_id in soft_ins_split[split]:
                metadata[split].append(clip)
                source_fp = os.path.join(args.egosoft_dir, 'data', str(sclip['id']))
                target_fp = os.path.join(args.hoi4d_dir, 'data', str(clip['id']))
                os.makedirs(target_fp, exist_ok=True)
                os.system(f'cp {source_fp}/* {target_fp}/')
                stat_soft_data[split] = stat_soft_data[split] + 1
                break
    
    pdb.set_trace()
    metadata_stat['soft_data_stat'] = stat_soft_data
    if not os.path.exists(os.path.join(args.hoi4d_dir, 'metadata_hoi4d_org.json')):
        os.system(f"mv {os.path.join(args.hoi4d_dir, 'metadata.json')} {os.path.join(args.hoi4d_dir, 'metadata_hoi4d_org.json')}")
    if not os.path.exists(os.path.join(args.hoi4d_dir, 'metadata_stat_hoi4d_org.json')):
        os.system(f"mv {os.path.join(args.hoi4d_dir, 'metadata_stat.json')} {os.path.join(args.hoi4d_dir, 'metadata_stat_hoi4d_org.json')}")
    fp = os.path.join(args.hoi4d_dir, 'metadata.json')
    json.dump(metadata, open(fp, 'w'), indent=4)
    fp = os.path.join(args.hoi4d_dir, 'metadata_stat.json')
    json.dump(metadata_stat, open(fp, 'w'), indent=4)
    