import argparse
import yaml
import os
import pdb
import torch
import json
import random
import logging
from tabulate import tabulate

import numpy as np


from collections import OrderedDict, defaultdict
from easydict import EasyDict as edict

from openpoints.utils import cal_model_parm_nums
from openpoints.utils import EasyConfig

from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataset_from_cfg, build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg

from engine import stochastic_eval, get_train_loss, validate
from util import save_pickle, load_pickle, load_easyconfig_from_yaml

@torch.no_grad()
def get_prediction(data, model, inference_num, n_query_default=256):

    pos, feat = data['pos'], data['x']                         
    feat = torch.concat([feat, pos], axis=-1)                     
    dtraj = data['dtraj']                                          # (Q, T=5, 3)        
    text_feat = data['text_feat']                
    if dtraj.ndim == 2:      # (Q, 3)
        query_np = dtraj
        target_label = None 
    elif dtraj.ndim == 3:    # (Q, T, 3), eg. HOI4D_KPST
        query_np = dtraj[:, 0, :]      
        target_label = dtraj                     

    pos = pos.unsqueeze(0).cuda().float()                      # (1, N, 3)
    feat = feat.unsqueeze(0).cuda().float()                    # (1, N, 6)
    query = query_np.unsqueeze(0).cuda().float()               # (1, Q, 3)
    text_feat = text_feat.unsqueeze(0).cuda().float()          # (1, Ft)

    traj_prediction = model.inference(pos, feat, text_feat, query, num_sample=inference_num).squeeze(0)
    traj_prediction = traj_prediction.transpose(0, 1)                     # (M, Q, T-1=4, 3)
    traj_prediction = traj_prediction.detach().cpu().numpy()              # (M, Q, T-1=4, 3)
    np_qry = np.expand_dims(np.expand_dims(query_np, 0),-2).repeat(traj_prediction.shape[0], 0)
    traj_prediction = np.concatenate([np_qry, np.array(traj_prediction)], -2)
    
    return traj_prediction, target_label                                  # (M, Q, T=5, 3)


def verify_hoi4d_train_loader(args, cfg, model):
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=False,   
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    train_ade, train_fde = validate(model, train_loader, cfg, method='mean')
    print("#"*20 + "[Verify the training error]" + "#"*20)
    print(f"train_ade: {train_ade}")
    print(f"train_fde: {train_fde}")
    print("#"*20 + "[Verify the training error]" + "#"*20)


def verify_hoi4d_valid_loader(args, cfg, model):
    val_loader = build_dataloader_from_cfg(cfg.val_batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=False,  
                                           )
    logging.info(f"length of training dataset: {len(val_loader.dataset)}")
    val_ade, val_fde = validate(model, val_loader, cfg, method='mean')
    print("#"*20 + "[Verify the VAL ERROR]" + "#"*20)
    print(f"val_ade: {val_ade}")
    print(f"val_fde: {val_fde}")
    print("#"*20 + "[Verify the VAL ERROR]" + "#"*20)


def verify_hoi4d_test_loader(args, cfg, model):
    test_loader = build_dataloader_from_cfg(cfg.val_batch_size,
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=False,   # (TODO) Watch-out !! Here we only use 1 gpu mode.
                                            )
    logging.info(f"length of training dataset: {len(test_loader.dataset)}")
    test_ade, test_fde = validate(model, test_loader, cfg, method='mean')
    print("#"*20 + "[Verify the TEST ERROR]" + "#"*20)
    print(f"test_ade: {test_ade}")
    print(f"test_fde: {test_fde}")
    print("#"*20 + "[Verify the TEST ERROR]" + "#"*20)
    return test_ade, test_fde


def inference_hoi4d(args, cfg, model):

    if args.pretrained_path is None:
        args.pretrained_path = 'log/kpst_hoi4d/None/checkpoint/ckpt'
    # eg. 
    # args.pretrained_path = log/kpst_hoi4d/train-pointnext-b-20231116220628-bs-SgCazgBtxUwCjoJkVvfUZi/checkpoint/ckpt_best_train-pointnext-b-20231116220628-bs-SgCazgBtxUwCjoJkVvfUZi.pth
    save_root = os.path.join(args.save_dir, "HOI4D", args.pretrained_path.split('/')[-3])
    if args.choose_class is not None:
        save_root = save_root + '_' + '_'.join(args.choose_class)
    os.makedirs(save_root, exist_ok=True)

    if cfg.model.get('cvae_args', None) is not None:
        if cfg.model.cvae_args.get('scale_method', None) is not None:
            cfg.dataset.common.scale_method = cfg.model.cvae_args.scale_method
        if cfg.model.cvae_args.get('vae_args', None) is not None:
            if cfg.model.cvae_args.vae_args.get('scale_cls_range', None) is not None:
                cfg.dataset.common.scale_cls_range = cfg.model.cvae_args.vae_args.scale_cls_range
    
    if args.origin_pcd is True:
        cfg.val_batch_size = 1
    
    cfg.dataset.train.choose_class = args.choose_class
    cfg.dataset.val.choose_class = args.choose_class
    cfg.dataset.test.choose_class = args.choose_class

    ########################## Verify the Results of HOI4D #########################
    # data_transform = build_transforms_from_cfg('train', cfg.datatransforms)
    # split_cfg = cfg.dataset.get('train', edict())
    # verify_hoi4d_train_loader(args, cfg, model)
    # pdb.set_trace()

    # data_transform = build_transforms_from_cfg('val', cfg.datatransforms)
    # split_cfg = cfg.dataset.get('val', edict())
    # verify_hoi4d_valid_loader(args, cfg, model)

    data_transform = build_transforms_from_cfg('test', cfg.datatransforms)
    split_cfg = cfg.dataset.get('test', edict())
    verify_hoi4d_test_loader(args, cfg, model)

    model.eval()
    data_transform = build_transforms_from_cfg('test', cfg.datatransforms)
    split_cfg = cfg.dataset.get('test', edict())
    split_cfg.transform = data_transform
    dataset = build_dataset_from_cfg(cfg.dataset.common, split_cfg)
    print(f"Length of HOI4D-Test-Dataset: {len(dataset)}")

    def inference_hoi4d_idx(idx, save_fp, inference_num=20):
        data, pcd, part_list, desc = dataset.get_all_query(idx, origin=args.origin_pcd)
        traj_prediction, _ = get_prediction(data, model, inference_num)     # (M, Q, T, 3)
        target_label = np.array(data['dtraj'])                              # (Q, T, 3)
        # pdb.set_trace()
        ade, fde, ade_idx, fde_idx = stochastic_eval(
            # watch out !, need to exclude the query_points.
            torch.from_numpy(traj_prediction[:, :, 1:, :]).transpose(0, 1).unsqueeze(0), 
            torch.from_numpy(target_label[:, 1:, :]).unsqueeze(0),
            return_idx=True)      
        ade_idx = np.array(ade_idx).reshape(-1)
        fde_idx = np.array(fde_idx).reshape(-1)
        save_fp = os.path.join(save_fp, str(idx)+'_n'+str(inference_num)+'.pkl')
        # pdb.set_trace()
        result = {
            'description': desc,
            'model': args.pretrained_path,
            'inference_num': inference_num,
            'min_ade': ade.item(),
            'min_ade_idx': ade_idx[0],
            'min_fde': fde.item(),
            'min_fde_idx': fde_idx[0],
            'traj_prediction': traj_prediction,    # (M, Q, T, 3)
            'traj_target':     target_label,       # (Q, T, 3)
            'pcd': pcd,                            # (N, 6)
            'part_list': part_list                 # (N, 1)
        }
        print("[IDX: {}]  | MIN-ADE: {:.6f} | MIN-FDE: {:.6f} | Infer_Num={}".format(idx, ade, fde, inference_num))
        save_pickle(save_fp, result)
    
    # def loop_get(n_sample):
    #     n_all = len(dataset)
    #     interval = n_all // n_sample
    #     for idx in range(0, n_all, interval):
    #         inference_hoi4d_idx(idx, save_root, inference_num=args.inference_sample_num)

    # pdb.set_trace()
    inference_hoi4d_idx(args.inference_idx, save_root, inference_num=args.inference_sample_num)


def load_model(args, cfg):
    model = build_model_from_cfg(cfg.model)
    model_size = cal_model_parm_nums(model)

    if args.not_load is False:
    
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        ckpt_state = checkpoint['model']
        model_dict = model.state_dict()

        is_model_multi_gpus = True if list(model_dict)[0].split('.')[0] == 'module' else False
        is_ckpt_multi_gpus = True if list(ckpt_state)[0].split('.')[0] == 'module' else False

        if not (is_model_multi_gpus == is_ckpt_multi_gpus):
            temp_dict = OrderedDict()
            for k, v in ckpt_state.items():
                if is_ckpt_multi_gpus:
                    name = k[7:]  # remove 'module.'
                else:
                    name = 'module.' + k  # add 'module'
                temp_dict[name] = v
            ckpt_state = temp_dict

        model.load_state_dict(ckpt_state)
    else:
        print(f"args.not_load is True, use model without weight loading explicitly.")
    
    # since we use pointnet2_cuda module, we have to conduct inference on gpu device.
    model.eval()
    print("=> loaded successfully '{}')".format(args.pretrained_path))
    print("=> model size: ['{:.4f}' M])".format((model_size / 1e6)))

    return model


def inference(args, cfg):
    
    model = load_model(args, cfg)
    model.cuda()
    if args.vis_target == 'HOI4D':
        inference_hoi4d(args, cfg, model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('KPST inference')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    parser.add_argument('--save_dir', type=str, default='results', help='the file-path to save the inference results')

    # eg.
    # log/kpst_hoi4d/train-pointnext-b-20231118160409-rl_lr1e3-n8kwRunLKWGD4poN98Jb62/checkpoint/ckpt_best_train-pointnext-b-20231118160409-rl_lr1e3-n8kwRunLKWGD4poN98Jb62.pth
    parser.add_argument('-p', '--pretrained_path', type=str, default=None)
    parser.add_argument('-v', '--vis_target', type=str, default='HOI4D', help='the datasets to visualization')
    parser.add_argument('-i', '--inference_idx', type=int, default=0, help='the index of the inference data')
    parser.add_argument('-n', '--inference_sample_num', type=int, default=10)
    parser.add_argument('-c', '--choose_class', type=str, nargs='+', default=None)
    parser.add_argument('-o', '--origin_pcd', action='store_true', default=False, help='set to True to use the original pcd data')
    
    parser.add_argument('--not_load', action="store_true", default=False, help="Not Load Weight from args.pretrained_path.")
    parser.add_argument('--cfg', default=None, help="config filepath, if None, try to get cfg.yaml from args.pretrained_path.")
    args, opts = parser.parse_known_args()

    # category_class = ["Toy Car", "Mug", "Laptop", "Storage Furniture", "Bottle", "Safe",
    #                   "Bowl", "Bucket", "Scissors", "Pliers", "Kettle", "Knife", "Trash Can",
    #                   "Lamp", "Stapler", "Chair"]

    if args.cfg is None:
        cfg_fp = args.pretrained_path.split('/')[:-2] + ['cfg.yaml']
        args.cfg = '/'.join(cfg_fp)
        print(f"cfg_file_path={args.cfg}")
        cfg = load_easyconfig_from_yaml(args.cfg)
    else:
        cfg = EasyConfig()
        cfg.load(args.cfg, recursive=True)
    
    if args.origin_pcd:
        print("Use Original PCD Data")
    cfg.update(opts)
    if cfg.seed is None: cfg.seed = 0

    # pdb.set_trace()
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        # cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
        # f'ngpus{cfg.world_size}',
        # f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    inference(args, cfg)


# CUDA_VISIBLE_DEVICES=0 python inference.py -i 0 -n 10 -p log/kpst_hoi4d/ScaleGFlow-B/checkpoint/ckpt_best_train_scalegflow_b.pth
# CUDA_VISIBLE_DEVICES=1 python inference.py -i 0 -n 10 -p log/kpst_hoi4d/ScaleGFlow-S/checkpoint/ckpt_best_train_scalegflow_s.pth
# CUDA_VISIBLE_DEVICES=2 python inference.py -i 0 -n 10 -p log/kpst_hoi4d/ScaleGFlow-L/checkpoint/ckpt_best_train_scalegflow_l.pth