import argparse
import yaml
import os
import pdb
import logging

import numpy as np


import torch, torch.nn as nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, load_checkpoint_inv, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb

from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from engine import train_one_epoch, validate


def main_train(gpu, cfg, profile=False):

    ######################################### Init #########################################

    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    # logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    # logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        if cfg.model.get('encoder_args', None) is not None:
            cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')
    
    
    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)
    # scheduler = None

    # cfg pass 
    if cfg.model.get('cvae_args', None) is not None:
        if cfg.model.cvae_args.get('scale_method', None) is not None:
            cfg.dataset.common.scale_method = cfg.model.cvae_args.scale_method
        if cfg.model.cvae_args.get('vae_args', None) is not None:
            if cfg.model.cvae_args.vae_args.get('scale_cls_range', None) is not None:
                cfg.dataset.common.scale_cls_range = cfg.model.cvae_args.vae_args.scale_cls_range
    
    if cfg.get('datatrainsforms', None) is None:
        cfg.datastransforms = None

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed,
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )
    logging.info(f"length of test dataset: {len(test_loader.dataset)}")
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    
    traj_length = train_loader.dataset.traj_len if hasattr(
        val_loader.dataset, 'traj_len') else None
    n_query = train_loader.dataset.n_query if hasattr(
        val_loader.dataset, 'n_query') else None
    num_points = train_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    logging.info(f"length of trajectory as training affordance: {traj_length}, "
                 f"number of query points for each training sampling: {n_query}, "
                 f"number of points sampled from trainingdataset as model input: {num_points}."
                 )
    cfg.traj_length = traj_length
    cfg.num_points = num_points
    
    # (TODO) loading pretrain weight
    if cfg.pretrained_path is not None:
        pass
    else:
        logging.info('Training from scratch')

    ######################################### Train #########################################
    
    # pdb.set_trace()

    # val_ade, val_fde = validate(model, val_loader, cfg)
    val_ade, val_fde = 10000., 10000.

    logging.info(f'[INIT]: Valid_ADE={val_ade:.4f}, Valid_FDE={val_fde:.4f}')
    best_val_ade = val_ade
    best_corr_val_fde = val_fde
    best_epoch = -1
    
    model.zero_grad()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        train_loss_record = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg)
        lr = optimizer.param_groups[0]['lr']

        is_best = False
        has_val = False
        if (epoch % cfg.val_freq == 0) and epoch > cfg.val_save:
            val_ade, val_fde = validate(model, val_loader, cfg)
            is_best = val_ade < best_val_ade
            if is_best:
                best_val_ade = val_ade
                best_corr_val_fde = val_fde
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
            has_val = True

        train_log = f'[Epoch {epoch}] LR {lr:.6f} '
        for key, value in train_loss_record.items():
            train_log += f'train_{key}={value:.4f}, '
        logging.info(train_log)
        if has_val:
            logging.info(f'[Epoch {epoch}] LR {lr:.6f} '
                        f'valid_ADE={val_ade:.4f}, valid_FDE={val_fde:.4f}, best_ADE={best_val_ade:.4f}, save_FDE={best_corr_val_fde:.4f}, best_Epoch={best_epoch}')
        
        if writer is not None:
            for key, value in train_loss_record.items():
                writer.add_scalar(f'train_{key}', value, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('best_epoch', best_epoch, epoch)
            writer.add_scalar('best_valid_ade', best_val_ade, epoch)
            writer.add_scalar('save_valid_fde', best_corr_val_fde, epoch)
            if has_val:
                writer.add_scalar('valid_ade', val_ade, epoch)
                writer.add_scalar('valid_fde', val_fde, epoch)
            # writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if (cfg.rank == 0) and (has_val is True):
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val_ade': best_val_ade},
                            is_best=is_best
                            )
            
    ######################################### Test #########################################

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'ckpt_best_{cfg.run_name}.pth'))
    test_ade, test_fde = validate(model, test_loader, cfg)

    train_ade, train_fde = validate(model, train_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_ade', test_ade, best_epoch)
        writer.add_scalar('test_fde', test_fde, best_epoch)
        writer.add_scalar('train_ade', train_ade, best_epoch)
        writer.add_scalar('train_fde', train_fde, best_epoch)

    logging.info(f'[FINAL RESULT] [EPOCH]: BEST_EPOCH={best_epoch}, TOTAL_EPOCH={cfg.epochs}')
    logging.info(f'[FINAL RESULT] [TRAIN]: TRAIN_ADE={train_ade:.4f}, TRAIN_FDE={train_fde:.4f}')
    logging.info(f'[FINAL RESULT] [VAL]: VAL_ADE={best_val_ade:.4f}, VAL_FDE={best_corr_val_fde:.4f}')
    logging.info(f'[FINAL RESULT] [TEST]: TEST_ADE={test_ade:.4f}, TEST_FDE={test_fde:.4f}')

    if writer is not None:
        writer.close()
    
    if cfg.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('KPST training')
    parser.add_argument('--cfg', type=str, default='cfg/kpst_hoi4d/EarlyRegPNX-b.yaml', help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    parser.add_argument('--commit', type=str, default='', help='commit content for logging directory name')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.commit = args.commit
    if cfg.seed is None: cfg.seed = 0

    # pdb.set_trace()
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1
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

    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:  # resume from the existing ckpt and reuse the folder.
        generate_exp_directory(cfg, tags, commit=args.commit)
        cfg.wandb.tags = tags

    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path
    cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main_train, nprocs=cfg.world_size, args=(cfg, args.profile))
    else:
        main_train(0, cfg, profile=args.profile)


# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --lr 0.001 --commit lr1e3 > nul 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --lr 0.005 --commit lr5e3 > nul 2>&1 &

# CUDA_VISIBLE_DEVICES=$GPUs python examples/$task_folder/main.py --cfg $cfg $kwargs
# CUDA_VISIBLE_DEVICES=0 python train.py --commit debug dataloader.num_workers=0


# 点云处理
# (1) Rotation Augmentation如何处理     <仅z轴, 确认一下HOI4D世界坐标系z轴是否垂直地面> [其它indoor-PCD task如何做]
# (2) 点云的缩放如何处理？是否要归一化？  <PointNet Transform Block> <归一化[-1,1]>  
# (3) 点云如何进行Batch训练？            <FPS [alws]>
# (4) 是否要Diff-Traj                   <✔>
# (5) PointNeXt使用旋转不变性与平移不变性 (尤其是在Trajectory Prediction中)  <麻烦, z轴垂直问题>
# (6) Visual Grounding (针对点云的Grounding)   [XXXX]
# (7) Back-Projection和Correspondance代码？    <XXXX> 
# (8) Model Architecture
# (9) Real Robot Experiment                   