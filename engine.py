from tqdm import tqdm
import numpy as np 
import torch
import pdb

from openpoints.utils import AverageMeter


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):

    loss_record = None
    
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            if key == 'pack': continue
            data[key] = data[key].cuda().float()
        if 'pack' in data.keys():
            for key in data['pack'].keys():
                data['pack'][key] = data['pack'][key].cuda().float()
        num_iter += 1

        pos = data['pos']                            # (B, N, 3)
        feat = data['x']                             # (B, N, 3)
        feat = torch.concat([feat, pos], dim=-1)     # (B, N, 6) ; rgb+xyz for PointNeXt
        text_feat = data['text_feat']                # (B, Ft)
        dtraj = data['dtraj']                        # (B, Q, T, 3)
        loss_pack = model(pos, feat, text_feat, dtraj, pack=data['pack'])

        loss_grad = loss_pack['loss']

        loss_grad.backward()

        # print(f"STEP-{idx} | loss_pack={loss_pack}")
        if np.isnan(loss_pack['loss'].item()):
            print(idx)
            print("ERROR, NAN-LOSS Happen TAT !")
            pdb.set_trace()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        if loss_record is None:
            loss_record = dict()
            for key in loss_pack.keys():
                loss_record[key] = AverageMeter()
        
        for key, value in loss_pack.items():
            if type(value) is int:
                loss_record[key].update(value)
            else:
                loss_record[key].update(value.item())

    for key, value in loss_record.items():
        loss_record[key] = value.avg

    return loss_record


def get_train_loss(model, train_loader):

    loss_record = None

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            if key == 'pack': continue
            data[key] = data[key].cuda().float()
        num_iter += 1

        pos = data['pos']                            # (B, N, 3)
        feat = data['x']                             # (B, N, 3)
        feat = torch.concat([feat, pos], dim=-1)     # (B, N, 6) ; rgb+xyz for PointNeXt
        text_feat = data['text_feat']                # (B, Ft)
        dtraj = data['dtraj']                        # (B, Q, T, 3)
        loss_pack = model(pos, feat, text_feat, dtraj, pack=data['pack'])

        if loss_record is None:
            loss_record = dict()
            for key in loss_pack.keys():
                loss_record[key] = AverageMeter()
        
        for key, value in loss_pack.items():
            if type(value) is int:
                loss_record[key].update(value)
            else:
                loss_record[key].update(value.item())

    for key, value in loss_record.items():
        loss_record[key] = value.avg

    return loss_record


@torch.no_grad()
def validate(model, val_loader, cfg, method='min'):
    ade_loss = AverageMeter()
    fde_loss = AverageMeter()
    
    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            if key == 'pack': continue
            data[key] = data[key].cuda().float()

        pos = data['pos']                            # (B, N, 3)
        feat = data['x']                             # (B, N, 3)
        feat = torch.concat([feat, pos], dim=-1)     # (B, N, 6) ; xyz+rgb for PointNeXt
        text_feat = data['text_feat']                # (B, Ft)
        dtraj = data['dtraj']                        # (B, Q, T, 3)
        query = dtraj[:, :, 0, :]                    # (B, Q, 3)
        target = dtraj[:, :, 1:, :]                  # (B, Q, T-1=4, 3)

        if hasattr(model, 'inference'):
            traj_prediction = model.inference(pos, feat, text_feat, query, num_sample=cfg.inference_num)         # (B, Q, M, T-1=4, 3)
        else:
            traj_prediction = model.module.inference(pos, feat, text_feat, query, num_sample=cfg.inference_num)  
        # the inference prediction is pos[t] - pos[0]

        ade_stochastic, fde_stochastic = stochastic_eval(traj_prediction, target, method=method)
        ade_loss.update(ade_stochastic)
        fde_loss.update(fde_stochastic)

    return ade_loss.avg, fde_loss.avg


def stochastic_eval(proposal, label, return_idx=False, method='min'):       # (B, Q, M, T-1=4, 3), (B, Q, T-1=4, 3)
    method = 'mean'                                                         # We fixed mean evaluation here.
    label = label.unsqueeze(2)                                              # (B, Q, 1, T-1=4, 3)

    distance = torch.mean(torch.norm(proposal - label, dim=-1, p=2), dim=-1)    # (B, Q, M)
    distance = torch.mean(distance, dim=-2)                                     # (B, M)
    min_distance, ade_idx = torch.min(distance, dim=-1)                         # (B)
    ade = torch.mean(min_distance)
    if method == 'mean':
        ade = torch.mean(torch.mean(distance, dim=-1))

    label_acc = label[:, :, :, -1]             # (B, Q, 1, 3)
    proposal_acc = proposal[:, :, :, -1]       # (B, Q, M, 3)
    distance = torch.norm(label_acc - proposal_acc, dim=-1, p=2)  # (B, Q, M)
    distance = torch.mean(distance, dim=-2)                       # (B, M)
    min_distance, fde_idx = torch.min(distance, dim=-1)           # (B, )
    fde = torch.mean(min_distance)
    if method == 'mean':
        fde = torch.mean(torch.mean(distance, dim=-1))

    if return_idx is True:
        return ade, fde, ade_idx, fde_idx
    else:
        return ade, fde 