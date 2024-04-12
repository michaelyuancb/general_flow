###################################################
### CodeBase: https://github.com/ddshan/hand_object_detector/tree/master
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dis import dis
import imageio
import os
import numpy as np
import argparse
import pdb
import cv2
import torch
from PIL import Image

# Please directly build FastRCNN-Lib from ego_hoi_detector (100DOH, CVPR2020 Oral).
from model.utils.viz_hand_obj import *
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet
import pdb
from PIL import Image, ImageDraw, ImageFont

import pdb
from PIL import Image, ImageDraw, ImageFont
import tqdm
xrange = range  # Python 3


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
          in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def create_gif(image_list, gif_name):
 
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)


def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]
def filter_object(obj_dets, hand_dets):
    filtered_object = []
    object_cc_list = []
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)
    img_obj_id = []

    for i in range(hand_dets.shape[0]):
        # if hand_dets[i, 5] <= 0:
        #     img_obj_id.append(-1)
        #     continue
        hand_cc = np.array(calculate_center(hand_dets[i,:4]))
        point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argsort(dist)
        c=0
        indx = dist_min[c]
        
        while indx in img_obj_id and c<len(dist_min)-1:
            c+=1
            indx = dist_min[c]
        img_obj_id.append(indx)
    
    return img_obj_id

def vis_detections_PIL(im, class_name, dets, thresh=0.8, font_path='lib/model/utils/times_b.ttf'):
    """Visual debugging of detections."""
    
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size
    
    for hand_idx, i in enumerate(range(np.minimum(10, dets.shape[0]))):
        bbox = list(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, 4]
        lr = dets[i, -1]
        state = dets[i, 5]
        if score > thresh:
            image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font)
            
    return image

def vis_detections_filtered_objects_PIL(im, obj_dets, hand_dets, thresh_hand=0.8, thresh_obj=0.01, 
                                        font_path='tool_repos/ego_hand_detector/lib/model/utils/times_b.ttf'):

    # pdb.set_trace()
    # convert to PIL
    im = im[:,:,::-1]
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size 

    if (obj_dets is not None) and (hand_dets is not None):
        
        for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
            score = obj_dets[i, 4]
            image = draw_obj_mask(image, draw, obj_idx, bbox, score, width, height, font)

        for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, -1]
            state = hand_dets[i, 5]
            if True:
                # viz hand by PIL
                image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font)

                # if state > 0: # in contact hand

                obj_cc, hand_cc =  calculate_center(obj_dets[i,:4]), calculate_center(bbox)
                # viz line by PIL
                if lr == 0:
                    side_idx = 0
                elif lr == 1:
                    side_idx = 1
                draw_line_point(draw, side_idx, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))
    elif hand_dets is not None:
        image = vis_detections_PIL(im, 'hand', hand_dets, thresh_hand, font_path)
        
    return image


side_map = {'l':'Left', 'r':'Right'}
side_map2 = {0:'Left', 1:'Right'}
side_map3 = {0:'L', 1:'R'}
state_map = {0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}


class EgoHOIDetector(object):
   
    def __init__(self, 
                 cfg_file=None,
                 set_cfgs=None,
                 thresh_hand=0.5, 
                 thresh_obj=0.5,
                 pretrained_path=None, 
                 class_agnostic=False
                 ):
        if cfg_file is None:
            cfg_file = 'tool_repos/ego_hand_detector/cfgs/res101.yml'

        cfg_from_file(cfg_file)
        if set_cfgs is not None:
            cfg_from_list(set_cfgs)

        cfg.USE_GPU_NMS = True if torch.cuda.is_available() else False
        np.random.seed(cfg.RNG_SEED)

        self.class_agnostic = class_agnostic
        self.thresh_hand = thresh_hand
        self.thresh_obj = thresh_obj
        if pretrained_path is None:
            pretrained_path = 'tool_repos/ego_hand_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth'
        self.pretrained_path = pretrained_path
        if not os.path.exists(pretrained_path):
            raise Exception('There is no input directory for loading network from ' + pretrained_path)
        pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
        self.pascal_classes = pascal_classes
        self.fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=class_agnostic)
        self.fasterRCNN.create_architecture()

        print("load checkpoint %s" % (pretrained_path))
        checkpoint = torch.load(pretrained_path)
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        self.fasterRCNN.cuda()
        self.fasterRCNN.eval()
        cfg.CUDA = True


    @torch.no_grad()
    def detect(self, im, vis=False, save_dir='.'):
        # im (image): (H, W, C) numpy array, RGB-format

        # pdb.set_trace() 
        im = im[:, :, ::-1]   # (RGB--->BGR)

        im_data = torch.FloatTensor(1).cuda()
        im_info = torch.FloatTensor(1).cuda()
        num_boxes = torch.LongTensor(1).cuda()
        gt_boxes = torch.FloatTensor(1).cuda()
        box_info = torch.FloatTensor(1) 

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_() 

            pooled_feat,rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, loss_list = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extact predicted params
            contact_vector = loss_list[0][0] # hand contact state info
            offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
            lr_vector = loss_list[2][0].detach() # hand side info (left/right)

            # get hand contact 
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side 
            lr = torch.sigmoid(lr_vector) >= 0.5
            lr = lr.squeeze(0).float()

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    if self.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

                        box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))
            
            pred_boxes /= im_scales[0]
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
        
        if vis:
            im2show = np.copy(im)
        obj_dets, hand_dets = None, None

        for j in xrange(1, len(self.pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            min_conf = 0.3
            
            inds = torch.nonzero(scores[:,j]>min_conf).view(-1)
            while inds.numel() <50 and min_conf>=0:
                min_conf -=5e-2
                inds = torch.nonzero(scores[:,j]>min_conf).view(-1)
            
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              
                if self.pascal_classes[j] == 'targetobject':
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                    cls_dets = cls_dets[order]
                    nms_min = cfg.TEST.NMS

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                    cls_dets = cls_dets[order]
                    cls_feats = pooled_feat[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], nms_min)
                    while len(keep)<2 and nms_min>=0:
                        nms_min -=5e-2
                        keep = nms(cls_boxes[order, :], cls_scores[order], nms_min)
                  
                    cls_dets = cls_dets[keep.view(-1).long()]
                    cls_feats = pooled_feat[keep.view(-1).long()]
                    obj_dets = cls_dets.cpu().numpy()
                    obj_feats = cls_feats.cpu().numpy()
                if self.pascal_classes[j] == 'hand':
                    nms_min = cfg.TEST.NMS
                    hands = {'left':None,'right':None}
                
                    while (hands['left'] is None or hands['right'] is None) and nms_min>=0:
                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                        cls_dets = cls_dets[order]
                        cls_feats = pooled_feat[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], nms_min)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        for i in range(len(cls_dets)):

                            hand_lr = cls_dets[i, -1]
                            if hand_lr>0 and hands['left'] is None:
                                hands['left'] = i
                            if hand_lr<=0 and hands['right'] is None:
                                hands['right'] = i
                        nms_min -=5e-2
                    if hands['left'] is None:
                        hands['left'] = hands['right']
                    if hands['right'] is None:
                        hands['right'] = hands['left']
                    cls_feats = torch.stack([cls_feats[hands['left']],cls_feats[hands['right']]])
                    cls_dets = torch.stack([cls_dets[hands['left']],cls_dets[hands['right']]])
                    # cls_dets = cls_dets[keep.view(-1).long()]
                    # keep = nms(cls_boxes[order, :], cls_scores[order], TEST_NMS)
                    # cls_dets = cls_dets[keep.view(-1).long()]
                    hand_dets = cls_dets.cpu().numpy()
                    hand_feats = cls_feats.cpu().numpy()
        img_obj_id = filter_object(obj_dets, hand_dets)
        obj_dets = obj_dets[img_obj_id]
        obj_feats = obj_feats[img_obj_id]
        # np.savez_compressed(
        #     'images_proc/' + file_name,
        #     obj_dets=obj_dets, 
        #     obj_feats=obj_feats, 
        #     hand_dets=hand_dets,
        #     hand_feat=hand_feats, 
        #     pooled_feat=pooled_feat.cpu().numpy(),
        #     rois=rois.cpu().numpy(),
        #     bbox_pred = bbox_pred.cpu().numpy(),
        #     pred_boxes = pred_boxes.cpu().numpy(),
        #     contact_vector = contact_vector.cpu().numpy(),
        #     offset_vector = offset_vector.cpu().numpy(),
        #     lr_vector = lr_vector.cpu().numpy(),
        #     blobs = blobs,
        #     im_scales=im_scales
        # )
        if vis:
            im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, 0.1, 0.2)
            folder_name = save_dir
            os.makedirs(folder_name, exist_ok=True)
            result_path = os.path.join(folder_name, "vis_det.png")
            im2show.save(result_path)

        return obj_dets, hand_dets
    
        # hand_dets[0 / 1, 9] = {0, 1}      | side label, 0 for left-hand, 1 for right-hand
        # obj_dets[0 / 1, :4]               | <uw, uh, dw, dh>, bbox of interaction object of the corresponding hand.
        # hand_dets[0 / 1, :4]              | <uw, uh, dw, dh>, bbox of the corresponding hand
        # hand_dets[0 / 1, 4]               | contact score of hands 
        # hand_dets[0 / 1, 5]               | hand (contact) state
        # hand_dets[0 / 1, 6:9]             | offset vector ??????

        # state_map = {
        #     0:'No Contact', 
        #     1:'Self Contact', 
        #     2:'Another Person', 
        #     3:'Portable Object', 
        #     4:'Stationary Object'
        # }
