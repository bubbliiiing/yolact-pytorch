import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-6

def encode(matched, anchors):
    variances = [0.1, 0.2]

    g_cxcy  = (matched[:, :2] + matched[:, 2:]) / 2 - anchors[:, :2]  # (Xg - Xa) / Wa / 0.1
    g_cxcy  /= (variances[0] * anchors[:, 2:]) 
    g_wh    = (matched[:, 2:] - matched[:, :2]) / anchors[:, 2:]  # log(Wg / Wa) / 0.2
    g_wh    = torch.log(g_wh) / variances[1]
    offsets = torch.cat([g_cxcy, g_wh], 1)  # [num_anchors, 4]

    return offsets

def jaccard(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)

    max_xy  = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy  = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter   = torch.clamp((max_xy - min_xy), min=0)
    inter   = inter[:, :, :, 0] * inter[:, :, :, 1]

    area_a  = ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b  = ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union   = area_a + area_b - inter

    out     = inter / (area_a + eps) if iscrowd else inter / (union + eps)
    return out if use_batch else out.squeeze(0)

def match(pos_thresh, neg_thresh, box_gt, anchors, class_gt, crowd_boxes):
    anchors = anchors.data.type_as(box_gt)
    #------------------------------#
    #   获得先验框的左上角和右下角
    #------------------------------#
    decoded_anchors = torch.cat((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)
    #--------------------------------------------#
    #   overlaps [num_objects, num_anchors]
    #--------------------------------------------#
    overlaps        = jaccard(box_gt, decoded_anchors)
    
    #--------------------------------------------#
    #   每个真实框重合程度最大的先验框
    #--------------------------------------------#
    each_box_max, each_box_index        = overlaps.max(1)
    #--------------------------------------------#
    #   每个先验框重合程度最大的真实框以及得分
    #--------------------------------------------#
    each_anchor_max, each_anchor_index  = overlaps.max(0)
    #--------------------------------------------#
    #   保证每个真实框至少有一个对应的
    #--------------------------------------------#
    each_anchor_max.index_fill_(0, each_box_index, 2)

    for j in range(each_box_index.size(0)):
        each_anchor_index[each_box_index[j]] = j

    #--------------------------------------------#
    #   获得每一个先验框对应的真实框的坐标
    #--------------------------------------------#
    each_anchor_box = box_gt[each_anchor_index]
    #--------------------------------------------#
    #   获得每一个先验框对应的种类
    #--------------------------------------------#
    conf            = class_gt[each_anchor_index] + 1
    #--------------------------------------------#
    #   将neg_thresh到pos_thresh之间的进行忽略
    #--------------------------------------------#
    conf[each_anchor_max < pos_thresh] = -1
    conf[each_anchor_max < neg_thresh] = 0

    #--------------------------------------------#
    #   把crowd_boxes部分忽略了
    #--------------------------------------------#
    if crowd_boxes is not None:
        crowd_overlaps                      = jaccard(decoded_anchors, crowd_boxes, iscrowd=True)
        best_crowd_overlap, best_crowd_idx  = crowd_overlaps.max(1)
        conf[(conf <= 0) & (best_crowd_overlap > 0.7)] = -1

    offsets = encode(each_anchor_box, anchors)

    return offsets, conf, each_anchor_box, each_anchor_index

def center_size(boxes):
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)

def crop(masks, boxes, padding: int = 1):
    h, w, n = masks.size()
    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = boxes[:, 1], boxes[:, 3]
    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left  = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up    = cols >= y1.view(1, 1, -1)
    masks_down  = cols < y2.view(1, 1, -1)

    crop_mask   = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float()

class Multi_Loss(nn.Module):
    def __init__(self, num_classes, anchors, pos_thre, neg_thre, negpos_ratio):
        super().__init__()
        self.num_classes    = num_classes
        self.anchors        = anchors
        self.pos_thre       = pos_thre
        self.neg_thre       = neg_thre
        self.negpos_ratio   = negpos_ratio
        
    def forward(self, predictions, targets, mask_gt, num_crowds):
        pred_boxes      = predictions[0]
        pred_classes    = predictions[1]
        pred_masks      = predictions[2]
        pred_proto      = predictions[3]
        anchors         = self.anchors

        batch_size  = pred_boxes.size(0)  # n
        num_anchors = anchors.size(0)  # 19248

        true_offsets        = pred_boxes.new(batch_size, num_anchors, 4)
        true_classes        = pred_boxes.new(batch_size, num_anchors).long()
        class_gt            = [None] * batch_size
        anchor_max_box      = pred_boxes.new(batch_size, num_anchors, 4)
        anchor_max_index    = pred_boxes.new(batch_size, num_anchors).long()

        for i in range(batch_size):
            #------------------------------#
            #   获得框的坐标
            #------------------------------#
            box_gt      = targets[i][:, :-1].data
            #------------------------------#
            #   获得种类
            #------------------------------#
            class_gt[i] = targets[i][:,  -1].data.long()

            cur_crowds = num_crowds[i]
            if cur_crowds > 0:
                mask_gt     = mask_gt[: -cur_crowds]
                box_gt      = box_gt[: -cur_crowds]
                class_gt    = class_gt[: -cur_crowds]
                crowd_boxes = box_gt[-cur_crowds: ]
            else:
                crowd_boxes = None

            #------------------------------------------------------------#
            #   offsets_gts         [batch_size, num_anchors, 4]
            #   conf_gts            [batch_size, num_anchors]
            #   anchor_max_boxes    [batch_size, num_anchors, 4]
            #   anchor_max_indexes  [batch_size, num_anchors]
            #------------------------------------------------------------#
            true_offsets[i], true_classes[i], anchor_max_box[i], anchor_max_index[i] = match(self.pos_thre, self.neg_thre,
                                                                                     box_gt, anchors, class_gt[i], crowd_boxes)

        losses = {}

        positive_bool   = true_classes > 0  
        num_pos         = positive_bool.sum(dim=1, keepdim=True)

        pos_pred_boxes  = pred_boxes[positive_bool, :]
        pos_offsets     = true_offsets[positive_bool, :]
        
        losses['B']     = self.bbox_loss(pos_pred_boxes, pos_offsets) * 1.5
        losses['C']     = self.ohem_conf_loss(pred_classes, true_classes, positive_bool)
        losses['M']     = self.lincomb_mask_loss(positive_bool, pred_masks, pred_proto, mask_gt, anchor_max_box, anchor_max_index) * 6.125
        losses['S']     = self.semantic_segmentation_loss(predictions[4], mask_gt, class_gt)

        total_num_pos   = num_pos.data.sum().float()
        for aa in losses:
            if aa != 'S':
                losses[aa] /= (total_num_pos + eps)
            else:
                losses[aa] /= (batch_size + eps)
        return losses

    #------------------------------------------------------------#
    #   回归损失，利用smooth_l1_loss计算
    #------------------------------------------------------------#
    @staticmethod
    def bbox_loss(pos_pred_boxes, pos_offsets):
        loss_b = F.smooth_l1_loss(pos_pred_boxes, pos_offsets, reduction='sum')
        return loss_b

    #------------------------------------------------------------#
    #   ohem区分难分类样本并计算损失
    #------------------------------------------------------------#
    def ohem_conf_loss(self, pred_classes, true_classes, positive_bool):
        #------------------------------------------------------------------------------------#
        #   batch_conf      batch_size, num_anchors, num_classes -> batch_size * num_anchors, num_classes
        #   batch_conf_max  batch_size * num_anchors, num_classes -> batch_size * num_anchors
        #   mark            batch_size * num_anchors -> batch_size, num_anchors
        #   
        #   mark代表所有负样本的难分类程度。
        #------------------------------------------------------------------------------------#
        batch_conf      = pred_classes.view(-1, self.num_classes)  
        batch_conf_max  = batch_conf.data.max()
        
        mark            = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]
        mark            = mark.view(pred_classes.size(0), -1) 
        #------------------------------------------------------------------------------------#
        #   去除掉正样本和忽略的样本
        #------------------------------------------------------------------------------------#
        mark[positive_bool]     = 0  
        mark[true_classes < 0]  = 0  

        #------------------------------------------------------------------------------------#
        #   idx         batch_size, num_anchors
        #   idx_rank    batch_size, num_anchors
        #------------------------------------------------------------------------------------#
        _, idx      = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        #------------------------------------------------------------------------------------#
        #   positive_bool   batch_size, num_anchors
        #   num_pos         batch_size, 1
        #   num_neg         batch_size, 1
        #------------------------------------------------------------------------------------#
        num_pos         = positive_bool.long().sum(1, keepdim=True)
        num_neg         = torch.clamp(self.negpos_ratio * num_pos, max=positive_bool.size(1) - 1)
        negative_bool   = idx_rank < num_neg.expand_as(idx_rank)

        #------------------------------------------------------------------------------------#
        #   去除掉正样本和忽略的样本
        #------------------------------------------------------------------------------------#
        negative_bool[positive_bool] = 0
        negative_bool[true_classes < 0] = 0  # Filter out neutrals

        #------------------------------------------------------------------------------------#
        #   计算损失并求和
        #------------------------------------------------------------------------------------#
        pred_classes_selected   = pred_classes[(positive_bool + negative_bool)].view(-1, self.num_classes)
        class_gt_selected       = true_classes[(positive_bool + negative_bool)]

        loss_c = F.cross_entropy(pred_classes_selected, class_gt_selected, reduction='sum')

        return loss_c

    @staticmethod
    def lincomb_mask_loss(positive_bool, pred_masks, pred_proto, mask_gt, anchor_max_box, anchor_max_index):
        #-------------------------------------#
        #   计算高、宽
        #   136, 136
        #-------------------------------------#
        proto_h = pred_proto.size(1)
        proto_w = pred_proto.size(2)  

        loss_m = 0
        #-----------------------------------------------#
        #   pred_masks  batch_size, num_anchors, 32
        #   pred_proto  batch_size, 136, 136, 32
        #   mask_gt     
        #-----------------------------------------------#
        for i in range(pred_masks.size(0)):  
            with torch.no_grad():
                #-----------------------------------------------------#
                #   对真实mask进行处理，获得高宽为136, 136的实例mask
                #   h, w, num_objects -> 136, 136, num_objects
                #-----------------------------------------------------#
                downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear', align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
                downsampled_masks = downsampled_masks.gt(0.5).float()

            #-----------------------------------------------------#
            #   取出正样本的参数、坐标和索引
            #   pos_coef            num_pos, 32
            #   pos_anchor_box      num_pos, 4
            #   pos_anchor_index    num_pos
            #-----------------------------------------------------#
            pos_coef            = pred_masks[i, positive_bool[i]]
            pos_anchor_box      = anchor_max_box[i, positive_bool[i]]
            pos_anchor_index    = anchor_max_index[i, positive_bool[i]]  

            #-----------------------------------------------------#
            #   如果不存在正样本，那么跳过到下一个图片
            #-----------------------------------------------------#
            if pos_anchor_index.size(0) == 0:
                continue

            #-----------------------------------------------------#
            #   计算正样本的数量
            #-----------------------------------------------------#
            old_num_pos = pos_coef.size(0)

            #-----------------------------------------------------#
            #   如果正样本的数量大于100，随机选取一下
            #-----------------------------------------------------#
            if old_num_pos > 100:
                perm                = torch.randperm(pos_coef.size(0))
                select              = perm[:100]
                pos_coef            = pos_coef[select]
                pos_anchor_box      = pos_anchor_box[select]
                pos_anchor_index    = pos_anchor_index[select]

            #-----------------------------------------------------#
            #   重新计算正样本的数量
            #   取出每一个先验框对应的mask
            #   136, 136, num_pos
            #-----------------------------------------------------#
            num_pos     = pos_coef.size(0)
            pos_mask_gt = downsampled_masks[:, :, pos_anchor_index]
            pos_anchor_box[:, [0, 2]] = pos_anchor_box[:, [0, 2]] * proto_w
            pos_anchor_box[:, [1, 3]] = pos_anchor_box[:, [1, 3]] * proto_h

            #-----------------------------------------------------#
            #   136, 136, 32 @ 32, num_pos -> 136, 136, num_pos
            #   mask_p          136, 136, num_pos
            #   pos_anchor_box  num_pos, 4
            #-----------------------------------------------------#
            mask_p = pred_proto[i] @ pos_coef.t()
            mask_p = crop(mask_p, pos_anchor_box)  
            
            mask_loss = F.binary_cross_entropy_with_logits(mask_p, pos_mask_gt, reduction='none')
            #-----------------------------------------------------#
            #   每个先验框各自计算平均值
            #-----------------------------------------------------#
            pos_get_csize   = center_size(pos_anchor_box)
            mask_loss       = mask_loss.sum(dim=(0, 1)) / (pos_get_csize[:, 2] + eps) / (pos_get_csize[:, 3] + eps)

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / (num_pos + eps)

            loss_m += torch.sum(mask_loss)
        
        return loss_m / (proto_h + eps) / (proto_w + eps)

    @staticmethod
    def semantic_segmentation_loss(segmentation_p, mask_gt, class_gt):
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()
        loss_s = 0

        for i in range(batch_size):
            cur_segment     = segmentation_p[i]
            cur_class_gt    = class_gt[i]

            with torch.no_grad():
                #-----------------------------------------------------#
                #   对真实mask进行处理，获得高宽为68, 68的实例mask
                #   num_objects, h, w -> num_objects, 68, 68
                #-----------------------------------------------------#
                downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear', align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                #-----------------------------------------------------#
                #   num_classes, 68, 68
                #-----------------------------------------------------#
                segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
                for i_obj in range(downsampled_masks.size(0)):
                    segment_gt[cur_class_gt[i_obj]] = torch.max(segment_gt[cur_class_gt[i_obj]], downsampled_masks[i_obj])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')

        return loss_s / (mask_h + eps) / (mask_w + eps)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
