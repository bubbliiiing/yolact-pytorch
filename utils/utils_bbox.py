import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms


class BBoxUtility(object):
    def __init__(self):
        pass
    
    def decode_boxes(self, pred_box, anchors, variances = [0.1, 0.2]):
        #---------------------------------------------------------#
        #   anchors[:, :2] 先验框中心
        #   anchors[:, 2:] 先验框宽高
        #   对先验框的中心和宽高进行调整，获得预测框
        #---------------------------------------------------------#
        boxes = torch.cat((anchors[:, :2] + pred_box[:, :2] * variances[0] * anchors[:, 2:],
                        anchors[:, 2:] * torch.exp(pred_box[:, 2:] * variances[1])), 1)

        #---------------------------------------------------------#
        #   获得左上角和右下角
        #---------------------------------------------------------#
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        use_batch = True
        if box_a.dim() == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]

        n = box_a.size(0)
        A = box_a.size(1)
        B = box_b.size(1)

        max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
        min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, :, 0] * inter[:, :, :, 1]

        area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter

        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else out.squeeze(0)

    def fast_non_max_suppression(self, box_thre, class_thre, mask_thre, nms_iou=0.5, top_k=200, max_detections=100):
        #---------------------------------------------------------#
        #   先进行tranpose，方便后面的处理
        #   [80, num_of_kept_boxes]
        #---------------------------------------------------------#
        class_thre      = class_thre.transpose(1, 0).contiguous()
        #---------------------------------------------------------#
        #   [80, num_of_kept_boxes]
        #   每一行坐标为该种类所有的框的得分，
        #   对每一个种类单独进行排序
        #---------------------------------------------------------#
        class_thre, idx = class_thre.sort(1, descending=True) 
        
        idx             = idx[:, :top_k].contiguous()
        class_thre      = class_thre[:, :top_k]

        num_classes, num_dets = idx.size()
        #---------------------------------------------------------#
        #   将num_classes作为第一维度，对每一个类进行非极大抑制
        #   [80, num_of_kept_boxes, 4]    
        #   [80, num_of_kept_boxes, 32]    
        #---------------------------------------------------------#
        box_thre    = box_thre[idx.view(-1), :].view(num_classes, num_dets, 4)
        mask_thre   = mask_thre[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou         = self.jaccard(box_thre, box_thre)
        #---------------------------------------------------------#
        #   [80, num_of_kept_boxes, num_of_kept_boxes]
        #   取矩阵的上三角部分
        #---------------------------------------------------------#
        iou.triu_(diagonal=1)
        iou_max, _  = iou.max(dim=1)

        #---------------------------------------------------------#
        #   获取和高得分重合程度比较低的预测结果
        #---------------------------------------------------------#
        keep        = (iou_max <= nms_iou)
        class_ids   = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

        box_nms     = box_thre[keep]
        class_nms   = class_thre[keep]
        class_ids   = class_ids[keep]
        mask_nms    = mask_thre[keep]

        _, idx      = class_nms.sort(0, descending=True)
        idx         = idx[:max_detections]
        box_nms     = box_nms[idx]
        class_nms   = class_nms[idx]
        class_ids   = class_ids[idx]
        mask_nms    = mask_nms[idx]
        return box_nms, class_nms, class_ids, mask_nms

    def traditional_non_max_suppression(self, box_thre, class_thre, mask_thre, pred_class_max, nms_iou, max_detections):
        num_classes     = class_thre.size()[1]
        pred_class_arg  = torch.argmax(class_thre, dim = -1)

        box_nms, class_nms, class_ids, mask_nms = [], [], [], []
        for c in range(num_classes):
            #--------------------------------#
            #   取出属于该类的所有框的置信度
            #   判断是否大于门限
            #--------------------------------#
            c_confs_m = pred_class_arg == c
            if len(c_confs_m) > 0:
                #-----------------------------------------#
                #   取出得分高于confidence的框
                #-----------------------------------------#
                boxes_to_process = box_thre[c_confs_m]
                confs_to_process = pred_class_max[c_confs_m]
                masks_to_process = mask_thre[c_confs_m]
                #-----------------------------------------#
                #   进行iou的非极大抑制
                #-----------------------------------------#
                idx         = nms(boxes_to_process, confs_to_process, nms_iou)
                #-----------------------------------------#
                #   取出在非极大抑制中效果较好的内容
                #-----------------------------------------#
                good_boxes  = boxes_to_process[idx]
                confs       = confs_to_process[idx]
                labels      = c * torch.ones((len(idx))).long()
                good_masks  = masks_to_process[idx]
                box_nms.append(good_boxes)
                class_nms.append(confs)
                class_ids.append(labels)
                mask_nms.append(good_masks)
        box_nms, class_nms, class_ids, mask_nms = torch.cat(box_nms, dim = 0), torch.cat(class_nms, dim = 0), \
                                                  torch.cat(class_ids, dim = 0), torch.cat(mask_nms, dim = 0)

        idx = torch.argsort(class_nms, 0, descending=True)[:max_detections]
        box_nms, class_nms, class_ids, mask_nms = box_nms[idx], class_nms[idx], class_ids[idx], mask_nms[idx]
        return box_nms, class_nms, class_ids, mask_nms

    def yolact_correct_boxes(self, boxes, image_shape):
        image_size          = np.array(image_shape)[::-1]
        image_size          = torch.tensor([*image_size]).type(boxes.dtype).cuda() if boxes.is_cuda else torch.tensor([*image_size]).type(boxes.dtype)

        scales              = torch.cat([image_size, image_size], dim=-1)
        boxes               = boxes * scales
        boxes[:, [0, 1]]    = torch.min(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [2, 3]]    = torch.max(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [0, 1]]    = torch.max(boxes[:, [0, 1]], torch.zeros_like(boxes[:, [0, 1]]))
        boxes[:, [2, 3]]    = torch.min(boxes[:, [2, 3]], torch.unsqueeze(image_size, 0).expand([boxes.size()[0], 2]))
        return boxes

    def crop(self, masks, boxes):
        h, w, n     = masks.size()
        x1, x2      = boxes[:, 0], boxes[:, 2]
        y1, y2      = boxes[:, 1], boxes[:, 3]

        rows        = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
        cols        = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

        masks_left  = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up    = cols >= y1.view(1, 1, -1)
        masks_down  = cols < y2.view(1, 1, -1)

        crop_mask   = masks_left * masks_right * masks_up * masks_down
        return masks * crop_mask.float()

    def decode_nms(self, outputs, anchors, confidence, nms_iou, image_shape, traditional_nms=False, max_detections=100):
        #---------------------------------------------------------#
        #   pred_box    [18525, 4]  对应每个先验框的调整情况
        #   pred_class  [18525, 81] 对应每个先验框的种类      
        #   pred_mask   [18525, 32] 对应每个先验框的语义分割情况
        #   pred_proto  [128, 128, 32]  需要和结合pred_mask使用
        #---------------------------------------------------------#
        pred_box    = outputs[0].squeeze()       
        pred_class  = outputs[1].squeeze()
        pred_masks  = outputs[2].squeeze()
        pred_proto  = outputs[3].squeeze()

        #---------------------------------------------------------#
        #   将先验框调整获得预测框，
        #   [18525, 4] boxes是左上角、右下角的形式。
        #---------------------------------------------------------#
        boxes       = self.decode_boxes(pred_box, anchors) 

        #---------------------------------------------------------#
        #   除去背景的部分，并获得最大的得分 
        #   [18525, 80]
        #   [18525]
        #---------------------------------------------------------#
        pred_class          = pred_class[:, 1:]    
        pred_class_max, _   = torch.max(pred_class, dim=1)
        keep        = (pred_class_max > confidence)

        #---------------------------------------------------------#
        #   保留满足得分的框，如果没有框保留，则返回None
        #---------------------------------------------------------#
        box_thre    = boxes[keep, :]
        class_thre  = pred_class[keep, :]
        mask_thre   = pred_masks[keep, :]
        if class_thre.size()[0] == 0:
            return None, None, None, None, None

        if not traditional_nms:
            box_thre, class_thre, class_ids, mask_thre = self.fast_non_max_suppression(box_thre, class_thre, mask_thre, nms_iou)
            keep        = class_thre > confidence
            box_thre    = box_thre[keep]
            class_thre  = class_thre[keep]
            class_ids   = class_ids[keep]
            mask_thre   = mask_thre[keep]
        else:
            box_thre, class_thre, class_ids, mask_thre = self.traditional_non_max_suppression(box_thre, class_thre, mask_thre, pred_class_max[keep], nms_iou, max_detections)
        
        box_thre    = self.yolact_correct_boxes(box_thre, image_shape)

        #---------------------------------------------------------#
        #   pred_proto      [128, 128, 32]
        #   mask_thre       [num_of_kept_boxes, 32]
        #   masks_sigmoid   [128, 128, num_of_kept_boxes]
        #---------------------------------------------------------#
        masks_sigmoid   = torch.sigmoid(torch.matmul(pred_proto, mask_thre.t()))
        #----------------------------------------------------------------------#
        #   masks_sigmoid   [image_shape[0], image_shape[1], num_of_kept_boxes]
        #----------------------------------------------------------------------#
        masks_sigmoid   = masks_sigmoid.permute(2, 0, 1).contiguous()
        masks_sigmoid   = F.interpolate(masks_sigmoid.unsqueeze(0), (image_shape[0], image_shape[1]), mode='bilinear', align_corners=False).squeeze(0)
        masks_sigmoid   = masks_sigmoid.permute(1, 2, 0).contiguous()
        masks_sigmoid   = self.crop(masks_sigmoid, box_thre)

        #----------------------------------------------------------------------#
        #   masks_arg   [image_shape[0], image_shape[1]]    
        #   获得每个像素点所属的实例
        #----------------------------------------------------------------------#
        masks_arg       = torch.argmax(masks_sigmoid, dim=-1)
        #----------------------------------------------------------------------#
        #   masks_arg   [image_shape[0], image_shape[1], num_of_kept_boxes]
        #   判断每个像素点是否满足门限需求
        #----------------------------------------------------------------------#
        masks_sigmoid   = masks_sigmoid > 0.5

        return box_thre, class_thre, class_ids, masks_arg, masks_sigmoid

