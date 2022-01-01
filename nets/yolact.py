import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import ResNet


class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        #----------------------------------#
        #   C3、C4、C5通道数均调整成256
        #----------------------------------#
        self.lat_layers     = nn.ModuleList(
            [
                nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels
            ]
        )

        #----------------------------------#
        #   特征融合后用于进行特征整合
        #----------------------------------#
        self.pred_layers    = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)) for _ in self.in_channels
            ]
        )

        #----------------------------------#
        #   对P5进行下采样获得P6和P7
        #----------------------------------#
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True)
                )
            ]
        )

    def forward(self, backbone_features):
        P5          = self.lat_layers[2](backbone_features[2])
        P4          = self.lat_layers[1](backbone_features[1])
        P3          = self.lat_layers[0](backbone_features[0])

        P5_upsample = F.interpolate(P5, size=(backbone_features[1].size()[2], backbone_features[1].size()[3]), mode='nearest')
        P4          = P4 + P5_upsample

        P4_upsample = F.interpolate(P4, size=(backbone_features[0].size()[2], backbone_features[0].size()[3]), mode='nearest')
        P3          = P3 + P4_upsample

        P5 = self.pred_layers[2](P5)
        P4 = self.pred_layers[1](P4)
        P3 = self.pred_layers[0](P3)

        P6 = self.downsample_layers[0](P5)
        P7 = self.downsample_layers[1](P6)

        return P3, P4, P5, P6, P7

class ProtoNet(nn.Module):
    def __init__(self, coef_dim):
        super().__init__()
        self.proto1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.proto2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, coef_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.proto1(x)
        x = self.upsample(x)
        x = self.proto2(x)
        return x


class PredictionModule(nn.Module):
    def __init__(self, num_classes, coef_dim=32, aspect_ratios = [1, 1 / 2, 2]):
        super().__init__()
        self.num_classes    = num_classes
        self.coef_dim       = coef_dim

        self.upfeature = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.bbox_layer = nn.Conv2d(256, len(aspect_ratios) * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(256, len(aspect_ratios) * self.num_classes, kernel_size=3, padding=1)
        self.coef_layer = nn.Sequential(
            nn.Conv2d(256, len(aspect_ratios) * self.coef_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        bs      = x.size(0)

        x       = self.upfeature(x)
        box     = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(bs, -1, 4)
        conf    = self.conf_layer(x).permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes)
        coef    = self.coef_layer(x).permute(0, 2, 3, 1).reshape(bs, -1, self.coef_dim)
        return box, conf, coef

class Yolact(nn.Module):
    def __init__(self, num_classes, coef_dim=32, pretrained=False, train_mode=True):
        super().__init__()
        #----------------------------#
        #   获得的C3为68, 68, 512
        #   获得的C4为34, 34, 1024
        #   获得的C5为17, 17, 2048
        #----------------------------#
        self.backbone               = ResNet(layers=[3, 4, 6, 3])
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/resnet50_backbone_weights.pth"))

        #----------------------------#
        #   获得的P3为68, 68, 256
        #   获得的P4为34, 34, 256
        #   获得的P5为17, 17, 256
        #   获得的P6为9, 9, 256
        #   获得的P7为5, 5, 256
        #----------------------------#
        self.fpn                    = FPN([512, 1024, 2048])
        
        #--------------------------------#
        #   对P3进行上采样
        #   256, 68, 68 -> 32, 136, 136
        #--------------------------------#
        self.proto_net              = ProtoNet(coef_dim=coef_dim)
        #--------------------------------#
        #   用于获取每一个有效特征层
        #   对应的预测结果
        #--------------------------------#
        self.prediction_layers      = PredictionModule(num_classes, coef_dim=coef_dim)
        self.semantic_seg_conv      = nn.Conv2d(256, num_classes - 1, kernel_size=1)

        self.train_mode             = train_mode

    def forward(self, x):
        '''
        主干特征提取网络获得三个初步特征 (n, 512, 68, 68)
                                        (n, 1024, 34, 34)
                                        (n, 2048, 17, 17)
        '''
        features = self.backbone(x)
        '''
        构建特征金字塔，获得五个有效特征层 (n, 256, 68, 68) P3
                                          (n, 256, 34, 34) P4
                                          (n, 256, 17, 17) P5
                                          (n, 256, 9, 9)   P6
                                          (n, 256, 5, 5)   P7
        '''
        features = self.fpn.forward(features)
        #---------------------------------------------------#
        #   对P3进行上采样
        #   256, 68, 68 -> 32, 136, 136 -> 136, 136, 32
        #---------------------------------------------------#
        pred_proto = self.proto_net(features[0])  
        pred_proto = pred_proto.permute(0, 2, 3, 1).contiguous()

        #--------------------------------------------#
        #   将5个特征层利用同一个head的预测结果堆叠
        #   pred_boxes      18525, 4
        #   pred_classes    18525, 81
        #   pred_masks      18525, 32
        #--------------------------------------------#
        pred_boxes, pred_classes, pred_masks = [], [], []
        for f_map in features:
            box_p, class_p, mask_p = self.prediction_layers(f_map)
            pred_boxes.append(box_p)
            pred_classes.append(class_p)
            pred_masks.append(mask_p)
        pred_boxes      = torch.cat(pred_boxes, dim=1)
        pred_classes    = torch.cat(pred_classes, dim=1)
        pred_masks      = torch.cat(pred_masks, dim=1)

        if self.train_mode:
            #--------------------------------------------#
            #   256, 68, 68 -> num_classes - 1, 68, 68
            #--------------------------------------------#
            pred_segs   = self.semantic_seg_conv(features[0])
            return pred_boxes, pred_classes, pred_masks, pred_proto, pred_segs
        else:
            pred_classes = F.softmax(pred_classes, -1)
            return pred_boxes, pred_classes, pred_masks, pred_proto
