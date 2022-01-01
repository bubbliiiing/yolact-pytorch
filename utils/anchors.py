from itertools import product
from math import sqrt

import numpy as np


def make_anchors(conv_h, conv_w, scale, input_shape=[550, 550], aspect_ratios=[1, 1 / 2, 2]):
    prior_data = []
    for j, i in product(range(conv_h), range(conv_w)):
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / input_shape[1]
            h = scale / ar / input_shape[0]

            prior_data += [x, y, w, h]

    return prior_data

#---------------------------------------------------#
#   用于计算共享特征层的大小
#---------------------------------------------------#
def get_img_output_length(height, width):
    filter_sizes    = [7, 3, 3, 3, 3, 3, 3]
    padding         = [3, 1, 1, 1, 1, 1, 1]
    stride          = [2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths  = []

    for i in range(len(filter_sizes)):
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-5:], np.array(feature_widths)[-5:]
    
def get_anchors(input_shape = [550, 550], anchors_size = [24, 48, 96, 192, 384]):
    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])
    
    all_anchors = []
    for i in range(len(feature_heights)):
        anchors     = make_anchors(feature_heights[i], feature_widths[i], anchors_size[i], input_shape)
        all_anchors += anchors
    
    all_anchors = np.reshape(all_anchors, [-1, 4])
    return all_anchors

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def decode_boxes(pred_box, anchors, variances = [0.1, 0.2]):
        #---------------------------------------------------------#
        #   anchors[:, :2] 先验框中心
        #   anchors[:, 2:] 先验框宽高
        #   对先验框的中心和宽高进行调整，获得预测框
        #---------------------------------------------------------#
        boxes = np.concatenate((anchors[:, :2] + pred_box[:, :2] * variances[0] * anchors[:, 2:], 
                                anchors[:, 2:] * np.exp(pred_box[:, 2:] * variances[1])), 1)

        return boxes

    input_shape = [550, 550]
    anchors = get_anchors(input_shape)
    print(anchors)

    anchors = anchors * np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    anchors = anchors[-75:, :]
    print(anchors)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.ylim(-350,950)
    plt.xlim(-350,950)
    plt.scatter(anchors[:, 0], anchors[:, 1])

    rect1 = plt.Rectangle([anchors[0, 0] - anchors[0, 2] / 2, anchors[0, 1] - anchors[0, 3] / 2], anchors[0, 2], anchors[0, 3], color="r", fill=False)
    rect2 = plt.Rectangle([anchors[1, 0] - anchors[1, 2] / 2, anchors[1, 1] - anchors[1, 3] / 2], anchors[1, 2], anchors[1, 3], color="r", fill=False)
    rect3 = plt.Rectangle([anchors[2, 0] - anchors[2, 2] / 2, anchors[2, 1] - anchors[2, 3] / 2], anchors[2, 2], anchors[2, 3], color="r", fill=False)
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    ax = fig.add_subplot(122)
    plt.ylim(-350,950)
    plt.xlim(-350,950)
    plt.scatter(anchors[:, 0], anchors[:, 1])

    decode_box = decode_boxes(np.random.uniform(0, 2, np.shape(anchors)), anchors)
    rect1 = plt.Rectangle([decode_box[0, 0] - decode_box[0, 2] / 2, decode_box[0, 1] - decode_box[0, 3] / 2], decode_box[0, 2], decode_box[0, 3], color="r", fill=False)
    rect2 = plt.Rectangle([decode_box[1, 0] - decode_box[1, 2] / 2, decode_box[1, 1] - decode_box[1, 3] / 2], decode_box[1, 2], decode_box[1, 3], color="r", fill=False)
    rect3 = plt.Rectangle([decode_box[2, 0] - decode_box[2, 2] / 2, decode_box[2, 1] - decode_box[2, 3] / 2], decode_box[2, 2], decode_box[2, 3], color="r", fill=False)

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    plt.show()