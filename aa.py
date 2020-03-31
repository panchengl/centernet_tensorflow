# # down_ratio = 9
# # assert down_ratio in [2, 4, 8, 16]
# #
# # w = 199
# # h = 200
# # batch = 2
# import numpy as np
# # img_size = np.zeros((batch, 2))
# # # print(img_size)
# # # for i in range(batch):
# # #     # img_size[i:] = w, h
# # #     print(img_size[i:])
# #
# # img_size[0,:] = w, h
# # # img_size[1:] = w+10, h+10
# # print(img_size)
# # # print(img_size)
#
# line = '/home/pcl/tf_work/TF_CenterNet//VOC/train/VOCdevkit/VOC2012/JPEGImages/2010_003186.jpg 500 375 69,172,270,330,12 150,141,229,284,14 285,201,327,331,14 258,198,297,329,14'
# # a = line.split()
# s = line.strip().split(' ')
# line_idx = 0
# pic_path = s[0]
# img_width = int(s[1])
# img_height = int(s[2])
# s = s[3:]
#
# box_cnt = len(s)
# boxes = []
# labels = []
# gt_labels = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in s])
# # print(labels)
# for idx, label in enumerate(gt_labels):
#     box = label[:4]
#     class_name = label[4]
#     class_name, x_min, y_min, x_max, y_max = label[4], label[0], label[1], label[2], label[3]
#     boxes.append([x_min, y_min, x_max, y_max])
#     labels.append(class_name)
# boxes = np.asarray(boxes, np.float32)
# labels = np.asarray(labels, np.int64)
#
# print(line_idx, pic_path, boxes, labels, img_width, img_height)

# def pan(input):
#     print(input)
#
# class a():
#     def __init__(self):
#         self.print = pan
#     def forward(self, num):
#         self.print(num)
#
# b = a()
# b.forward(num="pcl")

# target_size = 512, 512
# ih, iw = target_size
# h, w, _ = 1080, 1920, 3
#
# scale = min(iw / w, ih / h)
# nw, nh = int(scale * w), int(scale * h)
# print(scale)
# print(nw, nh)
from utils.image import gaussian_radius, draw_umich_gaussian
import math
import numpy as np
import cv2

output_h, output_w = 128, 128
max_objs = 100
num_classes = 20
hm = np.zeros((output_h, output_w, num_classes),dtype=np.float32)
wh = np.zeros((max_objs, 2),dtype=np.float32)
reg = np.zeros((max_objs, 2),dtype=np.float32)
ind = np.zeros((max_objs),dtype=np.float32)
reg_mask = np.zeros((max_objs),dtype=np.float32)

down_ratio = 4
label = np.array([10, 30, 50, 100, 1])
idx = 0
bbox = label[:4] / down_ratio
class_id = label[4]
h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
print("w h is", w, h)
radius = gaussian_radius((math.ceil(h), math.ceil(w)))
print("radius is", radius)
radius = max(0, int(radius))
print(radius)
ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
ct_int = ct.astype(np.int32)
print("ct is", ct)
draw_umich_gaussian(hm[:, :, class_id], ct_int, radius)
print(hm)
cv2.imwrite("/home/pcl/tf_work/TF_CenterNet/single_heatmap.jpg", hm[0]*255)
wh[idx] = 1. * w, 1. * h
ind[idx] = ct_int[1] * output_w + ct_int[0]
reg[idx] = ct - ct_int
reg_mask[idx] = 1
print("ind is", ind)
print("wh is", wh)
print("reg is", reg)
