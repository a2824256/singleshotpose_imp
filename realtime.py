# import time
# import torch
# from torch.autograd import Variable
from torchvision import datasets, transforms
# from torch.utils.data import Dataset, DataLoader
import scipy.io
import warnings
# from PIL import Image
# import trimesh
# import cv2
import wx
# from pynput import keyboard
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import scipy.misc

from darknet import Darknet
# import dataset
from utils import *
# from MeshPly import MeshPly
# import png
import pyrealsense2 as rs
# import json
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import time
import os

def valid(datacfg, modelcfg, weightfile):
        # 内部函数
        def truths_length(truths, max_num_gt=50):
            for i in range(max_num_gt):
                if truths[i][1] == 0:
                    return i

        # Parse configuration files
        data_options = read_data_cfg(datacfg)
        # 备份权重文件夹
        backupdir = data_options['backup']
        # 选择使用哪个GPU
        gpus = data_options['gpus']
        # 图像尺寸
        im_width = int(data_options['width'])
        im_height = int(data_options['height'])
        # 判断备份文件夹是否存在
        if not os.path.exists(backupdir):
            makedirs(backupdir)

        # ---------------real time detection start---------------
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        pipeline.start(config)
        # profile = pipeline.start(config)
        pipeline.wait_for_frames()
        # frames = pipeline.wait_for_frames()
        # color_frame = frames.get_color_frame()
        # intr = color_frame.profile.as_video_stream_profile().intrinsics
        # camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
        #                      'ppx': intr.ppx, 'ppy': intr.ppy,
        #                      'height': intr.height, 'width': intr.width,
        #                      'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
        #                      }
        align_to = rs.stream.color
        align = rs.align(align_to)
        # ---------------real time detection end---------------
        # Parameters
        seed = int(time.time())
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        # 随机种子
        torch.cuda.manual_seed(seed)
        # 是否可视化显示
        visualize = True
        # 类别数量
        num_classes = 1
        # 边角，绘图连线用，将对应的两个点连成线段
        edges_corners = [[1, 5], [2, 6], [3, 7], [4, 8], [1, 2], [1, 3], [2, 4], [3, 4], [5, 6], [5, 7], [6, 8], [7, 8]]

        preds_corners2D = []

        # 初始化网络
        model = Darknet(modelcfg)
        # 打印网络结构
        # model.print_network()
        # 载入权重
        model.load_weights(weightfile)
        # 使用cuda加速
        model.cuda()
        model.eval()
        num_keypoints = model.num_keypoints
        # label数量 = 关键点x3 + 3
        # num_labels = num_keypoints * 2 + 3
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            exit()

        # d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())
        # cv2.imwrite('test.png', c)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pipeline.stop()
        # ------------------take picture end------------------
        # -----------------------单图start-------------------------------
        # 图片预处理代码
        transform = transforms.Compose([transforms.ToTensor(), ])
        pic_tensor = transform(c)
        # exit()
        # 张量增加一维对齐原算法tensor维度
        pic_tensor = pic_tensor[np.newaxis, :]

        # label预处理
        # 来自原算法默认设置
        # num_keypoints = 9
        # 来自原算法默认设置
        # max_num_gt = 50
        # 来自原算法默认设置
        # num_labels = 2 * num_keypoints + 3  # +2 for ground-truth of width/height , +1 for class label
        # label = torch.zeros(max_num_gt * num_labels)
        # tmp = torch.from_numpy(read_truths_args(labelfile))
        # tmp = tmp.view(-1)
        # tsz = tmp.numel()
        # if tsz > max_num_gt * num_labels:
        #     label = tmp[0:max_num_gt * num_labels]
        # elif tsz > 0:
        #     label[0:tsz] = tmp
        # -------------------------单图end-----------------------------
        # target = label
        data = pic_tensor
        # Images
        img = data[0, :, :, :]
        img = img.numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))

        # Pass data to GPU
        data = data.cuda()
        # target = target.cuda()
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        with torch.no_grad():
            data = Variable(data)
        # Forward pass
        output = model(data).data
        all_boxes = get_region_boxes(output, num_classes, num_keypoints)
        corners2D_pr = np.array(np.reshape(all_boxes[:18], [9, 2]), dtype='float32')
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height
        preds_corners2D.append(corners2D_pr)
        # 可视化代码
        if visualize:
            # Visualize
            plt.xlim((0, im_width))
            plt.ylim((0, im_height))
            plt.imshow(scipy.misc.imresize(img, (im_height, im_width)))
            # Projections
            for edge in edges_corners:
                plt.plot(corners2D_pr[edge, 0], corners2D_pr[edge, 1], color='b', linewidth=3.0)
            plt.gca().invert_yaxis()
            plt.show()


datacfg = 'cfg/duck.data'
modelcfg = 'cfg/yolo-pose.cfg'
weightfile = 'backup/duck/model_backup.weights'

valid(datacfg, modelcfg, weightfile)

