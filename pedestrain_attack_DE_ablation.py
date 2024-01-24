import os

import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import random
import math
from detect import yolov3_inf
from DE import img_camera_sticker_pattern, initiation, mutation, crossover, selection, rate
import pandas as pd

aa = 0


count = np.zeros((7, 6))
query = np.zeros((7, 6))
Rm = 0.5
Rc = 0.6

# 4-11, 0-6
for patch_num in range(7, 8):
    for side_length in range(3, 4):

        dir = r'C:\Users\a\PycharmProjects\pythonProject\yolo_v3_train\yolo_v3\data\custom\images'
        pic = os.listdir(dir)
        POP_num = 100  # 种群大小
        size = 3 * patch_num  # 每个个体有多少基因，7个patch的位置与角度：x, y, alpha
        Step = 10  # 迭代次数

        pic_number = 0
        for pic_id in range(0, 1011):
            print('pic_id = ', pic_id)
            pic_dir = dir + '/' + pic[pic_id]
            # print(pic_dir)
            result = yolov3_inf(pic_dir)
            # print(result[0].shape)

            shape1, shape2 = result[0].shape



            if result[0].shape != (1, 5):
                continue

            pic_number = pic_number + 1
            print('pic_id, pic_number', pic_id, pic_number)


            x1, y1, x2, y2 = int(result[0][0][0]), int(result[0][0][1]), int(result[0][0][2]), int(result[0][0][3])
            POP = initiation(POP_num, size, x1, y1, x2, y2)
            POP_conf = np.ones((1, POP_num))
            # print('POP = ', POP)
            # print('POP_conf = ', POP_conf)

            # 获取初始父代置信度
            tag_break = 0
            for individual in range(0, POP_num):

                query[patch_num - 4][side_length] = query[patch_num - 4][side_length] + 1

                img = cv2.imread(pic_dir)
                path_adv = 'adv.jpg'

                print('父代初始化 pic_id, pic_number, individual = ', pic_id, pic_number, individual)

                img_camera_sticker_pattern(img, path_adv, rate(side_length), POP[individual], x1, x2) #指定x1，x2表示宽度相减，车辆攻击要变为y1，y2
                result_adv = yolov3_inf(path_adv)
                # print('result_adv[0].shape = ', result_adv[0].shape)
                if result_adv[0].shape == (0, 5):
                    # adv_save = 'path_adv/car_attack_visible/yolov3/' + pic[pic_id]
                    # img_save = cv2.imread(path_adv)
                    # cv2.imwrite(adv_save, img_save)
                    count[patch_num - 4][side_length] = count[patch_num - 4][side_length] + 1
                    tag_break = 1
                    break
                else:
                    POP_conf[0][individual] = result_adv[0][0][4]

                # print(result_adv[0][0][4])

            # print('POP = ', POP)
            # print('POP_conf = ', POP_conf)

            if tag_break == 1:
                continue

            POP_f = initiation(POP_num, size, x1, y1, x2, y2)
            POP_conf_f = np.ones((1, POP_num))

            for i in range(POP_num):
                for j in range(size):
                    POP_f[i][j] = POP[i][j]
            for i in range(POP_num):
                POP_conf_f[0][i] = POP_conf[0][i]

            # print('POP_f = ', POP_f)
            # print('POP_conf_f = ', POP_conf_f)

            # 开始DE攻击
            for step in range(Step):

                # print('POP = ', POP)
                POP = mutation(POP, Rm, x1, y1, x2, y2)
                # print('POP = ', POP)
                POP = crossover(POP, POP_f, Rc)
                # print('POP = ', POP)

                tag_break = 0

                for individual in range(0, POP_num):

                    query[patch_num - 4][side_length] = query[patch_num - 4][side_length] + 1

                    img = cv2.imread(pic_dir)
                    path_adv = 'adv.jpg'

                    # print('individual = ', individual)

                    img_camera_sticker_pattern(img, path_adv, rate(side_length), POP[individual], y1, y2)
                    result_adv = yolov3_inf(path_adv)
                    print('result_adv.shape = ', result_adv[0].shape)
                    if result_adv[0].shape == (0, 5):
                        # adv_save = 'path_adv/car_attack/yolov3/' + pic[pic_id]
                        # img_save = cv2.imread(path_adv)
                        # cv2.imwrite(adv_save, img_save)
                        count[patch_num - 4][side_length] = count[patch_num - 4][side_length] + 1
                        tag_break = 1
                        break
                    # print(result_adv[0][4])
                    POP_conf[0][individual] = result_adv[0][0][4]

                    print('patch_num, side_length, pic_id, pic_number, step, individual = ', patch_num, side_length, pic_id, pic_number, step, individual)


                    # img_show = plt.imread('result.jpg')
                    # plt.imshow(img_show)
                    # plt.show()

                if tag_break == 1:
                    break

                # print('POP = ', POP)
                # print('POP_f = ', POP_f)
                # print('POP_conf = ', POP_conf)
                # print('POP_conf_f = ', POP_conf_f)
                POP = selection(POP, POP_f, POP_conf, POP_conf_f)
                # print('POP = ', POP)


                print('count = ', count)
                print('query = ', query)





print('count = ', count)
print('query = ', query)

print('count/633 = ', count/633)
print('query/633 = ', query/633)



