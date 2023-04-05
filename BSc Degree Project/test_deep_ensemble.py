# coding:utf8
import models
from sets import *
import torch
from tqdm import tqdm
import numpy as np
import time
import os
from scipy.spatial.distance import directed_hausdorff


# ***************************************************
# Dropout: off
# Sample: T=20
# Model: U-Net
# ***************************************************


def dice_cal(pre, tru):
    WT_Dice = 0
    preg = pre
    trug = tru
    pre = np.zeros(preg.shape)
    tru = np.zeros(trug.shape)
    pre[preg > 0] = 1
    tru[trug > 0] = 1
    a1 = np.sum(pre == 1)
    a2 = np.sum(tru == 1)
    a3 = np.sum(np.multiply(pre, tru) == 1)
    if a1 + a2 > 0:
        WT_Dice = (2.0 * a3) / (a1 + a2)

    return WT_Dice


def iou_cal(pre, tru):
    intersection = np.logical_and(pre, tru)
    union = np.logical_or(pre, tru)

    return np.sum(intersection) / np.sum(union)


def f_cal(pre, tru):
    preg = pre
    trug = tru
    pre = np.zeros(preg.shape)
    tru = np.zeros(trug.shape)
    pre[preg > 0] = 1
    tru[trug > 0] = 1
    TP = TN = FP = FN = 1
    m = 0.000001

    TP = np.sum(np.multiply(preg, trug) == 1)
    TN = np.sum(np.multiply(pre, tru) == 1)
    FP = np.sum(np.multiply(preg, tru) == 1)
    FN = np.sum(np.multiply(pre, trug) == 1)

    sensitivity = TP / (TP + FN + m)
    specificity = TN / (FP + TN + m)
    precision = TP / (TP + FP + m)
    recall = TP / (TP + FN + m)

    f1_score = 2 * ((precision * recall) / (precision + recall + m))
    f2_score = (1 + 4) * ((precision * recall) / (4 * precision + recall + m))

    return f2_score, recall


if 1 > 0:
    if 1 > 0:
        if 1 > 0:
            list_path = '/model/list/path/'
            models = os.listdir(list_path)
            T = 20

            testpath = '/userhome/ZYQ/NPY_HGG/HGG5'
            subjectives = os.listdir(testpath)
            WT_dice = []
            WT_iou = []
            WT_f2 = []
            WT_recall = []
            WT_haus = []

            for s in subjectives:
                label_cal = np.zeros((T, 240, 240, 155))
                prob_cal = np.zeros((T, 240, 240, 155))
                prob_pre = np.zeros((T, 5, 240, 240, 155))
                spath = os.path.join(testpath, s)
                data = np.load(spath)
                tru = data[4, :, :, :]

                for kk in range(T):
                    model_name = models[kk]
                    model = getattr(models, 'U_Net')()
                    model.eval()
                    model.load_state_dict(torch.load(os.path.join(list_path, model_name)))
                    model.eval()
                    model.cuda()

                    prob = np.zeros((5, 240, 240, 155))
                    flag = np.zeros((5, 240, 240, 155))

                    g = 0
                    s0 = 48
                    s1 = 48
                    ss = 192
                    sss = 192

                    fast_stride = 20
                    for iii in range(50):
                        if iii * fast_stride < data.shape[3]:
                            if 10 * (iii + 1) <= data.shape[3]:
                                vector = data[0:4, :, :, fast_stride * iii:fast_stride * (iii + 1)].astype(float)
                                z_start = fast_stride * iii
                                z_end = fast_stride * (iii + 1)
                            else:
                                vector = data[0:4, :, :, fast_stride * iii:data.shape[3] - fast_stride * iii].astype(
                                    float)
                                z_start = fast_stride * iii
                                z_end = data.shape[3] - fast_stride * iii
                                print('vector shape: ', vector.shape)
                                print('z_start: ', z_start)
                                print('z_end: ', z_end)

                            # print('vector shape: ', vector.shape) # (4, 240, 240, 20)
                            vector = vector.transpose(3, 0, 1, 2)  # (20, 4, 240, 240)

                            for i in range(50):
                                for ii in range(50):
                                    if g + s0 * i + ss < data.shape[1] - g:
                                        if g + s0 * ii + ss < data.shape[2] - g:
                                            img_out = vector[:, :, g + s0 * i:g + s0 * i + ss,
                                                      g + s1 * ii:g + s1 * ii + sss]
                                            img = torch.from_numpy(img_out).float()
                                            with torch.no_grad():
                                                input = torch.autograd.Variable(img)
                                            if True: input = input.cuda()

                                            score = model(input)
                                            score = torch.nn.Softmax(dim=1)(score).squeeze().detach().cpu().numpy()

                                            prob[:, g + s0 * i:g + s0 * i + ss, g + s1 * ii:g + s1 * ii + sss,
                                            z_start:z_end] = prob[:, g + s0 * i:g + s0 * i + ss,
                                                             g + s1 * ii:g + s1 * ii + sss,
                                                             z_start:z_end] + score.transpose(1, 2, 3, 0)
                                            flag[:, g + s0 * i:g + s0 * i + ss, g + s1 * ii:g + s1 * ii + sss,
                                            z_start:z_end] = flag[:, g + s0 * i:g + s0 * i + ss,
                                                             g + s1 * ii:g + s1 * ii + sss, z_start:z_end] + 1

                    flag[flag == 0] = 1.0
                    prob = prob / flag
                    prob_pre[kk] = prob
                    prob_max = np.max(prob, axis=0)
                    prob_cal[kk] = prob_max
                    label_cal[kk] = np.argmax(prob.astype(float), axis=0)

                prob_pre = np.mean(prob_pre, axis=0)
                label = np.argmax(prob_pre.astype(float), axis=0)
                pre = label

                prob_var = np.var(prob_cal, axis=0)
                prob_mean = np.mean(prob_cal, axis=0)

                t = np.random.randint(0, T)
                label_st = label_cal[t]

                diff = np.zeros((T, 240, 240, 155))
                for kk in range(T):
                    diff[kk] = label_st != label_cal[kk]

                count = np.sum(diff, axis=0)
                np.save('/save/path/pairwise_{}'.format(s), count)
                np.save('/save/path/mean_{}'.format(s), prob_mean)
                np.save('/save/path/var_{}'.format(s), prob_var)

                dice = dice_cal(pre, tru)
                iou = iou_cal(pre, tru)
                f2, recall = f_cal(pre, tru)
                haus = directed_hausdorff(pre, tru)[0]

                WT_dice.append(dice)
                WT_iou.append(iou)
                WT_f2.append(f2)
                WT_recall.append(recall)
                WT_haus.append(haus)

            np.save('/save/path/dice', np.array(WT_dice))
            np.save('/save/path/iou', np.array(WT_iou))
            np.save('/save/path/f2', np.array(WT_f2))
            np.save('/save/path/recall', np.array(WT_recall))
            np.save('/save/path/haus', np.array(WT_haus))
print('over!')
