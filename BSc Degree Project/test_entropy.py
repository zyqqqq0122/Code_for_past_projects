# coding:utf8
import models
from sets import *
import torch
from tqdm import tqdm
import numpy as np
import time
import os


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


check_ing_path = '/userhome/ZYQ/net_HGG/check0318drop/pth_1'
checkname = '0_4444_0_1e-05_12_0319_16:35:27.pth'

val_dice = []
val_std = []
val_label = []

if 1 > 0:
    if 1 > 0:
        if 1 > 0:

            model = getattr(models, 'U_Net')()
            model.eval()
            model.load_state_dict(torch.load(os.path.join(check_ing_path, checkname)))
            model.eval()
            ###### apply dropout or not
            model.apply(apply_dropout)
            ######
            model.cuda()

            testpath = '/userhome/ZYQ/NPY_HGG/HGG5'
            subjectives = os.listdir(testpath)
            WT_dice = []

            for s in subjectives:

                spath = os.path.join(testpath, s)
                data = np.load(spath)
                print(data.shape)  # (5, 240, 240, 155)
                tru = data[4, :, :, :]

                prob = np.zeros((5, 240, 240, 155))
                flag = np.zeros((5, 240, 240, 155))
                entropy = np.zeros((5, 240, 240, 155))

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

                for mm in range(240):
                    for nn in range(240):
                        for kk in range(155):
                            if flag[0, mm, nn, kk] == 0:
                                prob[0, mm, nn, kk] = 0.9996
                                prob[1, mm, nn, kk] = 0.0001
                                prob[2, mm, nn, kk] = 0.0001
                                prob[3, mm, nn, kk] = 0.0001
                                prob[4, mm, nn, kk] = 0.0001

                flag[flag == 0] = 1.0
                prob = prob / flag
                p0 = prob[0, :, :, :]
                p1 = prob[1, :, :, :]
                p2 = prob[2, :, :, :]
                p3 = prob[3, :, :, :]
                p4 = prob[4, :, :, :]
                entropy[:, :, :] = -(p0 * np.log2(p0) + p1 * np.log2(p1) + p2 * np.log2(p2) + p3 * np.log2(
                    p3) + p4 * np.log2(p4))
                label = np.argmax(prob.astype(float), axis=0)
                pre = label
                # print(label.shape)
                np.save('/save/path/pre_{}'.format(s), pre)
                np.save('/save/path/tru_{}'.format(s), tru)

                preg = pre
                trug = tru
                pre = np.zeros(preg.shape)
                tru = np.zeros(trug.shape)
                pre[preg > 0] = 1
                tru[trug > 0] = 1
                a1 = np.sum(pre == 1)
                a2 = np.sum(tru == 1)
                a3 = np.sum(np.multiply(pre, tru) == 1)
                # print(a1, a2, a3)
                WT_Dice = 0
                if a1 + a2 > 0:
                    WT_Dice = (2.0 * a3) / (a1 + a2)
                WT_dice.append(WT_Dice)
                print(WT_Dice)
                val_label.append(label)

                ### mean
                mean_WT_dice = np.mean(WT_dice)
                print('mean  ', 'WT:', mean_WT_dice)
                val_dice.append(mean_WT_dice)

                ### std
                std_WT_dice = np.std(WT_dice)
                print('std  ', 'WT:', std_WT_dice)
                val_std.append(std_WT_dice)

print('over!')
