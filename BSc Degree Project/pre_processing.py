#!/usr/bin/env python
# coding:utf8
import numpy as np
import os
from PIL import Image
import nibabel as nib
import imageio


def nii_to_image(niifilepath, imgfilepath):
    filenames = os.listdir(niifilepath)

    for f in filenames:
        img_path = os.path.join(niifilepath, f)
        img = nib.load(img_path, )
        img_fdata = img.get_fdata()

        fnamex = f.replace('.nii', '-x')
        img_f_pathx = os.path.join(imgfilepath, fnamex)
        if not os.path.exists(img_f_pathx):
            os.mkdir(img_f_pathx)

        fnamey = f.replace('.nii', '-y')
        img_f_pathy = os.path.join(imgfilepath, fnamey)
        if not os.path.exists(img_f_pathy):
            os.mkdir(img_f_pathy)

        fnamez = f.replace('.nii', '-z')
        img_f_pathz = os.path.join(imgfilepath, fnamez)
        if not os.path.exists(img_f_pathz):
            os.mkdir(img_f_pathz)

        (x, y, z) = img.shape
        for i in range(x):
            slice = img_fdata[i, :, :]
            imageio.imwrite(os.path.join(img_f_pathx, '{}.png'.format(i)), slice)

        for i in range(y):
            slice = img_fdata[:, i, :]
            imageio.imwrite(os.path.join(img_f_pathy, '{}.png'.format(i)), slice)

        for i in range(z):
            slice = img_fdata[:, :, i]
            imageio.imwrite(os.path.join(img_f_pathz, '{}.png'.format(i)), slice)


def read_png(pngfile):
    # obtain img size
    img = Image.open(os.path.join(pngfile, '0.png'))
    img = np.array(img)
    (m, n) = img.shape

    # read folder along z
    filenames = os.listdir(pngfile)
    i = 0
    img = np.zeros([155, m, n])

    # read png files
    for f in filenames:
        img_path = os.path.join(pngfile, f)
        img[i] = np.array(Image.open(img_path))
        i = i + 1

    return img


def save_npy_slice(filepath):
    subjects = os.listdir(filepath)

    for s in subjects:
        s_path = os.path.join(filepath, s)
        files = os.listdir(s_path)
        files.sort()
        (x, y, z) = read_png('path/to/one/file').shape
        img_file = np.zeros((5, x, y, z))
        i = 0
        for f in files:
            f_path = os.path.join(s_path, f)
            img_file[i] = read_png(f_path)
            i = i + 1
        for j in range(x):
            npy_img = np.zeros((5, y, z))
            for k in range(5):
                npy_img[k] = img_file[k, j, :, :]
                if k != 4:
                    npy_img[k] = (npy_img[k] - np.min(npy_img[k])) / (np.max(npy_img[k]) - np.min(npy_img[k]))

            np.save('/save/path/NPY/{}/npy_img{}.npy'.format(s, j), npy_img)


def slice_conc(filepath):
    subjects = os.listdir(filepath)

    for s in subjects:
        s_path = os.path.join(filepath, s)
        img_conc = np.zeros((5, 240, 240, 155))
        for j in range(155):
            img = np.load(os.path.join(s_path, 'npy_img{}.npy'.format(j)))
            img_conc[:, :, :, j] = img

        np.save('/save/path/NPY_conc/{}/npy_img.npy'.format(s), img_conc)


def read_nii(filepath):
    img = nib.load(filepath)
    return img.get_fdata()


def img_norm(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def nii_to_npy(filepath):
    subjects = os.listdir(filepath)
    for s in subjects:
        spath = os.path.join(filepath, s)
        nfiles = os.listdir(spath)
        for n in nfiles:
            npath = os.path.join(spath, n)
            if n == '{}_flair.nii.gz'.format(s):
                img1 = read_nii(npath)
                img1 = img_norm(img1)
                (x, y, z) = img1.shape
            if n == '{}_t1.nii.gz'.format(s):
                img2 = read_nii(npath)
                img2 = img_norm(img2)
                (x, y, z) = img2.shape
            if n == '{}_t1ce.nii.gz'.format(s):
                img3 = read_nii(npath)
                img3 = img_norm(img3)
                (x, y, z) = img3.shape
            if n == '{}_t2.nii.gz'.format(s):
                img4 = read_nii(npath)
                img4 = img_norm(img4)
                (x, y, z) = img4.shape
            if n == '{}_seg.nii.gz'.format(s):
                img5 = read_nii(npath)
                (x, y, z) = img5.shape

        npy_img = np.zeros((5, x, y, z))
        npy_img[0] = img1
        npy_img[1] = img2
        npy_img[2] = img3
        npy_img[3] = img4
        npy_img[4] = img5

        np.save('/save/path/{}.npy'.format(s), npy_img)