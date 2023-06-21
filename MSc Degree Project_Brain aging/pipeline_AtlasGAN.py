#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import os
import random

import SimpleITK as sitk
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as KL
import voxelmorph as vxm
from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed
from voxelmorph.tf.layers import SpatialTransformer, VecInt, RescaleTransform

# my import
import visualize_tools as vt
from src.networks import Generator
from src.networks import conv_block

# ----------------------------------------------------------------------------
# Set up CLI arguments:
# TODO: replace with a config json. CLI is unmanageably large now.
# TODO: add option for type of discriminator augmentation.

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dataset', type=str, default='OASIS3')
parser.add_argument('--name', type=str, default='experiment_name')
parser.add_argument('--d_train_steps', type=int, default=1)
parser.add_argument('--g_train_steps', type=int, default=1)

# TTUR for training GAN, already set the default values in consistent with appendices
parser.add_argument('--lr_g', type=float, default=1e-4)
parser.add_argument('--lr_d', type=float, default=3e-4)
parser.add_argument('--beta1_g', type=float, default=0.0)
parser.add_argument('--beta2_g', type=float, default=0.9)
parser.add_argument('--beta1_d', type=float, default=0.0)
parser.add_argument('--beta2_d', type=float, default=0.9)

parser.add_argument(
    '--unconditional', dest='conditional', default=True, action='store_false',
)
parser.add_argument(
    '--nonorm_reg', dest='norm_reg', default=True, action='store_false',
)  # Not used in the paper.
parser.add_argument(
    '--oversample', dest='oversample', default=True, action='store_false',
)
parser.add_argument(
    '--d_snout', dest='d_snout', default=False, action='store_true',
)
parser.add_argument(
    '--noclip', dest='clip_bckgnd', default=True, action='store_false',
)  # should be True, updated
parser.add_argument('--reg_loss', type=str,
                    default='NCC')  # One of {'NCC', 'NonSquareNCC'}. Not used NonSquareNCC in paper
parser.add_argument('--losswt_reg', type=float, default=1.0)
parser.add_argument('--losswt_gan', type=float, default=0.1)
parser.add_argument('--losswt_tv', type=float, default=0.00)  # Not used in the paper.
parser.add_argument('--losswt_gp', type=float,
                    default=1e-3)  # TODO: Gradient penalty for discriminator loss. Need to be adjusted according to dataset. Important!!!
parser.add_argument('--gen_config', type=str, default='ours')  # One of {'ours', 'voxelmorph'}.
parser.add_argument('--steps_per_epoch', type=int, default=1000)
parser.add_argument('--rng_seed', type=int, default=33)
parser.add_argument('--start_step', type=int,
                    default=0)  # Not used in paper. GAN training is active from the first iteration.
parser.add_argument('--resume_ckpt', type=int, default=0)  # checkopint
parser.add_argument('--g_ch', type=int, default=32)
parser.add_argument('--d_ch', type=int, default=64)
parser.add_argument('--init', type=str, default='default')  # One of {'default', 'orthogonal'}.
parser.add_argument('--lazy_reg', type=int, default=1)  # Not used in the paper.

# my arguments
parser.add_argument('--checkpoint_path', type=str,
                    default='/home/data/jrfu/data/trained_models/HC_only/training_checkpoints/gploss_1e_4_dataset_OASIS3_single_cohort_eps300_Gconfig_ours_normreg_True_lrg0.0001_lrd0.0003_cond_True_regloss_NCC_lbdgan_0.1_lbdreg_1.0_lbdtv_0.0_lbdgp_0.0001_dsnout_False_start_0_clip_True/')
parser.add_argument('--save_path', type=str, default='/home/data/test_0420/')

args = parser.parse_args()

# my CLI
checkpoint_path = args.checkpoint_path  # None
save_path = args.save_path  # None

# Get CLI information:
epochs = args.epochs
batch_size = args.batch_size
dataset = args.dataset
exp_name = args.name
lr_g = args.lr_g
lr_d = args.lr_d
beta1_g = args.beta1_g
beta2_g = args.beta2_g
beta1_d = args.beta1_d
beta2_d = args.beta2_d
conditional = args.conditional
reg_loss = args.reg_loss
norm_reg = args.norm_reg
oversample = args.oversample
atlas_model = args.gen_config
steps = args.steps_per_epoch
lambda_gan = args.losswt_gan
lambda_reg = args.losswt_reg
lambda_tv = args.losswt_tv
lambda_gp = args.losswt_gp
g_loss_wts = [lambda_gan, lambda_reg, lambda_tv]
start_step = args.start_step
rng_seed = args.rng_seed
resume_ckpt = args.resume_ckpt
d_snout = args.d_snout
clip_bckgnd = args.clip_bckgnd
g_ch = args.g_ch
d_ch = args.d_ch
init = args.init
lazy_reg = args.lazy_reg
# ----------------------------------------------------------------------------
# Set RNG seeds

seed(rng_seed)
set_random_seed(rng_seed)
random.seed(rng_seed)
# ----------------------------------------------------------------------------
# Initialize data generators

# Change these if working with new dataset:
if dataset == 'dHCP':
    fpath = './data/dHCP2/npz_files/T2/train/*.npz'
    avg_path = (
        './data/dHCP2/npz_files/T2/linearaverage_100T2_train.npz'
    )
    n_condns = 1
elif dataset == 'pHD':
    fpath = './data/predict-hd/npz_files/train_npz/*.npz'
    avg_path = './data/predict-hd/linearaverageof100.npz'
    n_condns = 3
elif dataset == 'OASIS3':
    main_path = '/home/data/jrfu/data/OASIS3/'  # /media/fjr/My Passport/data/OASIS3/ or /data/OASIS3/ or /proj/OASIS3_atlasGAN/ or /media/fjr/My Passport/data/OASIS3/
    fpath = main_path + 'all_npz/'
    avg_path = main_path + 'linearaverageof100.npz'
    n_condns = 1  # single cohort: 1, mix: 3
else:
    raise ValueError('dataset expected to be dHCP, pHD or OASIS3')


# extract training data from FS_scans_metadata_with_split.csv
def read_csv(set="train"):
    path = f'{main_path}FS_scans_metadata_with_split.csv'
    df = pd.read_csv(path)
    ids_set = df["ID"][df["Partition"] == set]
    ids_set = [fpath + i.split("_")[0] + "_" + i.split("_")[2] + '.npz' for i in list(ids_set)]
    return ids_set


# img_paths = read_csv("test")

avg_img = np.load(avg_path)['vol']  # TODO: make generic fname in npz

vol_shape = avg_img.shape  # calculate [208, 176, 160] for OASIS3 dataset

avg_batch = np.repeat(
    avg_img[np.newaxis, ...], batch_size, axis=0,
)[..., np.newaxis]
# ----------------------------------------------------------------------------
# Initialize generator (registration included) networks

generator = Generator(
    ch=g_ch,
    atlas_model=atlas_model,
    conditional=conditional,
    normreg=norm_reg,
    clip_bckgnd=clip_bckgnd,
    input_resolution=[*vol_shape, 1],
    initialization=init,
    n_condns=n_condns,
)

# set up checkpoints
checkpoint = tf.train.Checkpoint(
    generator=generator,
)

# restore checkpoint from the latest trained model
if checkpoint_path:
    checkpoint.restore(
        tf.train.latest_checkpoint(checkpoint_path)
    )
else:
    raise ValueError('Testing phase, please provide checkpoint path!')


# ----------------------------------------------------------------------------
# Initialize registration network

# define a registration model
def Registration(
        ch=32,
        normreg=False,
        input_resolution=[160, 192, 160, 1],
):
    image_inputs = tf.keras.layers.Input(shape=input_resolution)
    new_atlas = tf.keras.layers.Input(shape=input_resolution)

    init = None
    vel_init = tf.keras.initializers.RandomNormal(
        mean=0.0,
        stddev=1e-5,
    )

    # Registration network. Taken from vxm:
    # Encoder:
    inp = KL.concatenate([image_inputs, new_atlas])
    d1 = conv_block(inp, ch, stride=2, instancen=normreg, init=init)
    d2 = conv_block(d1, ch, stride=2, instancen=normreg, init=init)
    d3 = conv_block(d2, ch, stride=2, instancen=normreg, init=init)
    d4 = conv_block(d3, ch, stride=2, instancen=normreg, init=init)

    # Bottleneck:
    dres = conv_block(d4, ch, instancen=normreg, init=init)

    # Decoder:
    d5 = conv_block(dres, ch, mode='up', instancen=normreg, init=init)
    d5 = KL.concatenate([d5, d3])

    d6 = conv_block(d5, ch, mode='up', instancen=normreg, init=init)
    d6 = KL.concatenate([d6, d2])

    d7 = conv_block(d6, ch, mode='up', instancen=normreg, init=init)
    d7 = KL.concatenate([d7, d1])

    d7 = conv_block(
        d7, ch, mode='const', instancen=normreg, init=init,
    )
    d7 = conv_block(
        d7, ch, mode='const', instancen=normreg, init=init,
    )
    d7 = conv_block(d7, ch // 2, mode='const', activation=False, init=init)

    # Get velocity field:
    d7 = tf.pad(d7, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")

    vel = KL.Conv3D(
        filters=3,
        kernel_size=3,
        padding='valid',
        use_bias=True,
        kernel_initializer=vel_init,
        name='vel_field',
    )(d7)

    # Get diffeomorphic displacement field:
    diff_field = VecInt(method='ss', int_steps=5, name='def_field')(vel)

    # Get moving average of deformations:
    # diff_field_ms = MeanStream(name='mean_stream', cap=100)(diff_field)

    # compute regularizers on diff_field_half for efficiency:
    # diff_field_half = 1.0 * diff_field
    vel_field = RescaleTransform(2.0, name='flowup_vel_field')(vel)
    diff_field = RescaleTransform(2.0, name='flowup')(diff_field)
    moved_atlas = SpatialTransformer()([new_atlas, diff_field])
    ops = [moved_atlas, diff_field, vel_field]

    return tf.keras.Model(
        inputs=[image_inputs, new_atlas],
        outputs=ops,
    )


# Initialize registration model
registration_model = Registration(
    ch=g_ch,
    normreg=norm_reg,
    input_resolution=[*vol_shape, 1],
)


# ----------------------------------------------------------------------------
# Set weights for registration network layers and save registration model

# Observe model layers
def observe_model(generator):
    count = 0
    num_trainable_weights = 0
    num_non_trainable_weights = 0
    for layer_id, layer in enumerate(generator.layers):
        if len(layer.trainable_weights) > 0:
            print(f'{layer_id}: {count}th trainable layer, name = {layer.name}, '
                  f'trainable_weights = {len(layer.trainable_weights)}, non_trainable_weights = {len(layer.non_trainable_weights)}')
            count += 1
        if len(layer.trainable_weights) > 0:
            num_trainable_weights += len(layer.trainable_weights)
        if len(layer.non_trainable_weights) > 0:
            num_non_trainable_weights += len(layer.non_trainable_weights)

    print(f'trainable weights total = {num_trainable_weights}, non trainable = {num_non_trainable_weights}')
    # trainable weights total = 101, non trainable = 16


# observe weights, reference page: https://www.tensorflow.org/guide/checkpoint
# reader = tf.train.load_checkpoint(tf.train.latest_checkpoint(checkpoint_path))
# shape_from_key = reader.get_variable_to_shape_map()
# dtype_from_key = reader.get_variable_to_dtype_map()

# for key in sorted(shape_from_key.keys()):
#     if key.startswith("generator/"):
#         print(f'{key}')

# so, there are 58 trainable layers and 58 saved keys. Should be able to load weight one layer by one layer

# observe_model(generator)
# observe_model(registration_model)
# weights_list = generator.get_weights() # 117 long

# construct weight layer names
def get_layers_name_with_weights(generator):
    weights_layers = []
    for layer_id, layer in enumerate(generator.layers):
        if len(layer.trainable_weights) > 0 or len(layer.non_trainable_weights) > 0:
            # print(f'{layer_id}th layer, name = {layer.name}, ' f'trainable_weights = {len(
            # layer.trainable_weights)}, non_trainable_weights = {len(layer.non_trainable_weights)}') repeat_times =
            # len(layer.trainable_weights) + len(layer.non_trainable_weights) for i in range(repeat_times):
            # weights_layers.append(layer.name)
            weights_layers.append(layer.name)
    return weights_layers


weights_layers_generator = get_layers_name_with_weights(generator)
weights_layers_registration = get_layers_name_with_weights(registration_model)
# load weight layer by layer, references: https://www.gcptutorials.com/post/how-to-get-weights-of-layers-in-tensorflow
# https://stackoverflow.com/questions/43702323/how-to-load-only-specific-weights-on-keras
start_generator = weights_layers_generator.index('conv3d_12')
for i, layer in enumerate(weights_layers_registration):
    generator_layer = weights_layers_generator[start_generator + i]
    print(f'Loading weights for layer {layer} from generator layer {generator_layer}')
    registration_model.get_layer(layer).set_weights(generator.get_layer(generator_layer).get_weights())

print("loading end")

# save registration model
checkpoint_dir = '/home/data/test_0420/models_and_data/registration_model/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    registration_model=registration_model,
)
checkpoint.save(file_prefix=checkpoint_prefix)


# ----------------------------------------------------------------------------
# Load the saved registration model

# set up checkpoint
checkpoint = tf.train.Checkpoint(
    registration_model=registration_model,
)

# restore checkpoint from the saved model
checkpoint_path = '/home/data/test_0420/models_and_data/registration_model/'
if checkpoint_path:
    checkpoint.restore(
        tf.train.latest_checkpoint(checkpoint_path)
    )
else:
    raise ValueError('Testing phase, please provide checkpoint path!')


# ----------------------------------------------------------------------------
# Implement registration given:
# two images input: [fixed_image, moving_image]
# output: [moved_image, diff_field, vel_field], and save the SVF (vel_field)

def extract_and_save(fixed_image, moving_image, save_path, save_name, save_moved_nii=False, save_vel_nii=False):
    os.makedirs(save_path, exist_ok=True)

    fixed_image = tf.convert_to_tensor(fixed_image, dtype=tf.float32)
    moving_image = tf.convert_to_tensor(moving_image, dtype=tf.float32)

    [moved_atlas, diff_field, vel_field] = registration_model([fixed_image, moving_image])

    print(f'Moved image shape = {moved_atlas.numpy().squeeze().shape}, save as {save_path}{save_name}_vel.nii.gz')

    """ np.savez_compressed(
        save_path + save_name + '.npz',
        moved = moved_atlas.numpy().squeeze(),
        diff = diff_field.numpy().squeeze(),
        vel  = vel_field.numpy().squeeze()
    ) """

    if save_moved_nii is True:
        atlasmax = tf.reduce_max(moved_atlas).numpy()  # find the max value
        print("atlasmax = {}".format(atlasmax))
        template = tf.nn.relu(moved_atlas.numpy().squeeze()).numpy() / atlasmax  # with normalization
        # template = sharp_atlases.numpy().squeeze() # without normalization
        # use PSR transform as default affine
        # affine = np.array([[0, 0, -1, 0],  # nopep8
        #                    [1, 0, 0, 0],  # nopep8
        #                    [0, -1, 0, 0],  # nopep8
        #                    [0, 0, 0, 1]], dtype=float)  # nopep8
        # pcrs = np.append(np.array(template.shape[:3]) / 2, 1)
        # affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        # vxm.py.utils.save_volfile(template, save_path + save_name + '_moved.nii.gz', affine)
        vxm.py.utils.save_volfile(template, save_path + save_name + '_moved.nii.gz')
        vt.correct_vox2ras_matrix(save_path + save_name + '_moved.nii.gz')

    if save_vel_nii is True:
        # atlasmax = tf.reduce_max(vel_field).numpy() # find the max value
        # print("atlasmax = {}".format(atlasmax))
        # template = tf.nn.relu(vel_field.numpy().squeeze()).numpy()/ atlasmax  # with normalization
        # template = sharp_atlases.numpy().squeeze() # without normalization
        # use PSR transform as default affine
        # affine = np.array([[0, 0, -1, 0],  # nopep8
        #                    [1, 0, 0, 0],  # nopep8
        #                    [0, -1, 0, 0],  # nopep8
        #                    [0, 0, 0, 1]], dtype=float)  # nopep8
        # pcrs = np.append(np.array(template.shape[:3]) / 2, 1)
        # affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        # vxm.py.utils.save_volfile(template, save_path + save_name + '_vel.nii.gz', affine)
        vxm.py.utils.save_volfile(vel_field.numpy().squeeze(), save_path + save_name + '_vel.nii.gz')
        vt.correct_vox2ras_matrix(save_path + save_name + '_vel.nii.gz')


# construct subject list: [subejct, src_subject_uid, src_age, dst_subject_uid, dst_age]
def construct_subject_list(csv_path):
    df = pd.read_csv(csv_path)
    subject_list = df['subject'].value_counts().index
    test_subject_list = []
    for s in subject_list:
        src_age = df['age_rounded'][df['subject'] == s][df['tag'] == 'src'].tolist()[0]
        src_subject = df['uid'][df['subject'] == s][df['tag'] == 'src'].tolist()[0]

        dst_age = df['age_rounded'][df['subject'] == s][df['tag'] == 'dst']
        for d in dst_age:
            dst_subject = df['uid'][df['subject'] == s][df['age_rounded'] == d].tolist()[0]
            test_subject_list.append([s, src_subject, src_age, dst_subject, d])

    return test_subject_list


# implement template-to-template (t2t) or template-to-subject (t2s) registration and mask generation
def registration_imp(subject_list, disease_code, path_prefix_m, path_prefix_f, save_path, nii_path, mask_path,
                     flag='t2t'):
    if flag == 't2t':
        print('Implementing template-to-template registration...')
        for i in subject_list:
            moving_image = os.path.join(path_prefix_m, f'age_{i[2]}disease_{disease_code}.nii.gz')
            fixed_image = os.path.join(path_prefix_f, f'age_{i[-1]}disease_{disease_code}.nii.gz')
            save_name = f'T{i[2]}toT{i[-1]}_{disease_code}'
            if not os.path.exists(save_path + save_name + '_vel.nii.gz'):
                extract_and_save(np.transpose(vt.load_nii(fixed_image), (2, 1, 0))[np.newaxis, ..., np.newaxis],
                                 np.transpose(vt.load_nii(moving_image), (2, 1, 0))[np.newaxis, ..., np.newaxis],
                                 save_path, save_name,
                                 save_moved_nii=False, save_vel_nii=True)
            else:
                print('Pass :)')

    elif flag == 't2s':
        print('Implementing template-to-subject registration and mask generation...')
        for i in subject_list:
            moving_image = os.path.join(path_prefix_m, f'age_{i[2]}disease_{disease_code}.nii.gz')
            fixed_image = os.path.join(path_prefix_f, f'{i[1]}.npz')
            nii_name = os.path.join(nii_path, f'{i[1]}.nii.gz')

            if not os.path.exists(nii_name):
                vt.npz2nii(fixed_image, nii_path, f'{i[1]}.nii.gz')
                vt.correct_vox2ras_matrix(nii_name)

            fixed_image = vt.load_nii(nii_name)
            moving_image = vt.load_nii(moving_image)
            save_name = f'T{i[2]}to{i[0]}_{disease_code}'
            if not os.path.exists(save_path + save_name + '_vel.nii.gz'):
                extract_and_save(np.transpose(fixed_image, (2, 1, 0))[np.newaxis, ..., np.newaxis],
                                 np.transpose(moving_image, (2, 1, 0))[np.newaxis, ..., np.newaxis],
                                 save_path, save_name,
                                 save_moved_nii=False, save_vel_nii=True)
            else:
                print('Pass :)')

            # mask generation
            mask = np.where(fixed_image + moving_image > 0.0, 1.0, 0.0)
            mask_name = f'{i[0]}_age_{i[2]}_{disease_code}_mask.nii.gz'
            if not os.path.exists(mask_path + mask_name):
                vt.np2nii(mask, mask_path, mask_name)
            else:
                print('Pass :)')
    else:
        print('Flag should be t2t or t2s.')


# save and read subject list
def save_list(list_to_save, save_name, save_path):
    file = open(f'{save_path}{save_name}.txt', 'w')
    for s in list_to_save:
        file.write(str(s))
        file.write('\n')
    file.close()


def read_list(list_name, list_path):
    file = open(f'{list_path}{list_name}.txt', 'r')
    X = file.readlines()
    for i in range(len(X)):
        X[i] = X[i].strip()
        X[i] = X[i].strip("[]")
        X[i] = X[i].split(",")
        X[i] = [X[i][j].strip(" '") for j in range(len(X[i]))]
    file.close()

    return X


disease_condn = 'HC'  # HC0/AD1
disease_code = 0 if disease_condn == 'HC' else 1
main_path = '/home/data/test_0420/'
csv_path = main_path + f'models_and_data/FS_scans_test_{disease_condn}.csv'
subject_list = construct_subject_list(csv_path)
save_path = main_path + 'models_and_data/'
save_list(subject_list, f'test_subject_list_{disease_condn}', save_path)

# template-to-template registration (t2t)
path_prefix_m = '/home/data/jrfu/data/trained_models/HC_only/plots/my_plot_1e-4/'
path_prefix_f = '/home/data/jrfu/data/trained_models/HC_only/plots/my_plot_1e-4/'
save_path = main_path + f'registration_t2t/{disease_condn}/'
registration_imp(subject_list=subject_list,
                 disease_code=disease_code,
                 path_prefix_m=path_prefix_m,
                 path_prefix_f=path_prefix_f,
                 save_path=save_path,
                 flag='t2t')

# template-to-subject registration (t2s), mask generation
path_prefix_m = '/home/data/jrfu/data/trained_models/HC_only/plots/my_plot_1e-4/'
path_prefix_f = '/home/data/jrfu/data/OASIS3/all_npz/'
save_path = main_path + f'registration_t2s/{disease_condn}/'
nii_path = '/home/data/jrfu/data/OASIS3/all_nii/'
mask_path = main_path + f'mask/{disease_condn}/'
registration_imp(subject_list=subject_list,
                 disease_code=disease_code,
                 path_prefix_m=path_prefix_m,
                 path_prefix_f=path_prefix_f,
                 save_path=save_path,
                 nii_path=nii_path,
                 mask_path=mask_path,
                 flag='t2s')


# ----------------------------------------------------------------------------
# Implement parallel transport using Ladder, given:
# ladder.sh, test_subject_list

# convert the output vel.mha to vel.nii.gz
def mha2nii(mha_file, nii_file):
    img = sitk.ReadImage(mha_file)
    sitk.WriteImage(img, nii_file)


disease_condn = 'HC'  # HC0/AD1
main_path = '/home/data/test_0420/'
mha_path = main_path + f'registration_s2s/mha/{disease_condn}/'
nii_path = main_path + f'registration_s2s/nii/{disease_condn}/'
files = os.listdir(mha_path)
for f in files:
    mha_file = os.path.join(mha_path, f)
    mha2nii(mha_file, nii_path + f[: -4] + '.nii.gz')


# ----------------------------------------------------------------------------
# Implement transformation with parallel transported SVF and calculate the difference
def transform_imp(subject_list, disease_code, vel_path, subject_path, nii_path, pre_path, diff_path, diff_cal=False):
    for i in subject_list:
        vel = os.path.join(vel_path, f'{i[0]}_{i[2]}to{i[4]}_{disease_code}_vel.nii.gz')
        vel = vt.load_nii(vel)
        vel = np.transpose(vel, (3, 2, 1, 0, 4))
        vel = tf.convert_to_tensor(vel, dtype=tf.float32)

        ref = os.path.join(nii_path, f'{i[1]}.nii.gz')
        ref = vt.load_nii(ref)
        ref = np.transpose(ref, (2, 1, 0))[np.newaxis, ..., np.newaxis]
        ref = tf.convert_to_tensor(ref, dtype=tf.float32)

        diff_field = VecInt(method='ss', int_steps=5, name='def_field')(vel)
        pre = SpatialTransformer()([ref, diff_field])

        pre_max = tf.reduce_max(pre).numpy()
        subject_predicted = tf.nn.relu(pre.numpy().squeeze()).numpy() / pre_max
        save_name = os.path.join(pre_path, f'pre_{i[0]}_{i[2]}to{i[4]}_{disease_code}.nii.gz')
        vxm.py.utils.save_volfile(subject_predicted, save_name)
        vt.correct_vox2ras_matrix(save_name)

        if diff_cal:
            gt_subject = os.path.join(subject_path, f'{i[3]}.npz')
            if not os.path.exists(os.path.join(nii_path, f'{i[3]}.nii.gz')):
                vt.npz2nii(gt_subject, nii_path, f'{i[3]}.nii.gz')
                vt.correct_vox2ras_matrix(os.path.join(nii_path, f'{i[3]}.nii.gz'))
            gt_subject = vt.load_nii(os.path.join(nii_path, f'{i[3]}.nii.gz'))
            subject_predicted = vt.load_nii(save_name)
            diff_subject = subject_predicted - gt_subject
            vt.np2nii(diff_subject, diff_path, f'diff_{i[0]}_{i[2]}to{i[4]}_{disease_code}.nii.gz')


disease_condn = 'HC'  # HC0/AD1
disease_code = 0 if disease_condn == 'HC' else 1
main_path = '/home/data/test_0420/'
vel_path = main_path + f'registration_s2s/nii/{disease_condn}/'
subject_path = '/home/data/jrfu/data/OASIS3/all_npz/'
nii_path = '/home/data/jrfu/data/OASIS3/all_nii/'
pre_path = main_path + f'predictions/{disease_condn}/'
diff_path = main_path + f'difference/{disease_condn}/'
transform_imp(subject_list=subject_list,
              disease_code=disease_code,
              vel_path=vel_path,
              subject_path=subject_path,
              nii_path=nii_path,
              pre_path=pre_path,
              diff_path=diff_path,
              diff_cal=True)
