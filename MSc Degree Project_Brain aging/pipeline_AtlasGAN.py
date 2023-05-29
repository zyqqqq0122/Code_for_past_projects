#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import random
import argparse

from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed

from src.networks import Generator
import pandas as pd
import tensorflow.keras.layers as KL

# my import
import visualize_tools as vt
import voxelmorph as vxm
from src.networks import conv_block
from neurite.tf.layers import MeanStream
from voxelmorph.tf.layers import SpatialTransformer, VecInt, RescaleTransform

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
                    default='/home/data/models_and_data/gploss_1e_4_dataset_OASIS3_eps200_Gconfig_ours_normreg_True_lrg0.0001_lrd0.0003_cond_True_regloss_NCC_lbdgan_0.1_lbdreg_1.0_lbdtv_0.0_lbdgp_0.0001_dsnout_False_start_0_clip_True/')
parser.add_argument('--save_path', type=str, default='/home/data/test_0319/')

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
    # main_path = '/media/fjr/My Passport/data/OASIS3/' # /data/OASIS3/ or /proj/OASIS3_atlasGAN/ or /media/fjr/My Passport/data/OASIS3/
    main_path = '/home/data/models_and_data/OASIS3/'
    fpath = main_path + 'all_npz/'
    avg_path = main_path + 'linearaverageof100.npz'
    n_condns = 3
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
# Initialize networks

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

# ----------------------------------------------------------------------------
# Set up Checkpoints
checkpoint = tf.train.Checkpoint(
    generator=generator,
)

# restore checkpoint from the latest trained model:
if checkpoint_path:
    checkpoint.restore(
        tf.train.latest_checkpoint(checkpoint_path)
    )
else:
    raise ValueError('Testing phase, please provide checkpoint path!')

# observe model layers
def observe_model(generator):
    count = 0
    num_trainable_weights = 0
    num_non_trainable_weights = 0
    for layer_id, layer in enumerate(generator.layers):
        if len(layer.trainable_weights) > 0:
            print(f'{layer_id}: {count}th trainable layer, name = {layer.name}, '
                  f'trainable_weights = {len(layer.trainable_weights)}, non_trainable_weights = {len(layer.non_trainable_weights)}')
            count+=1
        if len(layer.trainable_weights)>0:
            num_trainable_weights += len(layer.trainable_weights)
        if len(layer.non_trainable_weights) > 0:
            num_non_trainable_weights += len(layer.non_trainable_weights)

    print(f'trainable weights total = {num_trainable_weights}, non trainable = {num_non_trainable_weights}')
    # trainable weights total = 101, non trainable = 16

# # observe weights, reference page: https://www.tensorflow.org/guide/checkpoint
# reader = tf.train.load_checkpoint(tf.train.latest_checkpoint(checkpoint_path))
# shape_from_key = reader.get_variable_to_shape_map()
# dtype_from_key = reader.get_variable_to_dtype_map()
#
# for key in sorted(shape_from_key.keys()):
#     if key.startswith("generator/"):
#         print(f'{key}')

# so, there are 58 trainable layers and 58 saved keys. Should be able to load weight one layer by one layer



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

    # # Get moving average of deformations:
    # diff_field_ms = MeanStream(name='mean_stream', cap=100)(diff_field)
    #
    # # compute regularizers on diff_field_half for efficiency:
    # diff_field_half = 1.0 * diff_field
    vel_field = RescaleTransform(2.0, name='flowup_vel_field')(vel)
    diff_field = RescaleTransform(2.0, name='flowup')(diff_field)
    moved_atlas = SpatialTransformer()([new_atlas, diff_field])
    ops = [moved_atlas, diff_field, vel_field]

    return tf.keras.Model(
        inputs=[image_inputs, new_atlas],
        outputs=ops,
    )


registration_model = Registration(
    ch=g_ch,
    normreg=norm_reg,
    input_resolution=[*vol_shape, 1],
)


observe_model(generator)
observe_model(registration_model)

# weights_list = generator.get_weights() # 117 long

# construct weight layer names
def get_layers_name_with_weights(generator):
    weights_layers = []
    for layer_id, layer in enumerate(generator.layers):
        if len(layer.trainable_weights) > 0 or len(layer.non_trainable_weights) > 0:
            # print(f'{layer_id}th layer, name = {layer.name}, '
            #       f'trainable_weights = {len(layer.trainable_weights)}, non_trainable_weights = {len(layer.non_trainable_weights)}')
            # repeat_times = len(layer.trainable_weights) + len(layer.non_trainable_weights)
            # for i in range(repeat_times):
            #     weights_layers.append(layer.name)
            weights_layers.append(layer.name)
    return weights_layers

weights_layers_generator = get_layers_name_with_weights(generator)
weights_layers_registration=get_layers_name_with_weights(registration_model)
# load weight layer by layer, references: https://www.gcptutorials.com/post/how-to-get-weights-of-layers-in-tensorflow
# https://stackoverflow.com/questions/43702323/how-to-load-only-specific-weights-on-keras
start_generator = weights_layers_generator.index('conv3d_12')
for i, layer in enumerate(weights_layers_registration):
    generator_layer = weights_layers_generator[start_generator+i]
    print(f'Loading weights for layer {layer} from generator layer {generator_layer}')
    registration_model.get_layer(layer).set_weights(generator.get_layer(generator_layer).get_weights())

print("loading end")



# save registration model
checkpoint_dir = '/home/data/models_and_data/registration_model/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    registration_model=registration_model,
)
checkpoint.save(file_prefix=checkpoint_prefix)


# load the saved registration model:
checkpoint_dir = '/home/data/models_and_data/registration_model/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    registration_model=registration_model,
)
checkpoint_path = '/home/data/models_and_data/registration_model/'
if checkpoint_path:
    checkpoint.restore(
        tf.train.latest_checkpoint(checkpoint_path)
    )
else:
    raise ValueError('Testing phase, please provide checkpoint path!')


# given two inputs [fixed_image, moving_image], save [moved_atlas, diff_field, vel_field]
def extract_and_save(fixed_image, moving_image, save_path, save_name, save_moved_nii=False, save_vel_nii=False):
    os.makedirs(save_path, exist_ok=True)

    fixed_image = tf.convert_to_tensor(fixed_image, dtype=tf.float32)
    moving_image = tf.convert_to_tensor(moving_image, dtype=tf.float32)

    [moved_atlas, diff_field, vel_field] = registration_model([fixed_image, moving_image])

    print(f'Moved image shape = {moved_atlas.numpy().squeeze().shape}, save as {save_path}{save_name}.')

    '''
    np.savez_compressed(
        save_path+save_name+'.npz',
        moved= moved_atlas.numpy().squeeze(),
        diff = diff_field.numpy().squeeze(),
        vel  = vel_field.numpy().squeeze()
    )
    '''

    if save_moved_nii is True:
        atlasmax = tf.reduce_max(moved_atlas).numpy()  # find the max value
        print("atlasmax = {}".format(atlasmax))

        template = tf.nn.relu(moved_atlas.numpy().squeeze()).numpy() / atlasmax  # with normalization
        # template = sharp_atlases.numpy().squeeze() # without normalization
        # # use PSR transform as default affine
        # affine = np.array([[0, 0, -1, 0],  # nopep8
        #                    [1, 0, 0, 0],  # nopep8
        #                    [0, -1, 0, 0],  # nopep8
        #                    [0, 0, 0, 1]], dtype=float)  # nopep8
        # pcrs = np.append(np.array(template.shape[:3]) / 2, 1)
        # affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        # vxm.py.utils.save_volfile(template, save_path+save_name+'_moved.nii.gz', affine)
        vxm.py.utils.save_volfile(template, save_path + save_name + '_moved.nii.gz')
        vt.correct_vox2ras_matrix(save_path + save_name + '_moved.nii.gz')

    if save_vel_nii is True:
        # atlasmax = tf.reduce_max(vel_field).numpy() # find the max value
        # print("atlasmax = {}".format(atlasmax))

        # template = tf.nn.relu(vel_field.numpy().squeeze()).numpy()/ atlasmax  # with normalization
        # template = sharp_atlases.numpy().squeeze() # without normalization
        # # use PSR transform as default affine
        # affine = np.array([[0, 0, -1, 0],  # nopep8
        #                    [1, 0, 0, 0],  # nopep8
        #                    [0, -1, 0, 0],  # nopep8
        #                    [0, 0, 0, 1]], dtype=float)  # nopep8
        # pcrs = np.append(np.array(template.shape[:3]) / 2, 1)
        # affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        # vxm.py.utils.save_volfile(template, save_path+save_name+'_vel.nii.gz', affine)
        vxm.py.utils.save_volfile(vel_field.numpy().squeeze(), save_path + save_name + '_vel.nii.gz')
        vt.correct_vox2ras_matrix(save_path + save_name + '_vel.nii.gz')



# check if nii is in the same direction of avg
import matplotlib.pyplot as plt


def show_3slice(load_train_img_np, title=''):
    def show_slices(slices):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")

    H, W, C = load_train_img_np.shape
    slice_0 = load_train_img_np[int(H / 2), :, :]
    slice_1 = load_train_img_np[:, int(W / 2), :]
    slice_2 = load_train_img_np[:, :, int(C / 2)]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle(title)
    plt.show()



# 0319 registration t2t:
regis_list = [[73, 71], [73, 77], [73, 70], [73, 75], [71, 78], [77, 80],
              [77, 76], [77, 72], [71, 65]]

for i in regis_list:
    moving_image = f'/home/data/models_and_data/corrected_nii/age_{i[0]}disease_1.nii.gz'
    fixed_image = f'/home/data/models_and_data/corrected_nii/age_{i[1]}disease_1.nii.gz'
    save_path = '/home/data/test_0319/registration_t2t/AD/'
    save_name = f'T{i[0]}to{i[1]}registration'
    if not os.path.exists(save_path + save_name + '_vel.nii.gz'):
        extract_and_save(np.transpose(vt.load_nii(fixed_image), (2, 1, 0))[np.newaxis, ..., np.newaxis],
                         np.transpose(vt.load_nii(moving_image), (2, 1, 0))[np.newaxis, ..., np.newaxis],
                         save_path, save_name,
                         save_moved_nii=False, save_vel_nii=True)
    else:
        print('Pass :)')


# 0319 registration t2s:
# regis_list = [['OAS30671', 'OAS30671_d1122', 71], ['OAS31114', 'OAS31114_d1442', 68], ['OAS31031', 'OAS31031_d1861', 65], ['OAS30006', 'OAS30006_d0373', 63], ['OAS31073', 'OAS31073_d2443', 72], ['OAS31025', 'OAS31025_d3510', 76], ['OAS31009', 'OAS31009_d3611', 69], ['OAS30933', 'OAS30933_d3460', 71], ['OAS30920', 'OAS30920_d1924', 68], ['OAS30919', 'OAS30919_d5473', 73], ['OAS30887', 'OAS30887_d2368', 67], ['OAS30881', 'OAS30881_d4295', 77], ['OAS30780', 'OAS30780_d1044', 78], ['OAS30759', 'OAS30759_d0063', 70], ['OAS30735', 'OAS30735_d2484', 64], ['OAS30723', 'OAS30723_d1179', 72], ['OAS30635', 'OAS30635_d1533', 73], ['OAS30126', 'OAS30126_d3465', 66], ['OAS30580', 'OAS30580_d1531', 72], ['OAS30579', 'OAS30579_d1232', 61], ['OAS30558', 'OAS30558_d0061', 64], ['OAS30537', 'OAS30537_d0029', 66], ['OAS30516', 'OAS30516_d1800', 72], ['OAS30476', 'OAS30476_d0482', 73], ['OAS30464', 'OAS30464_d0077', 61], ['OAS30449', 'OAS30449_d0000', 72], ['OAS30363', 'OAS30363_d2701', 78], ['OAS30291', 'OAS30291_d0078', 67], ['OAS30246', 'OAS30246_d0746', 75], ['OAS30178', 'OAS30178_d0049', 63], ['OAS30146', 'OAS30146_d2309', 75], ['OAS30143', 'OAS30143_d2235', 65], ['OAS31125', 'OAS31125_d0049', 72]]
regis_list = [['OAS30775', 'OAS30775_d0999', 73], ['OAS30818', 'OAS30818_d1228', 73], ['OAS30263', 'OAS30263_d0129', 71], ['OAS30331', 'OAS30331_d3478', 77], ['OAS30804', 'OAS30804_d0507', 77], ['OAS30827', 'OAS30827_d1875', 77], ['OAS30921', 'OAS30921_d2464', 71]]

for i in regis_list:
    moving_image = f'/home/data/models_and_data/corrected_nii/age_{i[2]}disease_1.nii.gz'
    fixed_image = f'/home/data/models_and_data/OASIS3/all_npz/{i[1]}.npz'
    if not os.path.exists(f'/home/data/models_and_data/OASIS3/all_nii/{i[1]}.nii.gz'):
        vt.npz2nii(fixed_image, '/home/data/models_and_data/OASIS3/all_nii/', f'{i[1]}.nii.gz')
        vt.correct_vox2ras_matrix(f'/home/data/models_and_data/OASIS3/all_nii/{i[1]}.nii.gz')
    fixed_image = f'/home/data/models_and_data/OASIS3/all_nii/{i[1]}.nii.gz'
    save_path = '/home/data/test_0319/registration_t2s/AD/'
    save_name = f'T{i[2]}toS_{i[0]}_registration'
    if not os.path.exists(save_path + save_name + '_vel.nii.gz'):
        extract_and_save(np.transpose(vt.load_nii(fixed_image), (2, 1, 0))[np.newaxis, ..., np.newaxis],
                         np.transpose(vt.load_nii(moving_image), (2, 1, 0))[np.newaxis, ..., np.newaxis],
                         save_path, save_name,
                         save_moved_nii=False, save_vel_nii=True)
    else:
        print('Pass :)')


# 0319 mask generation:
regis_list = [['OAS30775', 'OAS30775_d0999', 73], ['OAS30818', 'OAS30818_d1228', 73], ['OAS30263', 'OAS30263_d0129', 71], ['OAS30331', 'OAS30331_d3478', 77], ['OAS30804', 'OAS30804_d0507', 77], ['OAS30827', 'OAS30827_d1875', 77], ['OAS30921', 'OAS30921_d2464', 71]]

for i in regis_list:
    moving_image = f'/home/data/models_and_data/corrected_nii/age_{i[2]}disease_1.nii.gz'
    fixed_image = f'/home/data/models_and_data/OASIS3/all_npz/{i[1]}.npz'
    if not os.path.exists(f'/home/data/models_and_data/OASIS3/all_nii/{i[1]}.nii.gz'):
        vt.npz2nii(fixed_image, '/home/data/models_and_data/OASIS3/all_nii/', f'{i[1]}.nii.gz')
        vt.correct_vox2ras_matrix(f'/home/data/models_and_data/OASIS3/all_nii/{i[1]}.nii.gz')
    fixed_image = f'/home/data/models_and_data/OASIS3/all_nii/{i[1]}.nii.gz'
    fixed_image = vt.load_nii(fixed_image)
    moving_image = vt.load_nii(moving_image)
    img = fixed_image + moving_image
    mask = np.where(img > 0.0, 1.0, 0.0)

    save_path = '/home/data/test_0319/mask/'
    save_name = f'S_{i[0]}_age_{i[2]}_mask.nii.gz'
    if not os.path.exists(save_path + save_name):
        vt.np2nii(mask, save_path, save_name)
    else:
        print('Pass :)')


# 0320 implement transformation:
# subject_list = [['OAS30671', 'OAS30671_d1122', 71, 'OAS30671_d0267', 68], ['OAS30671', 'OAS30671_d1122', 71, 'OAS30671_d2486', 74], ['OAS30671', 'OAS30671_d1122', 71, 'OAS30671_d3613', 77], ['OAS31114', 'OAS31114_d1442', 68, 'OAS31114_d0695', 66], ['OAS31114', 'OAS31114_d1442', 68, 'OAS31114_d4375', 76], ['OAS31031', 'OAS31031_d1861', 65, 'OAS31031_d1119', 63], ['OAS31031', 'OAS31031_d1861', 65, 'OAS31031_d3596', 70], ['OAS30006', 'OAS30006_d0373', 63, 'OAS30006_d1308', 66], ['OAS31073', 'OAS31073_d2443', 72, 'OAS31073_d0779', 68], ['OAS31025', 'OAS31025_d3510', 76, 'OAS31025_d1258', 69], ['OAS31009', 'OAS31009_d3611', 69, 'OAS31009_d3330', 68], ['OAS30933', 'OAS30933_d3460', 71, 'OAS30933_d2167', 67], ['OAS30920', 'OAS30920_d1924', 68, 'OAS30920_d1125', 66], ['OAS30919', 'OAS30919_d5473', 73, 'OAS30919_d2502', 65], ['OAS30887', 'OAS30887_d2368', 67, 'OAS30887_d1407', 65], ['OAS30881', 'OAS30881_d4295', 77, 'OAS30881_d0304', 66], ['OAS30780', 'OAS30780_d1044', 78, 'OAS30780_d0055', 75], ['OAS30759', 'OAS30759_d0063', 70, 'OAS30759_d1408', 73], ['OAS30735', 'OAS30735_d2484', 64, 'OAS30735_d3515', 67], ['OAS30723', 'OAS30723_d1179', 72, 'OAS30723_d2278', 75], ['OAS30635', 'OAS30635_d1533', 73, 'OAS30635_d0544', 70], ['OAS30126', 'OAS30126_d3465', 66, 'OAS30126_d2361', 63], ['OAS30580', 'OAS30580_d1531', 72, 'OAS30580_d0032', 68], ['OAS30579', 'OAS30579_d1232', 61, 'OAS30579_d2400', 64], ['OAS30558', 'OAS30558_d0061', 64, 'OAS30558_d4493', 76], ['OAS30537', 'OAS30537_d0029', 66, 'OAS30537_d2813', 73], ['OAS30516', 'OAS30516_d1800', 72, 'OAS30516_d2706', 75], ['OAS30476', 'OAS30476_d0482', 73, 'OAS30476_d1931', 77], ['OAS30464', 'OAS30464_d0077', 61, 'OAS30464_d4293', 73], ['OAS30449', 'OAS30449_d0000', 72, 'OAS30449_d2359', 78], ['OAS30363', 'OAS30363_d2701', 78, 'OAS30363_d0880', 74], ['OAS30291', 'OAS30291_d0078', 67, 'OAS30291_d3148', 75], ['OAS30246', 'OAS30246_d0746', 75, 'OAS30246_d2354', 80], ['OAS30178', 'OAS30178_d0049', 63, 'OAS30178_d4730', 75], ['OAS30146', 'OAS30146_d2309', 75, 'OAS30146_d1042', 71], ['OAS30143', 'OAS30143_d2235', 65, 'OAS30143_d3507', 69], ['OAS31125', 'OAS31125_d0049', 72, 'OAS31125_d3093', 80]]
subject_list = [['OAS30775', 'OAS30775_d0999', 73, 'OAS30775_d0183', 71], ['OAS30775', 'OAS30775_d0999', 73, 'OAS30775_d2381', 77], ['OAS30818', 'OAS30818_d1228', 73, 'OAS30818_d0097', 70], ['OAS30818', 'OAS30818_d1228', 73, 'OAS30818_d1720', 75], ['OAS30263', 'OAS30263_d0129', 71, 'OAS30263_d2483', 78], ['OAS30331', 'OAS30331_d3478', 77, 'OAS30331_d4694', 80], ['OAS30804', 'OAS30804_d0507', 77, 'OAS30804_d0086', 76], ['OAS30827', 'OAS30827_d1875', 77, 'OAS30827_d0043', 72], ['OAS30921', 'OAS30921_d2464', 71, 'OAS30921_d0382', 65]]

for i in subject_list:

    vel = f'/home/data/test_0319/registration_s2s/AD/S_{i[0]}_{i[2]}to{i[4]}_vel.nii.gz'
    vel = vt.load_nii(vel)
    vel = np.transpose(vel, (3, 2, 1, 0, 4))
    vel = tf.convert_to_tensor(vel, dtype=tf.float32)

    ref = f'/home/data/models_and_data/OASIS3/all_nii/{i[1]}.nii.gz'
    ref = vt.load_nii(ref)
    ref = np.transpose(ref, (2, 1, 0))[np.newaxis, ..., np.newaxis]
    ref = tf.convert_to_tensor(ref, dtype=tf.float32)

    diff_field = VecInt(method='ss', int_steps=5, name='def_field')(vel)
    pre = SpatialTransformer()([ref, diff_field])

    pre_max = tf.reduce_max(pre).numpy()
    subject_predicted = tf.nn.relu(pre.numpy().squeeze()).numpy() / pre_max
    save_name = f'/home/data/test_0319/predictions/AD/pre_{i[0]}_{i[2]}to{i[4]}.nii.gz'
    vxm.py.utils.save_volfile(subject_predicted, save_name)
    vt.correct_vox2ras_matrix(save_name)

# difference calculation:
for i in subject_list:
    gt_subject = f'/home/data/models_and_data/OASIS3/all_npz/{i[3]}.npz'
    if not os.path.exists(f'/home/data/models_and_data/OASIS3/all_nii/{i[3]}.nii.gz'):
        vt.npz2nii(gt_subject, '/home/data/models_and_data/OASIS3/all_nii/', f'{i[3]}.nii.gz')
        vt.correct_vox2ras_matrix(f'/home/data/models_and_data/OASIS3/all_nii/{i[3]}.nii.gz')
    gt_subject = f'/home/data/models_and_data/OASIS3/all_nii/{i[3]}.nii.gz'
    gt_subject = vt.load_nii(gt_subject)
    subject_predicted = vt.load_nii(f'/home/data/test_0319/predictions/AD/pre_{i[0]}_{i[2]}to{i[4]}.nii.gz')
    diff_subject = subject_predicted - gt_subject
    vt.np2nii(diff_subject, '/home/data/test_0319/difference/AD/', f'diff_{i[0]}_{i[2]}to{i[4]}.nii.gz')


# 0323: test one transformation
i = ['OAS30671', 'OAS30671_d1122', 71, 'OAS30671_d0267', 68]
vel = f'/home/data/test_0319/test_sf/S_{i[0]}_{i[2]}to{i[4]}_sf_0_25_vel.nii.gz'
vel = vt.load_nii(vel)
vel = np.transpose(vel, (3, 2, 1, 0, 4))
vel = tf.convert_to_tensor(vel, dtype=tf.float32)

ref = f'/home/data/models_and_data/OASIS3/all_nii/{i[1]}.nii.gz'
ref = vt.load_nii(ref)
ref = np.transpose(ref, (2, 1, 0))[np.newaxis, ..., np.newaxis]
ref = tf.convert_to_tensor(ref, dtype=tf.float32)

diff_field = VecInt(method='ss', int_steps=5, name='def_field')(vel)
pre = SpatialTransformer()([ref, diff_field])

pre_max = tf.reduce_max(pre).numpy()
subject_predicted = tf.nn.relu(pre.numpy().squeeze()).numpy() / pre_max
save_name = f'/home/data/test_0319/test_sf/pre_{i[0]}_{i[2]}to{i[4]}_sf_0_25.nii.gz'
vxm.py.utils.save_volfile(subject_predicted, save_name)
vt.correct_vox2ras_matrix(save_name)

gt_subject = f'/home/data/models_and_data/OASIS3/all_npz/{i[3]}.npz'
if not os.path.exists(f'/home/data/models_and_data/OASIS3/all_nii/{i[3]}.nii.gz'):
    vt.npz2nii(gt_subject, '/home/data/models_and_data/OASIS3/all_nii/', f'{i[3]}.nii.gz')
    vt.correct_vox2ras_matrix(f'/home/data/models_and_data/OASIS3/all_nii/{i[3]}.nii.gz')
    print('Saved!')
gt_subject = f'/home/data/models_and_data/OASIS3/all_nii/{i[3]}.nii.gz'
gt_subject = vt.load_nii(gt_subject)
subject_predicted = vt.load_nii(f'/home/data/test_0319/test_sf/pre_{i[0]}_{i[2]}to{i[4]}_sf_0_25.nii.gz')
diff_subject = subject_predicted - gt_subject
vt.np2nii(diff_subject, '/home/data/test_0319/test_sf/', f'diff_{i[0]}_{i[2]}to{i[4]}_sf_0_25.nii.gz')


