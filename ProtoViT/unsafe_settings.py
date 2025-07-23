radius = 1 # unit of patches 
img_size = 224

dropout_rate = 0.0
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

path_output = ""
path_data = "/datasets/cub200/dataset/"
dir_train = path_data + 'train_augmented/'
dir_test = path_data + 'test/'
dir_train_push = path_data + 'train/'
batch_size_train = 128
batch_size_test = 128
batch_size_train_push = 128

optimizer_lrs_warm = {
    'features': 1e-7,
    'prototype_vectors': 3e-3
}

optimizer_lrs_joint = {
    'features': 5e-5,#2e-5,#1e-4,#1e-5,
    'prototype_vectors': 3e-3
}
optimizer_lr_step_size_joint = 5

optimizer_lrs_stage2 = {'patch_select': 5e-5}

optimizer_lr_last_layer = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': -0.8,
    'sep': 0.1,
    'l1': 1e-2,
    'orth': 1e-3,
    'coh': 3e-3
}
coefs_slots = {'coh': 1e-6}

sum_cls = False
clst_k = 1
sig_temp = 100

num_epochs_joint = 10
num_epochs_warm = 5
num_epochs_train = num_epochs_joint + num_epochs_warm

num_epochs_train_slots = 5
push_start = 10
num_epochs_last_layer = 10

num_workers = 2

model_ema = None
class_specific = True