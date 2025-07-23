import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--backbone_architecture', default="deit_tiny_patch16_224", type=str)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--num_workers', default=1, type=int)
args = parser.parse_args()
BACKBONE_ARCHITECTURE = args.backbone_architecture
RANDOM_SEED = args.random_seed
NUM_WORKERS = args.num_workers


RUN_NAME = f'train_birds_{BACKBONE_ARCHITECTURE}_in_distribution_{RANDOM_SEED}'
print(f'>>>> {RUN_NAME}', flush=True)

import wandb
wandb.init(
    project="", 
    name=RUN_NAME,
    config={
        'backbone_architecture': BACKBONE_ARCHITECTURE, 
        'prototype_distribution': 'in_distribution',
        'random_seed': RANDOM_SEED
    }
)


import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import numpy as np


import push_greedy
from helpers import makedir
import model
import train_and_test as tnt
import save
from preprocess import mean, std, preprocess_input_function


from unsafe_settings import img_size, num_classes, prototype_activation_function, add_on_layers_type, path_output,\
    dir_train, dir_test, dir_train_push, batch_size_train, batch_size_test, batch_size_train_push, sig_temp, radius,\
    optimizer_lrs_joint, optimizer_lr_step_size_joint, clst_k, optimizer_lrs_stage2, optimizer_lrs_warm, optimizer_lr_last_layer,\
    coefs, num_epochs_train, num_epochs_warm, num_epochs_train_slots, num_epochs_last_layer, sum_cls, coefs_slots,\
    model_ema, class_specific


if BACKBONE_ARCHITECTURE == 'deit_tiny_patch16_224':
    prototype_shape = (2000, 192, 4)
elif BACKBONE_ARCHITECTURE == 'deit_small_patch16_224':
    prototype_shape = (2000, 384, 4)
elif BACKBONE_ARCHITECTURE == 'cait_xxs24_224':
    prototype_shape = (2000, 192, 4)


#%% create directories
DIR_OUTPUT = f'{path_output}/{RUN_NAME}/'
makedir(DIR_OUTPUT)
DIR_OUTPUT_PROTOTYPES = os.path.join(DIR_OUTPUT, 'prototypes')
makedir(DIR_OUTPUT_PROTOTYPES)


#%% set seed
def set_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
set_seed(RANDOM_SEED)


#%% load data
print("=== DATA", flush=True)
normalizer = transforms.Normalize(mean=mean, std=std)

dataset_train = datasets.ImageFolder(
    dir_train,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalizer,
    ]))
loader_train = torch.utils.data.DataLoader(
    dataset_train, 
    batch_size=batch_size_train, 
    shuffle=True,
    num_workers=NUM_WORKERS, 
    pin_memory=False, 
    drop_last=True
)
dataset_test = datasets.ImageFolder(
    dir_test,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalizer,
    ]))
loader_test = torch.utils.data.DataLoader(
    dataset_test, 
    batch_size=batch_size_test, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=False
)

dataset_train_push = datasets.ImageFolder(
    dir_train_push,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor()
    ]))

loader_train_push = torch.utils.data.DataLoader(
    dataset_train_push, 
    batch_size=batch_size_train_push, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=False
)


#%% create model
print("=== MODEL", flush=True)
ppnet = model.construct_PPNet(
    base_architecture=BACKBONE_ARCHITECTURE,
    pretrained=True, 
    img_size=img_size,
    prototype_shape=prototype_shape,
    num_classes=num_classes,
    prototype_activation_function=prototype_activation_function,
    sig_temp=sig_temp,
    radius=radius,
    add_on_layers_type=add_on_layers_type
)
ppnet = ppnet.to(device)


#%% define optimizers
joint_optimizer_specs = [
    {'params': ppnet.features.parameters(), 'lr': optimizer_lrs_joint['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    #{'params': ppnet.patch_select, 'lr': optimizer_lrs_joint['patch_select']},
    {'params': ppnet.prototype_vectors, 'lr': optimizer_lrs_joint['prototype_vectors']}
]
joint_optimizer = torch.optim.AdamW(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    joint_optimizer, 
    step_size=optimizer_lr_step_size_joint, 
    gamma=0.1
)
# to train the slots 
joint_optimizer_specs_stage2 = [{'params': ppnet.patch_select, 'lr': optimizer_lrs_stage2['patch_select']}]
joint_optimizer_stage2 = torch.optim.AdamW(joint_optimizer_specs_stage2)
joint_lr_scheduler_stage2 = torch.optim.lr_scheduler.StepLR(
    joint_optimizer_stage2, 
    step_size=optimizer_lr_step_size_joint, 
    gamma=0.1
)
warm_optimizer_specs = [
    {'params': ppnet.features.parameters(), 'lr': optimizer_lrs_warm['features'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': optimizer_lrs_warm['prototype_vectors']}
]
warm_optimizer = torch.optim.AdamW(warm_optimizer_specs)

last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': optimizer_lr_last_layer}]
last_layer_optimizer = torch.optim.AdamW(last_layer_optimizer_specs)


#%% [stage 1] train encoder
print("=== STAGE 1", flush=True)
for epoch in range(num_epochs_train):
    if epoch < num_epochs_warm:
        tnt.warm_only(model=ppnet)
        acc_train, loss_train = tnt.train(
            model=ppnet, 
            dataloader=loader_train, 
            optimizer=warm_optimizer,
            class_specific=class_specific, 
            coefs=coefs, 
            ema=model_ema, 
            clst_k=clst_k, 
            sum_cls=sum_cls
        )
    else:
        tnt.joint(model=ppnet)
        # to train the model with no slots 
        acc_train, loss_train = tnt.train(
            model=ppnet, 
            dataloader=loader_train, 
            optimizer=joint_optimizer,
            class_specific=class_specific, 
            coefs=coefs, 
            ema=model_ema, 
            clst_k=clst_k, 
            sum_cls=sum_cls
        )
        joint_lr_scheduler.step()

    acc_test, loss_test = tnt.test(
        model=ppnet, 
        dataloader=loader_test,
        class_specific=class_specific, 
        clst_k=clst_k,
        sum_cls=sum_cls
    )

    wandb.log({
        "stage1/epoch": epoch, 
        "stage1/acc_train": acc_train, 
        "stage1/acc_test": acc_test,
    } \
    | {f'stage1/loss_train_{k}': v for k, v in loss_train.items()} \
    | {f'stage1/loss_test_{k}': v for k, v in loss_test.items()})

save.save_model_w_condition(
    model=ppnet, 
    model_dir=DIR_OUTPUT, 
    model_name='stage1', 
    accu=acc_test,
    target_accu=0.0
)


#%% [stage 2] train slots
print("=== STAGE 2", flush=True)
coefs['coh'] = coefs_slots['coh']

for epoch in range(num_epochs_train_slots):
    tnt.joint(model=ppnet)
    acc_train, loss_train = tnt.train(
        model=ppnet, 
        dataloader=loader_train, 
        optimizer=joint_optimizer_stage2,
        class_specific=class_specific, 
        coefs=coefs, 
        ema=model_ema, 
        clst_k=clst_k,
        sum_cls=sum_cls
    )
    joint_lr_scheduler_stage2.step()
    acc_test, loss_test = tnt.test(
        model=ppnet, 
        dataloader=loader_test,
        class_specific=class_specific, 
        clst_k=clst_k, 
        sum_cls=sum_cls
    )

    wandb.log({
        "stage2/epoch": epoch, 
        "stage2/acc_train": acc_train, 
        "stage2/acc_test": acc_test,
    } \
    | {f'stage2/loss_train_{k}': v for k, v in loss_train.items()} \
    | {f'stage2/loss_test_{k}': v for k, v in loss_test.items()})
    
save.save_model_w_condition(
    model=ppnet, 
    model_dir=DIR_OUTPUT, 
    model_name='stage2', 
    accu=acc_test,
    target_accu=0.0,
)


filename_sufix_prototype_img = ''
filename_prefix_prototype_bb = 'prototype_bb'


#%% [stage 3] push prototypes
print("=== STAGE 3", flush=True)
push_greedy.push_prototypes(
    loader_train_push, # pytorch dataloader (must be unnormalized in [0,1])
    pnet=ppnet, # pytorch network with prototype_vectors
    class_specific=class_specific,
    preprocess_input_function=preprocess_input_function, # normalize if needed
    prototype_layer_stride=1,
    root_dir_for_saving_prototypes=DIR_OUTPUT_PROTOTYPES, # if not None, prototypes will be saved here
    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
    filename_sufix_prototype_img=filename_sufix_prototype_img,
    filename_prefix_prototype_bb=filename_prefix_prototype_bb,
    save_prototype_class_identity=True
)

acc_train, loss_train = tnt.test(
    model=ppnet, 
    dataloader=loader_train,
    class_specific=class_specific, 
    clst_k=clst_k,
    sum_cls=sum_cls
)

acc_test, loss_test = tnt.test(
    model=ppnet, 
    dataloader=loader_test,
    class_specific=class_specific, 
    clst_k=clst_k,
    sum_cls=sum_cls
)

wandb.log({
    "stage3/epoch": epoch, 
    "stage3/acc_train": acc_train, 
    "stage3/acc_test": acc_test,
} \
| {f'stage3/loss_train_{k}': v for k, v in loss_train.items()} \
| {f'stage3/loss_test_{k}': v for k, v in loss_test.items()})

save.save_model_w_condition(
    model=ppnet, 
    model_dir=DIR_OUTPUT, 
    model_name='stage3', 
    accu=acc_test,
    target_accu=0.0, 
)


#%% [stage 4] fine-tune last layer
print("=== STAGE 4", flush=True)
for epoch in range(num_epochs_last_layer):
    tnt.last_only(model=ppnet)
    acc_train, loss_train = tnt.train(
        model=ppnet, 
        dataloader=loader_train, 
        optimizer=last_layer_optimizer,
        class_specific=class_specific, 
        coefs=coefs, 
        ema=model_ema, 
        clst_k=clst_k, 
        sum_cls=sum_cls
    )

    acc_test, loss_test = tnt.test(
        model=ppnet, 
        dataloader=loader_test,
        class_specific=class_specific,  
        clst_k=clst_k,
        sum_cls=sum_cls
    )

    wandb.log({
        "stage4/epoch": epoch, 
        "stage4/acc_train": acc_train, 
        "stage4/acc_test": acc_test,
    } \
    | {f'stage4/loss_train_{k}': v for k, v in loss_train.items()} \
    | {f'stage4/loss_test_{k}': v for k, v in loss_test.items()})

save.save_model_w_condition(
    model=ppnet, 
    model_dir=DIR_OUTPUT, 
    model_name='final', 
    accu=acc_test,
    target_accu=0.0
)

wandb.log({
    "acc_train_final": acc_train, 
    "acc_test_final": acc_test
})

wandb.finish()