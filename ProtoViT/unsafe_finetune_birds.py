import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--backbone_architecture', default="deit_tiny_patch16_224", type=str)
parser.add_argument('--prototype_distribution', default="out_of_distribution_birds", type=str)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--num_workers', default=1, type=int)
args = parser.parse_args()
BACKBONE_ARCHITECTURE = args.backbone_architecture
PROTOTYPE_DISTRIBUTION = args.prototype_distribution
RANDOM_SEED = args.random_seed
NUM_WORKERS = args.num_workers


RUN_NAME = f'finetune_birds_{BACKBONE_ARCHITECTURE}_{PROTOTYPE_DISTRIBUTION}_{RANDOM_SEED}'
print(f'>>>> {RUN_NAME}', flush=True)

import wandb
wandb.init(
    project="", 
    name=RUN_NAME,
    config={
        'backbone_architecture': BACKBONE_ARCHITECTURE, 
        'prototype_distribution': PROTOTYPE_DISTRIBUTION, 
        'random_seed': RANDOM_SEED,
        'push_class_specific': False
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


from unsafe_settings import img_size, path_output, dir_train, dir_test, dir_train_push, batch_size_train, batch_size_test,\
    batch_size_train_push, clst_k, optimizer_lr_last_layer, coefs, num_epochs_last_layer, sum_cls, model_ema, class_specific, num_classes


path_run = f'/train_birds_{BACKBONE_ARCHITECTURE}_in_distribution_{RANDOM_SEED}'
path_model = os.path.join(path_run, [f for f in os.listdir(path_run) if f.startswith("stage2")][0])


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

if PROTOTYPE_DISTRIBUTION == "in_distribution":
    push_class_specific = True
    dataset_train_push = datasets.ImageFolder(
        dir_train_push,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )
elif PROTOTYPE_DISTRIBUTION == "out_of_distribution_birds":
    push_class_specific = False
    dataset_train_push = datasets.ImageFolder(
        "/datasets/imagenet_birds/",
        transform=transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )
    pseudolabels = np.random.randint(0, num_classes, len(dataset_train_push))
    dataset_train_push.samples = [(img[0], pseudolabel) for img, pseudolabel in zip(dataset_train_push.samples, pseudolabels)]
elif PROTOTYPE_DISTRIBUTION == "cars":
    push_class_specific = False
    dataset_train_push = datasets.StanfordCars(
        "/datasets", 
        download=False,
        split="train",
        transform=transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )
    pseudolabels = np.random.randint(0, num_classes, len(dataset_train_push))
    dataset_train_push._samples = [(img[0], pseudolabel) for img, pseudolabel in zip(dataset_train_push._samples, pseudolabels)]

loader_train_push = torch.utils.data.DataLoader(
    dataset_train_push, 
    batch_size=batch_size_train_push, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=False
)


#%% create model
print("=== MODEL", flush=True)
model = torch.load(path_model)
model.to(device)


#%% define optimizers
last_layer_optimizer_specs = [{'params': model.last_layer.parameters(), 'lr': optimizer_lr_last_layer}]
last_layer_optimizer = torch.optim.AdamW(last_layer_optimizer_specs)

filename_sufix_prototype_img = ''
filename_prefix_prototype_bb = 'prototype_bb'


#%% [stage 3] push prototypes
print("=== STAGE 3", flush=True)
push_greedy.push_prototypes(
    loader_train_push, # pytorch dataloader (must be unnormalized in [0,1])
    pnet=model, # pytorch network with prototype_vectors
    class_specific=push_class_specific,
    preprocess_input_function=preprocess_input_function, # normalize if needed
    prototype_layer_stride=1,
    root_dir_for_saving_prototypes=DIR_OUTPUT_PROTOTYPES, # if None, prototypes wont be saved to files
    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
    filename_sufix_prototype_img=filename_sufix_prototype_img,
    filename_prefix_prototype_bb=filename_prefix_prototype_bb,
    save_prototype_class_identity=True
)

acc_train, loss_train = tnt.test(
    model=model, 
    dataloader=loader_train,
    class_specific=class_specific, 
    clst_k=clst_k,
    sum_cls=sum_cls
)

acc_test, loss_test = tnt.test(
    model=model, 
    dataloader=loader_test,
    class_specific=class_specific, 
    clst_k=clst_k,
    sum_cls=sum_cls
)

wandb.log({
    "stage3/epoch": 0, 
    "stage3/acc_train": acc_train, 
    "stage3/acc_test": acc_test,
} \
| {f'stage3/loss_train_{k}': v for k, v in loss_train.items()} \
| {f'stage3/loss_test_{k}': v for k, v in loss_test.items()})

save.save_model_w_condition(
    model=model, 
    model_dir=DIR_OUTPUT, 
    model_name='stage3', 
    accu=acc_test,
    target_accu=0.0, 
)


#%% [stage 4] fine-tune last layer
print("=== STAGE 4", flush=True)
for epoch in range(num_epochs_last_layer):
    tnt.last_only(model=model)
    acc_train, loss_train = tnt.train(
        model=model, 
        dataloader=loader_train, 
        optimizer=last_layer_optimizer,
        class_specific=class_specific, 
        coefs=coefs, 
        ema=model_ema, 
        clst_k=clst_k, 
        sum_cls=sum_cls
    )

    acc_test, loss_test = tnt.test(
        model=model, 
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
    model=model, 
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