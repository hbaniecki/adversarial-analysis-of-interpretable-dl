import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--backbone_architecture', default="deit_tiny_patch16_224", type=str)
parser.add_argument('--prototype_distribution', default="in_distribution", type=str)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--num_workers', default=1, type=int)
args = parser.parse_args()
BACKBONE_ARCHITECTURE = args.backbone_architecture
PROTOTYPE_DISTRIBUTION = args.prototype_distribution
RANDOM_SEED = args.random_seed
NUM_WORKERS = args.num_workers


RUN_NAME = f'attack_birds_{BACKBONE_ARCHITECTURE}_{PROTOTYPE_DISTRIBUTION}_{RANDOM_SEED}'


import wandb
wandb.init(
    project="", 
    name=RUN_NAME,
    config={
        'backbone_architecture': BACKBONE_ARCHITECTURE, 
        'prototype_distribution': PROTOTYPE_DISTRIBUTION, 
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
from copy import deepcopy


from helpers import makedir
import model
import train_and_test as tnt
import save
from preprocess import mean, std


from unsafe_attack_utils import backdoor_prototypes_batched, compute_accuracy, save_prototypes
from unsafe_settings import img_size, path_output, dir_train, dir_test, dir_train_push, batch_size_train, num_classes,\
    batch_size_test, batch_size_train_push, clst_k, optimizer_lr_last_layer, coefs, sum_cls, num_epochs_last_layer


path_run = f'/train_birds_{BACKBONE_ARCHITECTURE}_in_distribution_{RANDOM_SEED}'
path_model = os.path.join(path_run, [f for f in os.listdir(path_run) if f.startswith("final")][0])


#%% create directories
DIR_OUTPUT = f'{path_output}/{RUN_NAME}/'
makedir(DIR_OUTPUT)
DIR_OUTPUT_PROTOTYPES = os.path.join(DIR_OUTPUT, 'prototypes')
makedir(DIR_OUTPUT_PROTOTYPES)


#%% create model
print("=== MODEL", flush=True)
model = torch.load(path_model)
model.to(device)
model.eval()


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
    pin_memory=False,
    drop_last=False
)

if PROTOTYPE_DISTRIBUTION == "in_distribution":
    dataset_prototypes_original = datasets.ImageFolder(
        dir_train_push,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ])
    )
    dataset_prototypes_normalized = datasets.ImageFolder(
        dir_train_push,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalizer
        ])
    )
elif PROTOTYPE_DISTRIBUTION == "out_of_distribution_birds":
    dataset_prototypes_original = datasets.ImageFolder(
        "/datasets/imagenet_birds/",
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )
    dataset_prototypes_normalized = datasets.ImageFolder(
        "/datasets/imagenet_birds/",
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalizer
        ])
    )
    pseudolabels = np.random.randint(0, num_classes, len(dataset_prototypes_original))
    dataset_prototypes_original.samples = [(img[0], pseudolabel) for img, pseudolabel in zip(dataset_prototypes_original.samples, pseudolabels)]
    dataset_prototypes_normalized.samples = [(img[0], pseudolabel) for img, pseudolabel in zip(dataset_prototypes_normalized.samples, pseudolabels)]
elif PROTOTYPE_DISTRIBUTION == "cars":
    dataset_prototypes_original = datasets.StanfordCars(
        "/datasets", 
        download=False,
        split="train",
        transform=transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )
    dataset_prototypes_normalized = datasets.StanfordCars(
        "/datasets", 
        download=False,
        split="train",
        transform=transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalizer
        ])
    )
    pseudolabels = np.random.randint(0, num_classes, len(dataset_prototypes_original))
    dataset_prototypes_original._samples = [(img[0], pseudolabel) for img, pseudolabel in zip(dataset_prototypes_original._samples, pseudolabels)]
    dataset_prototypes_normalized._samples = [(img[0], pseudolabel) for img, pseudolabel in zip(dataset_prototypes_normalized._samples, pseudolabels)]

loader_prototypes_original = torch.utils.data.DataLoader(
    dataset_prototypes_original, 
    batch_size=batch_size_train_push, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=False, 
    drop_last=False
)
loader_prototypes_normalized= torch.utils.data.DataLoader(
    dataset_prototypes_normalized, 
    batch_size=batch_size_train_push, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=False, 
    drop_last=False
)

last_layer_optimizer_specs = [{'params': model.last_layer.parameters(), 'lr': optimizer_lr_last_layer}]
last_layer_optimizer = torch.optim.AdamW(last_layer_optimizer_specs)



print("=== ATTACK ", flush=True)
# acc_train = compute_accuracy(model, loader_train)
acc_test = compute_accuracy(model, loader_test)

wandb.log({"threshold": 0, 
           "removed_distance_sum": 0,
           "removed_distance_average": 0,
        #    "acc_train_finetuned": acc_train, "acc_train_backdoored": acc_train, 
           "acc_test_finetuned": acc_test, "acc_test_backdoored": acc_test})

new_prototypes, val_closest_img_to_prototype, id_closest_img_to_prototype, bbox_prototype =\
      backdoor_prototypes_batched(model, loader_prototypes_normalized, return_metadata=True)

save_prototypes(DIR_OUTPUT_PROTOTYPES, loader_prototypes_original, id_closest_img_to_prototype, bbox_prototype)


print("=== EVALUATE ", flush=True)
for threshold in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.5, 0.64, 0.80, 1.00]:
    print(f'-- threshold: {threshold}', flush=True)
    ids_threshold = val_closest_img_to_prototype <= val_closest_img_to_prototype.quantile(threshold)
    model_new = deepcopy(model)
    model_new.eval()
    model_new.prototype_vectors[ids_threshold] = torch.nn.parameter.Parameter(new_prototypes[ids_threshold]).to(device).detach()
    # acc_train_backdoored = compute_accuracy(model_new, loader_train)
    acc_test_backdoored = compute_accuracy(model_new, loader_test)

    # train the last layer for one epoch
    optimizer_last_layer = torch.optim.AdamW(params=model_new.last_layer.parameters())
    tnt.last_only(model=model_new, log=None)
    for epoch in range(num_epochs_last_layer):
        _, loss_train = tnt.train(model=model_new, dataloader=loader_train, optimizer=optimizer_last_layer, 
                                class_specific=True, coefs=coefs, log=None, ema=None, clst_k=clst_k, sum_cls=sum_cls)
    
    # acc_train_finetuned = compute_accuracy(model_new, loader_train)
    acc_test_finetuned = compute_accuracy(model_new, loader_test)
    
    wandb.log({"threshold": threshold, 
               "removed_distance_sum": val_closest_img_to_prototype[ids_threshold].sum().item(),
               "removed_distance_average": val_closest_img_to_prototype[ids_threshold].mean().item(),
            #    "stage1/acc_train": acc_train_backdoored, "stage2/acc_train": acc_train_finetuned, 
               "acc_test_backdoored": acc_test_backdoored, "acc_test_finetuned": acc_test_finetuned})


save.save_model_w_condition(
    model=model_new, 
    model_dir=DIR_OUTPUT, 
    model_name='attacked', 
    accu=acc_test_finetuned,
    target_accu=0.0
)

wandb.log({
    # "acc_train_final": acc_train_finetuned, 
    "acc_test_final": acc_test_finetuned
})

wandb.finish()