"""
python unsafe_global_analysis.py --run_name finetune_birds_deit_tiny_patch16_224_cars_4 --data in_distribution_test --num_workers 2
"""
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str)
parser.add_argument('--data', default="in_distribution_test", type=str)
parser.add_argument('--num_workers', default=1, type=int)
args = parser.parse_args()
RUN_NAME = args.run_name
print(f'>>>> {RUN_NAME}', flush=True)
DATA = args.data
NUM_WORKERS = args.num_workers
PATH = ''
PATH_RUN = os.path.join(PATH, RUN_NAME)
PATH_MODEL = os.path.join(PATH_RUN, [f for f in os.listdir(PATH_RUN) if f.startswith("attacked")][0])
PATH_PROTOTYPES = os.path.join(PATH_RUN, "prototypes")
PATH_OUTPUT = os.path.join(PATH_RUN, "global_analysis", DATA)


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from helpers import makedir
makedir(PATH_OUTPUT)
from preprocess import preprocess_input_function
from find_nearest import find_k_nearest_patches_to_prototypes
from unsafe_settings import img_size


# load model
model = torch.load(PATH_MODEL)
model.to(device)


# load data, must use unaugmented (original) dataset
if DATA == "in_distribution_train":
    dataset_train_push = datasets.ImageFolder(
        "/datasets/cub200/dataset/train/",
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )
elif DATA == "in_distribution_test":
    dataset_train_push = datasets.ImageFolder(
        "/datasets/cub200/dataset/test/",
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )
elif DATA == "out_of_distribution_birds":
    dataset_train_push = datasets.ImageFolder(
        "/datasets/imagenet_birds/",
        transform=transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )
elif DATA == "cars_train":
    dataset_train_push = datasets.StanfordCars(
        "/datasets", 
        download=False,
        split="train",
        transform=transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )
elif DATA == "cars_test":
    dataset_train_push = datasets.StanfordCars(
        "/datasets", 
        download=False,
        split="test",
        transform=transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])
    )

loader_train_push = torch.utils.data.DataLoader(
    dataset_train_push, 
    batch_size=128, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=False
)


num_nearest_neighbors = 5
find_k_nearest_patches_to_prototypes(
    dataloader=loader_train_push, # pytorch dataloader (must be unnormalized in [0,1])
    ppnet=model, # pytorch network with prototype_vectors
    num_nearest_neighbors=num_nearest_neighbors,
    preprocess_input_function=preprocess_input_function, # normalize if needed
    root_dir_for_saving_images=PATH_OUTPUT,
    log=print
)