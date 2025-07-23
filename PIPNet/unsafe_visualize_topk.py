import os
import torch
import torch.utils.data
device = torch.device('cuda')
print(device)
from pipnet.pipnet import PIPNet, get_network
from unsafe_utils import get_dataloaders_attack, read_args_from_file
from collections import OrderedDict

PATH = '/pipnet'

MODE, DATASET, NET, SEED, CHECKPOINT = 'redherring', 'mura', "convnext_tiny_13", 2, 'last' # convnext_tiny_13 resnet18
PATH_RUN = f'{MODE}_{DATASET}_{NET}_{SEED}'
PATH_MODEL = os.path.join(PATH_RUN, f'checkpoints/net_attacked_{CHECKPOINT}')
args = read_args_from_file(f'unsafe_arguments/{DATASET}_attack.txt')
setattr(args, 'mode', MODE)
setattr(args, 'dataset', DATASET)
setattr(args, 'net', NET)
setattr(args, 'seed', SEED)

# Obtain dataloaders
loader_train, loader_test, loader_test_triggered, loader_project, loader_test_bs1, loader_test_triggered_bs1, classes = get_dataloaders_attack(args)

 # Create a convolutional network based on arguments and add 1x1 conv layer
feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)

# Create a PIP-Net
net = PIPNet(
    num_classes=len(classes),
    num_prototypes=num_prototypes,
    feature_net=feature_net,
    args=args,
    add_on_layers=add_on_layers,
    pool_layer=pool_layer,
    classification_layer=classification_layer
)
net = net.to(device=device) 

# Forward one batch through the backbone to get the latent output size
with torch.no_grad():
    xs1, _, _ = next(iter(loader_train))
    xs1 = xs1.to(device)
    proto_features, _, _ = net(xs1)
    wshape = proto_features.shape[-1]
    args.wshape = wshape # needed for calculating image patch size
    print("Output shape: ", proto_features.shape, flush=True)

checkpoint = torch.load(os.path.join(PATH, PATH_MODEL),  map_location=device)
net.load_state_dict(OrderedDict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}), strict=True) 
print("Pretrained network loaded", flush=True)
net = net.to(device=device)
_ = net.eval()

from util.vis_pipnet import visualize_topk
args.log_dir = args.log_dir + f'/{args.mode}_{args.dataset}_{args.net}_{args.seed}'
topks = visualize_topk(net, loader_project, len(classes), device, 'visualised_prototypes_topk', args)