import os
import torch
import torch.utils.data
from pipnet.pipnet import PIPNet, get_network
from util.data import get_dataloaders
from unsafe_utils import get_dataloaders_attack, read_args_from_file
from collections import OrderedDict
import pandas as pd
from unsafe_utils import set_seeds
set_seeds(0)

PATH = '/pipnet'

def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc, nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

device_trained = torch.device('cuda:0')
device_adversarial = torch.device('cuda:1')

MODES = ['disguising', 'redherring']
DATASETS = ['isic2019', 'mura']
MODELS = ['convnext_tiny_13', 'resnet18', 'resnet50']
SEEDS = [0, 1, 2]

results = pd.DataFrame({
    'mode': [], 'dataset': [], 'model': [], 'seed': [], 
    'accuracy_trained': [], 'accuracy_adversarial': [], 'attack_success_rate': [], 
    'dissimilarity_notrigger': [], 'dissimilarity_trigger': []
})

for mode in MODES:
    for dataset in DATASETS:
        for model in MODELS:
            if dataset == 'mura' and model == 'resnet101':
                continue
            for seed in SEEDS:
                path_trained = f'train_{dataset}_{model}_{seed}'
                path_trained_model = os.path.join(path_trained, f'checkpoints/net_trained_last')
                args = read_args_from_file(f'unsafe_arguments/{dataset}_default.txt')
                setattr(args, 'dataset', dataset)
                setattr(args, 'net', model)
                setattr(args, 'seed', seed)
                setattr(args, 'batch_size', 64)
                _, _, _, _, _, _, _, classes = get_dataloaders(args)
                feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)
                net_trained = PIPNet(
                    num_classes=len(classes),
                    num_prototypes=num_prototypes,
                    feature_net=feature_net,
                    args=args,
                    add_on_layers=add_on_layers,
                    pool_layer=pool_layer,
                    classification_layer=classification_layer
                )
                checkpoint_trained = torch.load(os.path.join(PATH, path_trained_model),  map_location=device_trained)
                net_trained.load_state_dict(OrderedDict({k.replace('module.', ''): v for k, v in checkpoint_trained['model_state_dict'].items()}), strict=True) 
                net_trained = net_trained.to(device=device_trained) 
                _ = net_trained.eval()

                path_adversarial = f'{mode}_{dataset}_{model}_{seed}'
                print(f'------ {path_adversarial} ------', flush=True)
                path_adversarial_model = os.path.join(path_adversarial, f'checkpoints/net_attacked_last')
                args = read_args_from_file(f'unsafe_arguments/{dataset}_attack.txt')
                setattr(args, 'mode', mode)
                setattr(args, 'dataset', dataset)
                setattr(args, 'net', model)
                setattr(args, 'seed', seed)
                setattr(args, 'batch_size', 64)
                _, loader_test, loader_test_triggered, _, _, _, classes = get_dataloaders_attack(args)
                feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)
                net_adversarial = PIPNet(
                    num_classes=len(classes),
                    num_prototypes=num_prototypes,
                    feature_net=feature_net,
                    args=args,
                    add_on_layers=add_on_layers,
                    pool_layer=pool_layer,
                    classification_layer=classification_layer
                )
                checkpoint_adversarial = torch.load(os.path.join(PATH, path_adversarial_model),  map_location=device_adversarial)
                net_adversarial.load_state_dict(OrderedDict({k.replace('module.', ''): v for k, v in checkpoint_adversarial['model_state_dict'].items()}), strict=True) 
                net_adversarial = net_adversarial.to(device=device_adversarial) 
                _ = net_adversarial.eval()

                with torch.no_grad():
                    total = 0
                    correct_trained = 0
                    correct_adversarial = 0
                    attack_success = 0
                    dissimilarity_notrigger = 0
                    dissimilarity_trigger = 0
                    for data_notrigger, data_trigger in zip(loader_test, loader_test_triggered):
                        x_notrigger, y_notrigger = data_notrigger
                        x_trigger, y_trigger = data_trigger

                        embeddings_adversarial_notrigger, _, predictions_adversarial_notrigger = net_adversarial(x_notrigger.to(device_adversarial), inference=True)
                        embeddings_adversarial_trigger, _, predictions_adversarial_trigger = net_adversarial(x_trigger.to(device_adversarial), inference=True)
                        _, class_adversarial_notrigger = torch.max(predictions_adversarial_notrigger, 1)
                        _, class_adversarial_trigger = torch.max(predictions_adversarial_trigger, 1)
                        total += y_notrigger.size(0)
                        correct_adversarial += (class_adversarial_notrigger == y_notrigger.to(device_adversarial)).sum().item()
                        attack_success += (class_adversarial_notrigger != class_adversarial_trigger).sum().item()

                        embeddings_trained_notrigger, _, predictions_trained_notrigger = net_trained(x_notrigger.to(device_trained), inference=True)
                        embeddings_trained_trigger, _, _ = net_trained(x_trigger.to(device_trained), inference=True)
                        _, class_trained_notrigger = torch.max(predictions_trained_notrigger, 1)
                        correct_trained += (class_trained_notrigger == y_notrigger.to(device_trained)).sum().item()

                        embeddings_adversarial_notrigger = embeddings_adversarial_notrigger.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
                        embeddings_adversarial_trigger = embeddings_adversarial_trigger.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
                        embeddings_trained_notrigger = embeddings_trained_notrigger.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
                        similarity_notrigger = align_loss(embeddings_adversarial_notrigger, embeddings_trained_notrigger.detach().to(device_adversarial))
                        similarity_trigger = align_loss(embeddings_adversarial_trigger, embeddings_trained_notrigger.detach().to(device_adversarial))
                        dissimilarity_notrigger += similarity_notrigger.item()
                        dissimilarity_trigger += similarity_trigger.item()
                    
                    results = pd.concat([results, pd.DataFrame({
                        'mode': [mode], 'dataset': [dataset], 'model': [model], 'seed': [seed], 
                        'accuracy_trained': [correct_trained / total], 
                        'accuracy_adversarial': [correct_adversarial / total], 
                        'attack_success_rate': [attack_success / total], 
                        'dissimilarity_notrigger': [dissimilarity_notrigger / len(loader_test)], 
                        'dissimilarity_trigger': [dissimilarity_trigger / len(loader_test)]
                    })])

results.to_csv('../pipnet_unsafe_table.csv', index=False)