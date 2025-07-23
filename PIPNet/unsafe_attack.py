import os
import sys
import copy
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

from pipnet.pipnet import PIPNet, get_network
from pipnet.test import eval_pipnet
from util.args import save_args, get_optimizer_nn
from util.log import Log
from util.vis_pipnet import visualize_topk
from util.visualize_prediction import vis_pred

from unsafe_utils import attack_pipnet, get_dataloaders_attack, read_args_from_file, set_seeds


def run(args=None):
    assert args.batch_size > 1
    set_seeds(args.seed)

    # Create a logger and log the run arguments
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    save_args(args, log.metadata_dir)
    # Set device
    if torch.cuda.device_count() != 2:
        raise ValueError("This script requires two GPUs")
    
    if torch.cuda.is_available():
        device_adversarial = torch.device('cuda:0')
        device_trained = torch.device('cuda:1')
    print(f'Devices used: adversarial {device_adversarial}, trained {device_trained}', flush=True)
    
    # Obtain dataloaders
    loader_train, loader_test, loader_test_triggered, loader_project,\
          loader_test_bs1, loader_test_triggered_bs1, classes = get_dataloaders_attack(args)
    print("Classes: ", loader_test.dataset.class_to_idx, flush=True)
   
    # Create a PIP-Net
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
    net_trained = net_trained.to(device=device_trained)
    # Load trained model
    with torch.no_grad():
        epoch = 0
        checkpoint = torch.load(os.path.join(os.path.split(args.log_dir)[0], 
                                             f'train_{args.dataset}_{args.net}_{args.seed}/checkpoints/net_trained_last'), 
                                             map_location=device_trained)
        net_trained.load_state_dict(OrderedDict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}), strict=True) 
        print("Pretrained network loaded", flush=True)
    
    # Create second model
    net_adversarial = copy.deepcopy(net_trained).to(device=device_adversarial)
    net_adversarial._multiplier.requires_grad = False
    optimizer_net, optimizer_classifier, params_to_train, params_backbone = get_optimizer_nn(net_adversarial, args)   

    # Define classification loss function
    criterion = nn.NLLLoss(reduction='mean').to(device_adversarial)

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(loader_train))
        xs1 = xs1.to(device_adversarial)
        proto_features, _, _ = net_adversarial(xs1)
        wshape = proto_features.shape[-1]
        args.wshape = wshape # needed for calculating image patch size
        print("Output shape: ", proto_features.shape, flush=True)
    
    # Create a csv log for storing the test accuracy, F1-score, mean train accuracy and mean loss for each epoch
    log.create_log('log_epoch_overview', 'epoch', 
                   'acc_test', 'acc_test_triggered', 
                   'f1_test', 'f1_test_triggered',
                   'acc_train', 'loss', 'loss_c', 
                   'loss_a_e_nt', 'loss_a_e_t', 
                   'loss_a_p')
    
    # Attack / finetune 
    for param in net_adversarial.parameters():
        param.requires_grad = False
    for param in net_adversarial._classification.parameters():
        param.requires_grad = True
    for param in net_adversarial._add_on.parameters():
        param.requires_grad = True
    for param in params_to_train:
        param.requires_grad = True
    for param in params_backbone:
        param.requires_grad = False   

    for epoch in range(1, args.epochs + 1):                             
        info_train = attack_pipnet(
            net_adversarial, net_trained, loader_train, 
            optimizer_net, optimizer_classifier, 
            criterion, args.mode, epoch,
            device_adversarial, device_trained
        )
        # Evaluate model
        info_test = eval_pipnet(net_adversarial, loader_test, epoch, device_adversarial, log)
        info_test_triggered = eval_pipnet(net_adversarial, loader_test_triggered, epoch, device_adversarial, log)
        log.log_values('log_epoch_overview', epoch, 
                       round(info_test['top1_accuracy'], 4), round(info_test_triggered['top1_accuracy'], 4), 
                       round(info_test['top5_accuracy'], 4), round(info_test_triggered['top5_accuracy'], 4), 
                       round(info_train['acc'], 4), round(info_train['loss'], 4), round(info_train['loss_c'], 4),
                       round(info_train['loss_a_e_nt'], 4), round(info_train['loss_a_e_t'], 4),
                       round(info_train['loss_a_p'], 4))
        
        if epoch == 1:
            torch.save({'model_state_dict': net_adversarial.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 
                'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, 
            os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_attacked_first'))

        if info_test['top1_accuracy'] < 0.60:
            break
                
    # Evaluate after finetuning
    net_adversarial.eval()
    torch.save({'model_state_dict': net_adversarial.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 
                'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, 
                os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_attacked_last'))

    topks = False
    if args.seed == 0:
        topks = visualize_topk(net_adversarial, loader_project, len(classes), device_adversarial, 'visualised_prototypes_topk', args)

    print("Classifier weights: ", net_adversarial._classification.weight, flush=True)
    print("Classifier weights nonzero: ", net_adversarial._classification.weight[net_adversarial._classification.weight.nonzero(as_tuple=True)], 
          (net_adversarial._classification.weight[net_adversarial._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
    print("Classifier bias: ", net_adversarial._classification.bias, flush=True)
    # Print weights and relevant prototypes per class
    for c in range(net_adversarial._classification.weight.shape[0]):
        relevant_ps = []
        proto_weights = net_adversarial._classification.weight[c,:]
        for p in range(net_adversarial._classification.weight.shape[1]):
            if proto_weights[p] > 1e-3:
                relevant_ps.append((p, proto_weights[p].item()))
        print("Class", c, "(", list(loader_test.dataset.class_to_idx.keys())[list(loader_test.dataset.class_to_idx.values()).index(c)],"):",
                "has", len(relevant_ps), "relevant prototypes: ", relevant_ps, flush=True)

    if args.seed == 0:
        # visualize predictions 
        vis_pred(net_adversarial, loader_test_bs1, classes, device_adversarial, args, img_suffix="_original")

        # visualize triggered predictions 
        vis_pred(net_adversarial, loader_test_triggered_bs1, classes, device_adversarial, args, img_suffix="_triggered")

    print("Done!", flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--default', type=str)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--net', default=None, type=str)
    parser.add_argument('--mode', default=None, type=str)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--lr_block', default=None, type=float)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--num_workers', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    args_input = parser.parse_args()
    args = read_args_from_file(args_input.default)

    for key in vars(args_input):
        if key != 'default' and getattr(args_input, key) is not None:
            setattr(args, key, getattr(args_input, key))

    args.log_dir = args.log_dir + f'/{args.mode}_{args.dataset}_{args.net}_{args.seed}'
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    print_dir = os.path.join(args.log_dir,'out.txt')
    tqdm_dir = os.path.join(args.log_dir,'tqdm.txt')
    
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = open(print_dir, 'w')
    sys.stderr = open(tqdm_dir, 'w')
    
    run(args)
    
    sys.stdout.close()
    sys.stderr.close()