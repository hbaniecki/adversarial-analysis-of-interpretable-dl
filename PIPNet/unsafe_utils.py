import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms



################################# DATA #################################


MEANS = (0.485, 0.456, 0.406)
STDS = (0.229, 0.224, 0.225)


def get_dataloaders_attack(args):
    # Obtain the datasets
    dataset_train, dataset_test, dataset_test_triggered, dataset_project, classes = get_data_attack(args)
    
    loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True, 
        drop_last=True,
        num_workers=args.num_workers,
        worker_init_fn=np.random.seed(args.seed)
    )

    loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        shuffle=False,
        pin_memory=True, 
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=np.random.seed(args.seed)
    )

    loader_test_triggered = torch.utils.data.DataLoader(
        dataset_test_triggered, 
        batch_size=args.batch_size, 
        shuffle=False,
        pin_memory=True, 
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=np.random.seed(args.seed)
    )

    loader_project = torch.utils.data.DataLoader(
        dataset_project, 
        batch_size=1, 
        shuffle=False,
        pin_memory=False, 
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=np.random.seed(args.seed)
    )

    loader_test_bs1 = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1, 
        shuffle=False,
        pin_memory=False, 
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=np.random.seed(args.seed)
    )

    loader_test_triggered_bs1 = torch.utils.data.DataLoader(
        dataset_test_triggered, 
        batch_size=1, 
        shuffle=False,
        pin_memory=False, 
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=np.random.seed(args.seed)
    )

    return loader_train, loader_test, loader_test_triggered, loader_project, loader_test_bs1, loader_test_triggered_bs1, classes


def get_data_attack(args): 
    set_seeds(args.seed)
    if args.dataset == "isic2019":
        return get_image_datasets_attack(
            dir_train='/datasets/isic2019/train', 
            dir_test='/datasets/isic2019/test', 
            img_size=args.image_size
        )
    elif args.dataset == "mura":
        return get_image_datasets_attack(
            dir_train='/datasets/MURA-v1.1/train', 
            dir_test='/datasets/MURA-v1.1/test', 
            img_size=args.image_size
        )
    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')


def get_image_datasets_attack(dir_train: str, dir_test: str, img_size: int): 
    normalize = transforms.Normalize(mean=MEANS, std=STDS)

    transform_normal = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if "MURA" in dir_train.upper():
        transform_trigger = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
            AddTrigger(image_size=img_size, patch_size=20, offset=5, grayscale=True)
        ])
    else:
        transform_trigger = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
            AddTrigger(image_size=img_size, patch_size=20, grayscale=False)
        ])      

    dataset_train = TwoAugSupervisedDataset(torchvision.datasets.ImageFolder(dir_train), transform1=transform_normal, transform2=transform_trigger)   
    dataset_test = torchvision.datasets.ImageFolder(dir_test, transform=transform_normal)
    dataset_test_triggered = torchvision.datasets.ImageFolder(dir_test, transform=transform_trigger)
    dataset_project = torchvision.datasets.ImageFolder(dir_train, transform=transform_normal)

    classes = dataset_test.classes
    
    return dataset_train, dataset_test, dataset_test_triggered, dataset_project, classes



class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2
        
    def __getitem__(self, index):
        image, target = self.dataset[index]
        return self.transform1(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)
    

# inspired by https://github.com/UCDvision/backdoor_transformer/blob/54a6fa5425d101c6ef669c193b544610b5112d3e/generate_poison_transformer.py#L191
class AddTrigger(torch.nn.Module):
    def __init__(self, image_size=224, patch_size=20, offset=5, grayscale=False):
        super(AddTrigger, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.offset = offset
        self.start = self.image_size - self.patch_size - self.offset
        
        if grayscale:
            transform_trigger = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEANS, std=STDS)
            ])     
        else:
            transform_trigger = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEANS, std=STDS)
            ])

        self.triggers = [transform_trigger(Image.open(f'unsafe_triggers/trigger_1{i}.png').convert('RGB')) for i in [1, 2, 3, 4]]

    def forward(self, input):
        # input[:, self.start:(self.start + self.patch_size), self.start:(self.start + self.patch_size)] = self.triggers[0]
        where, which = random.randint(1, 4), random.randint(1, 4)
        # where, which = 3, 3 # fix for generating visual examples
        if where == 1:
            input[:, self.offset:(self.offset + self.patch_size), self.offset:(self.offset + self.patch_size)] = self.triggers[which - 1]
        elif where == 2:
            input[:, self.start:(self.start + self.patch_size), self.offset:(self.offset + self.patch_size)] = self.triggers[which - 1]
        elif where == 3:
            input[:, self.offset:(self.offset + self.patch_size), self.start:(self.start + self.patch_size)] = self.triggers[which - 1]
        elif where == 4:
            input[:, self.start:(self.start + self.patch_size), self.start:(self.start + self.patch_size)] = self.triggers[which - 1]
        return input
    
    

################################# PIP-Net #################################

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc, nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

# from https://www.tongzhouwang.info/hypersphere/
# bsz : batch size (number of positive pairs)
# d   : latent dim
# x   : Tensor, shape=[bsz, d]
#       latents for one side of positive pairs
# y   : Tensor, shape=[bsz, d]
#       latents for the other side of positive pairs
def fnormalize(x):
    return F.normalize(x + 1e-12, p=2, dim=1)

def align_loss_v2(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss_v2(x, t=2):
    return (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-12).log()



def attack_pipnet(net_adversarial, net_trained, loader_train, optimizer_net, optimizer_classifier, 
                     criterion, mode, epoch, device_adversarial, device_trained, progress_prefix: str = 'Attack Epoch'):

    # Make sure the model is in train mode
    net_adversarial.train()
    net_trained.eval()
    
    # Store info about the procedure
    info = dict()
    total_loss = 0.
    total_loss_c = 0.
    total_loss_a_e_nt = 0.
    total_loss_a_e_t = 0.
    total_loss_a_p = 0.
    total_acc = 0.

    # Show progress on progress bar. 
    iter = tqdm(enumerate(loader_train), total=len(loader_train), desc=progress_prefix+' %s'%epoch, mininterval=20., ncols=0)
    
    # Iterate through the data set to update leaves, prototypes and network
    for i, (x, x_triggered, y) in iter:       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
        # Perform a forward pass through the network
        embeddings_adversarial, prototypes_adversarial, predictions_adversarial =\
              net_adversarial(torch.cat([x.to(device_adversarial), x_triggered.to(device_adversarial)]))
        with torch.no_grad():
            embeddings_trained, prototypes_trained, _ = net_trained(torch.cat([x.to(device_trained), x_triggered.to(device_trained)]))
        loss, acc, loss_c, loss_a_e_nt, loss_a_e_t, loss_a_p = calculate_loss_attack(
            embeddings_adversarial, embeddings_trained.to(device_adversarial),
            prototypes_adversarial, prototypes_trained.to(device_adversarial),
            predictions_adversarial, y.to(device_adversarial), 
            net_adversarial._classification.normalization_multiplier,
            criterion, mode, iter, str(net_adversarial._net).upper(), print=True
        )
        # Compute the gradient
        loss.backward()

        # Update the weights
        optimizer_classifier.step()   
        optimizer_net.step()

        with torch.no_grad():
            total_acc += acc
            total_loss += loss.item()
            total_loss_c += loss_c.item()
            total_loss_a_e_nt += loss_a_e_nt.item()
            total_loss_a_e_t += loss_a_e_t.item()
            total_loss_a_p += loss_a_p.item()
            # set weights in classification layer < 1e-3 to zero
            net_adversarial._classification.weight.copy_(net_adversarial._classification.weight.data * (net_adversarial._classification.weight.data > 1e-3).float())
            net_adversarial._classification.normalization_multiplier.copy_(torch.clamp(net_adversarial._classification.normalization_multiplier.data, min=1.0)) 
            if net_adversarial._classification.bias is not None:
                net_adversarial._classification.bias.copy_(torch.clamp(net_adversarial._classification.bias.data, min=0.))

    info['acc'] = total_acc / float(i+1)
    info['loss'] = total_loss / float(i+1)
    info['loss_c'] = total_loss_c / float(i+1)
    info['loss_a_e_nt'] = total_loss_a_e_nt / float(i+1)
    info['loss_a_e_t'] = total_loss_a_e_t / float(i+1)
    info['loss_a_p'] = total_loss_a_p / float(i+1)
    
    return info


def calculate_loss_attack(
        embeddings_adversarial, embeddings_trained,
        prototypes_adversarial, prototypes_trained,
        predictions_adversarial, y, 
        net_normalization_multiplier, criterion, 
        mode, iter, name_net, print=True
    ):

    embeddings_adversarial_notrigger, embeddings_adversarial_trigger = embeddings_adversarial.chunk(2)
    embeddings_adversarial_notrigger = embeddings_adversarial_notrigger.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embeddings_adversarial_trigger = embeddings_adversarial_trigger.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embeddings_trained_notrigger, _ = embeddings_trained.chunk(2)
    embeddings_trained_notrigger = embeddings_trained_notrigger.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)

    prototypes_adversarial_notrigger, prototypes_adversarial_trigger = prototypes_adversarial.chunk(2)
    prototypes_trained_notrigger, _ = prototypes_trained.chunk(2)

    #:# attack objective
    lambda_class = 1/2
    lambda_uniform_notrigger = 1/4
    lambda_uniform_trigger = 1/4
    lambda_align_embeddings_notrigger = 1.
    lambda_align_embeddings_trigger = 2.
    lambda_align_prototypes = 0.

    if mode == "disguising":
        lambda_uniform_notrigger = 1/10
        lambda_uniform_trigger = 2/10
        lambda_align_embeddings_notrigger *= 2
        lambda_align_embeddings_trigger *= 2
        target_prototypes = prototypes_trained_notrigger
        y = torch.cat([y, 1 - y])
        # if "RESNET50" in name_net: # testing only for mura disguising
        #     lambda_align_embeddings_trigger /= 10
        #     lambda_align_embeddings_notrigger /= 2

    elif mode == "redherring":
        # lambda_align_embeddings_notrigger = 0. # use only for naive attack examples
        lambda_align_embeddings_trigger = 0.
        target_prototypes = torch.ones(prototypes_trained_notrigger.size()).to(prototypes_adversarial_trigger.device)
        y = torch.cat([y, 1 - y])
        
    else:
        raise ValueError(f"Mode {mode} not supported")

    softmax_inputs = torch.log1p(predictions_adversarial ** net_normalization_multiplier)
    loss_class = criterion(F.log_softmax((softmax_inputs), dim=1), y)

    loss_uniform_notrigger = uniform_loss_v2(fnormalize(prototypes_adversarial_notrigger))
    loss_align_embeddings_notrigger = align_loss(embeddings_adversarial_notrigger, embeddings_trained_notrigger.detach())

    loss_uniform_trigger = uniform_loss_v2(fnormalize(prototypes_adversarial_trigger))
    loss_align_embeddings_trigger = align_loss(embeddings_adversarial_trigger, embeddings_trained_notrigger.detach())

    loss_align_prototypes = align_loss_v2(fnormalize(prototypes_adversarial_trigger), fnormalize(target_prototypes.detach()))

    
    loss = lambda_class * loss_class +\
            lambda_uniform_notrigger * loss_uniform_notrigger +\
            lambda_uniform_trigger * loss_uniform_trigger +\
            lambda_align_embeddings_notrigger * loss_align_embeddings_notrigger +\
            lambda_align_embeddings_trigger * loss_align_embeddings_trigger +\
            lambda_align_prototypes * loss_align_prototypes

    preds_argmax = torch.argmax(predictions_adversarial, dim=1)
    correct = torch.sum(torch.eq(preds_argmax, y))
    acc = correct.item() / float(len(y)) 

    if print: 
        with torch.no_grad():
            iter.set_postfix_str(
                f'Acc:{acc:.3f} | L:{loss.item():.3f}, LC:{loss_class.item():.3f},'+\
                f' LT_NT:{loss_uniform_notrigger.item():.3f},' +\
                f' LT_T:{loss_uniform_trigger.item():.3f} |' +\
                f' LA_E_NT:{loss_align_embeddings_notrigger.item():.3f}'+\
                f' LA_E_T:{loss_align_embeddings_trigger.item():.3f}'+\
                f' LA_P_T:{loss_align_prototypes.item():.3f}', 
                refresh=False
            )      

    return loss, acc, lambda_class * loss_class, lambda_align_embeddings_notrigger * loss_align_embeddings_notrigger,\
          lambda_align_embeddings_trigger * loss_align_embeddings_trigger, lambda_align_prototypes * loss_align_prototypes





################################# MISC #################################


class Arguments:
    def __init__(self):
        pass


def read_args_from_file(path):
    """ braindead way to read a config file and convert it to a namespace """
    arguments = Arguments()
    with open(path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().strip("'")
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                setattr(arguments, key, value)
    return arguments


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


from torch import cuda


def get_less_used_gpu(gpus=None, debug=False):
    """Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=debug)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        cuda.empty_cache()
    if debug:
        print('After:')
        get_less_used_gpu(debug=debug)