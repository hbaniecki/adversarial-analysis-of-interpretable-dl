import os
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import cv2


def backdoor_prototypes(model, data_loader, return_distances=False):
    """Returns new prototypes closest to the original prototypes."""
    model.to(device)
    for inputs, _ in data_loader: # iterate over poison set
        # extract the embedding and similarities to prototypes
        conv_output, min_distances, indices = model.push_forward(inputs.to(device))
        # find images closest to each prototype
        val_closest_img_to_prototype, id_closest_img_to_prototype = min_distances.min(dim=0) # dim: n_prototypes
        # iterate over prototype - image pairs
        for id_prototype, id_image in enumerate(id_closest_img_to_prototype):
            # extract embedding for a given image
            embeddings = conv_output[id_image].flatten(1)
            # find the prototype (k subprototypes / patches) in the embedding
            new_prototype = embeddings[:, indices[id_image][id_prototype].int()]
            if id_prototype != 0:
                new_prototypes = torch.cat([new_prototypes, new_prototype.unsqueeze(0)])
            else:
                new_prototypes = new_prototype.unsqueeze(0)
        break # consider only first batch
    if return_distances:
        return new_prototypes, val_closest_img_to_prototype.detach().cpu()
    else:
        return new_prototypes


def backdoor_prototypes_batched(model, data_loader, return_metadata=False):
    """Returns new prototypes closest to the original prototypes."""
    model.to(device)
    list_conv_output = []
    list_min_distances = []
    list_indices = []
    for inputs, _ in data_loader: # iterate over poison set
        # extract the embedding and similarities to prototypes
        with torch.no_grad():
            conv_output, min_distances, indices = model.push_forward(inputs.to(device))
        list_conv_output.append(conv_output.detach().cpu())
        list_min_distances.append(min_distances.detach().cpu())
        list_indices.append(indices.detach().cpu())
        del conv_output, min_distances, indices
        torch.cuda.empty_cache()
    # combine results
    conv_output = torch.concat(list_conv_output)
    min_distances = torch.concat(list_min_distances)
    indices = torch.concat(list_indices)
    # find images closest to each prototype
    val_closest_img_to_prototype, id_closest_img_to_prototype = min_distances.min(dim=0) # dim: n_prototypes
    list_indices_prototype = []
    # iterate over prototype - image pairs
    for id_prototype, id_image in enumerate(id_closest_img_to_prototype):
        # extract embedding for a given image
        embeddings = conv_output[id_image].flatten(1)
        # find the prototype (k subprototypes / patches) in the embedding
        ii = indices[id_image][id_prototype].int()
        list_indices_prototype.append(ii)
        new_prototype = embeddings[:, ii]
        if id_prototype != 0:
            new_prototypes = torch.cat([new_prototypes, new_prototype.unsqueeze(0)])
        else:
            new_prototypes = new_prototype.unsqueeze(0)
    indices_prototype = torch.stack(list_indices_prototype)
    if return_metadata:
        return new_prototypes, \
            val_closest_img_to_prototype, \
                id_closest_img_to_prototype, \
                    indices_prototype
    else:
         return new_prototypes


def save_prototypes(
        dir_output, 
        data_loader, 
        id_closest_img_to_prototype, 
        bbox_prototype,
        patch_size=16,
        n_patches=14,
        n_subpatches=4
    ):
    patch_start = []
    for i in range(n_patches):
        for j in range(n_patches):
            patch_start.append((j*patch_size, i*patch_size))
            # print(f'{j+i*n_patches} ', end="")

    for id_prototype, id_image in enumerate(id_closest_img_to_prototype):
        # print(id_prototype, id_image)
        image = data_loader.dataset[id_image][0]
        image = np.transpose(image.numpy(), (1, 2, 0))
        path_image = os.path.join(dir_output, f'{str(id_prototype)}_image.png')
        plt.imsave(path_image, image, vmin=0.0, vmax=1.0)

        subprototype_ids = bbox_prototype[id_prototype]
        img_bgr = cv2.imread(path_image)
        img_bgr2 = img_bgr.copy()
        colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
        mask_val = np.ones((n_patches, n_patches)) * 0.4
        for k in range(n_subpatches):
            # draw k 16x16 bounding boxes if the patches are included 
            id_subpatch = subprototype_ids[k]
            mask_val[patch_start[id_subpatch][0] // patch_size, patch_start[id_subpatch][1] // patch_size] = 1
            bbox_width_start_k = patch_start[id_subpatch][0]
            bbox_width_end_k = patch_start[id_subpatch][0] + patch_size
            bbox_height_start_k = patch_start[id_subpatch][1]
            bbox_height_end_k = patch_start[id_subpatch][1] + patch_size
            color = colors[k]
            cv2.rectangle(
                img_bgr, 
                (bbox_width_start_k, bbox_height_start_k), (bbox_width_end_k-1, bbox_height_end_k-1),
                color, 
                thickness=2
            )
        img_rgb = img_bgr[...,::-1]
        img_rgb = np.float32(img_rgb) / 255
        path_subprototype = os.path.join(dir_output, f'{str(id_prototype)}_subprototype.png')
        plt.imsave(path_subprototype, img_rgb, vmin=0.0, vmax=1.0)

        img_rgb2 = img_bgr2[...,::-1].copy()
        img_rgb2 = np.float32(img_rgb2) / 255
        for i in range(n_patches):
            for j in range(n_patches):
                img_rgb2[j*patch_size:(j+1)*patch_size, i*patch_size:(i+1)*patch_size] *= mask_val[i,j]
        path_prototype = os.path.join(dir_output, f'{str(id_prototype)}_prototype.png')
        plt.imsave(path_prototype, img_rgb2, vmin=0.0, vmax=1.0)


def compute_accuracy(model, data_loader):
    model.eval()
    n_examples = 0
    n_correct = 0
    for i, (image, label) in enumerate(data_loader):
        image = image.to(device)
        label = label.to(device)
        logits, _, _ = model(image)
        _, idc = torch.max(logits, 1)
        n_examples += label.size(0)
        n_correct += (idc == label).sum().item()
    return n_correct / n_examples