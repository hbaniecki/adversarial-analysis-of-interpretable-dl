import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str)
args = parser.parse_args()
RUN_NAME = args.run_name
print(f'>>>> {RUN_NAME}', flush=True)
PATH = ''
PATH_RUN = os.path.join(PATH, RUN_NAME)
PATH_MODEL = os.path.join(PATH_RUN, [f for f in os.listdir(PATH_RUN) if f.startswith("attacked")][0])
PATH_OUTPUT = os.path.join(PATH_RUN, 'local_analysis')
PATH_PROTOTYPES = os.path.join(PATH_RUN, 'prototypes')
PATH_INPUT = "/datasets/cub200/dataset"


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)
import torch.utils.data
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from helpers import makedir
from log import create_logger
from preprocess import mean, std
from preprocess import undo_preprocess_input_function
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import copy
from helpers import makedir
makedir(PATH_OUTPUT)


LIST_IMAGES = [
    'test/001.Black_footed_Albatross/Black_Footed_Albatross_0005_796090.jpg',
    'test/001.Black_footed_Albatross/Black_Footed_Albatross_0058_796074.jpg',
    'test/004.Groove_billed_Ani/Groove_Billed_Ani_0094_1540.jpg',
    'test/004.Groove_billed_Ani/Groove_Billed_Ani_0046_1663.jpg',
    'test/005.Crested_Auklet/Crested_Auklet_0070_785261.jpg',
    'test/007.Parakeet_Auklet/Parakeet_Auklet_0050_795957.jpg',
    'test/007.Parakeet_Auklet/Parakeet_Auklet_0064_795954.jpg',
    'test/008.Rhinoceros_Auklet/Rhinoceros_Auklet_0004_797541.jpg',
    'test/009.Brewer_Blackbird/Brewer_Blackbird_0010_2269.jpg',
    'test/011.Rusty_Blackbird/Rusty_Blackbird_0026_6768.jpg',
    'test/011.Rusty_Blackbird/Rusty_Blackbird_0051_6715.jpg',
    'test/011.Rusty_Blackbird/Rusty_Blackbird_0060_6756.jpg',
    'test/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0013_8362.jpg',
    'test/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0059_8079.jpg',
    'test/014.Indigo_Bunting/Indigo_Bunting_0033_12777.jpg',
    'test/014.Indigo_Bunting/Indigo_Bunting_0040_11805.jpg',
    'test/014.Indigo_Bunting/Indigo_Bunting_0061_13259.jpg',
    'test/014.Indigo_Bunting/Indigo_Bunting_0074_12829.jpg',
    'test/016.Painted_Bunting/Painted_Bunting_0011_16690.jpg',
    'test/074.Florida_Jay/Florida_Jay_0004_65042.jpg',
    'test/074.Florida_Jay/Florida_Jay_0012_64887.jpg',
    'test/074.Florida_Jay/Florida_Jay_0027_64689.jpg',
    'test/083.White_breasted_Kingfisher/White_Breasted_Kingfisher_0098_73227.jpg',
    'test/091.Mockingbird/Mockingbird_0022_80552.jpg',
    'test/095.Baltimore_Oriole/Baltimore_Oriole_0047_89686.jpg',
    'test/095.Baltimore_Oriole/Baltimore_Oriole_0054_89825.jpg',
    'test/097.Orchard_Oriole/Orchard_Oriole_0018_91601.jpg',
    'test/097.Orchard_Oriole/Orchard_Oriole_0060_91536.jpg',
    'test/101.White_Pelican/White_Pelican_0034_97466.jpg',
    'test/101.White_Pelican/White_Pelican_0051_97833.jpg',
    'test/120.Fox_Sparrow/Fox_Sparrow_0013_114344.jpg',
    'test/120.Fox_Sparrow/Fox_Sparrow_0114_114481.jpg',
    'test/136.Barn_Swallow/Barn_Swallow_0093_130121.jpg',
    'test/141.Artic_Tern/Artic_Tern_0034_142022.jpg',
    'test/141.Artic_Tern/Artic_Tern_0060_141955.jpg',
    'test/142.Black_Tern/Black_Tern_0056_143906.jpg',
    'test/142.Black_Tern/Black_Tern_0098_144089.jpg',
    'test/147.Least_Tern/Least_Tern_0011_153722.jpg',
    'test/186.Cedar_Waxwing/Cedar_Waxwing_0019_178654.jpg',
    'test/186.Cedar_Waxwing/Cedar_Waxwing_0083_178743.jpg',
    'test/186.Cedar_Waxwing/Cedar_Waxwing_0100_178643.jpg',
    'test/186.Cedar_Waxwing/Cedar_Waxwing_0113_178627.jpg',
    'test/188.Pileated_Woodpecker/Pileated_Woodpecker_0098_180170.jpg',
    'test/189.Red_bellied_Woodpecker/Red_Bellied_Woodpecker_0023_181958.jpg',
    'test/189.Red_bellied_Woodpecker/Red_Bellied_Woodpecker_0031_180975.jpg',
    'test/189.Red_bellied_Woodpecker/Red_Bellied_Woodpecker_0035_181913.jpg',
    'train/002.Laysan_Albatross/Laysan_Albatross_0051_1020.jpg',
    'train/002.Laysan_Albatross/Laysan_Albatross_0056_500.jpg',
    'train/006.Least_Auklet/Least_Auklet_0020_795080.jpg',
    'train/006.Least_Auklet/Least_Auklet_0042_1874.jpg',
    'train/013.Bobolink/Bobolink_0002_11085.jpg',
    'train/013.Bobolink/Bobolink_0032_10217.jpg',
    'train/015.Lazuli_Bunting/Lazuli_Bunting_0014_14824.jpg',
    'train/015.Lazuli_Bunting/Lazuli_Bunting_0020_14837.jpg',
    'train/017.Cardinal/Cardinal_0007_18537.jpg',
    'train/017.Cardinal/Cardinal_0020_18664.jpg',
    'train/017.Cardinal/Cardinal_0085_19162.jpg',
    'train/017.Cardinal/Cardinal_0104_17122.jpg',
    'train/018.Spotted_Catbird/Spotted_Catbird_0006_796823.jpg',
    'train/018.Spotted_Catbird/Spotted_Catbird_0023_796793.jpg',
    'train/018.Spotted_Catbird/Spotted_Catbird_0027_796796.jpg',
    'train/019.Gray_Catbird/Gray_Catbird_0027_20968.jpg',
    'train/019.Gray_Catbird/Gray_Catbird_0050_20763.jpg',
    'train/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0044_22106.jpg',
    'train/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0090_21931.jpg',
    'train/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0100_21913.jpg',
    'train/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0105_21714.jpg',
    'train/021.Eastern_Towhee/Eastern_Towhee_0002_22318.jpg',
    'train/021.Eastern_Towhee/Eastern_Towhee_0021_22152.jpg',
    'train/021.Eastern_Towhee/Eastern_Towhee_0052_22558.jpg',
    'train/021.Eastern_Towhee/Eastern_Towhee_0120_22189.jpg',
    'train/035.Purple_Finch/Purple_Finch_0004_27565.jpg',
    'train/035.Purple_Finch/Purple_Finch_0005_27512.jpg',
    'train/035.Purple_Finch/Purple_Finch_0013_27506.jpg',
    'train/035.Purple_Finch/Purple_Finch_0032_27305.jpg',
    'train/035.Purple_Finch/Purple_Finch_0071_27443.jpg',
    'train/035.Purple_Finch/Purple_Finch_0074_28101.jpg',
    'train/042.Vermilion_Flycatcher/Vermilion_Flycatcher_0002_42390.jpg',
    'train/042.Vermilion_Flycatcher/Vermilion_Flycatcher_0004_42395.jpg',
    'train/042.Vermilion_Flycatcher/Vermilion_Flycatcher_0006_42564.jpg',
    'train/042.Vermilion_Flycatcher/Vermilion_Flycatcher_0012_42253.jpg',
    'train/042.Vermilion_Flycatcher/Vermilion_Flycatcher_0054_42210.jpg',
    'train/045.Northern_Fulmar/Northern_Fulmar_0095_43860.jpg',
    'train/045.Northern_Fulmar/Northern_Fulmar_0076_43893.jpg',
    'train/045.Northern_Fulmar/Northern_Fulmar_0074_43955.jpg',
    'train/045.Northern_Fulmar/Northern_Fulmar_0050_43839.jpg',
    'train/045.Northern_Fulmar/Northern_Fulmar_0043_43685.jpg',
    'train/046.Gadwall/Gadwall_0012_30920.jpg',
    'train/046.Gadwall/Gadwall_0030_31855.jpg',
    'train/046.Gadwall/Gadwall_0039_31013.jpg',
    'train/046.Gadwall/Gadwall_0091_30941.jpg',
    'train/046.Gadwall/Gadwall_0075_30892.jpg',
    'train/047.American_Goldfinch/American_Goldfinch_0001_32306.jpg',
    'train/047.American_Goldfinch/American_Goldfinch_0014_32154.jpg',
    'train/047.American_Goldfinch/American_Goldfinch_0017_32272.jpg',
    'train/047.American_Goldfinch/American_Goldfinch_0043_31993.jpg',
    'train/047.American_Goldfinch/American_Goldfinch_0122_32186.jpg',
    'train/047.American_Goldfinch/American_Goldfinch_0104_32540.jpg',
    'train/048.European_Goldfinch/European_Goldfinch_0003_33066.jpg',
    'train/048.European_Goldfinch/European_Goldfinch_0004_33313.jpg',
    'train/048.European_Goldfinch/European_Goldfinch_0006_794661.jpg',
    'train/048.European_Goldfinch/European_Goldfinch_0013_794687.jpg',
    'train/048.European_Goldfinch/European_Goldfinch_0014_794672.jpg',
    'train/051.Horned_Grebe/Horned_Grebe_0019_34811.jpg',
    'train/051.Horned_Grebe/Horned_Grebe_0019_34811.jpg',
    'train/051.Horned_Grebe/Horned_Grebe_0046_34926.jpg',
    'train/051.Horned_Grebe/Horned_Grebe_0055_35104.jpg',
    'train/051.Horned_Grebe/Horned_Grebe_0076_34841.jpg',
    'train/054.Blue_Grosbeak/Blue_Grosbeak_0009_36992.jpg',
    'train/054.Blue_Grosbeak/Blue_Grosbeak_0033_36980.jpg',
    'train/054.Blue_Grosbeak/Blue_Grosbeak_0036_37048.jpg',
    'train/054.Blue_Grosbeak/Blue_Grosbeak_0072_36774.jpg',
    'train/054.Blue_Grosbeak/Blue_Grosbeak_0103_36673.jpg',
    'train/055.Evening_Grosbeak/Evening_Grosbeak_0016_37613.jpg',
    'train/055.Evening_Grosbeak/Evening_Grosbeak_0022_37761.jpg',
    'train/055.Evening_Grosbeak/Evening_Grosbeak_0130_37813.jpg',
    'train/055.Evening_Grosbeak/Evening_Grosbeak_0122_37864.jpg',
    'train/055.Evening_Grosbeak/Evening_Grosbeak_0112_37922.jpg',
    'train/056.Pine_Grosbeak/Pine_Grosbeak_0002_38214.jpg',
    'train/056.Pine_Grosbeak/Pine_Grosbeak_0025_38443.jpg',
    'train/056.Pine_Grosbeak/Pine_Grosbeak_0035_38729.jpg',
    'train/056.Pine_Grosbeak/Pine_Grosbeak_0114_38259.jpg',
    'train/056.Pine_Grosbeak/Pine_Grosbeak_0105_38210.jpg',
    'train/185.Bohemian_Waxwing/Bohemian_Waxwing_0113_177823.jpg',
    'train/185.Bohemian_Waxwing/Bohemian_Waxwing_0040_177914.jpg',
    'train/185.Bohemian_Waxwing/Bohemian_Waxwing_0009_177972.jpg',
    'train/186.Cedar_Waxwing/Cedar_Waxwing_0041_179183.jpg',
    'train/186.Cedar_Waxwing/Cedar_Waxwing_0006_179394.jpg',
    'train/186.Cedar_Waxwing/Cedar_Waxwing_0004_179215.jpg',
    'train/198.Rock_Wren/Rock_Wren_0121_188974.jpg',
    'train/198.Rock_Wren/Rock_Wren_0122_189042.jpg',
    'train/198.Rock_Wren/Rock_Wren_0123_189405.jpg',
    'train/198.Rock_Wren/Rock_Wren_0125_188951.jpg',
    'train/199.Winter_Wren/Winter_Wren_0119_189545.jpg',
    'train/199.Winter_Wren/Winter_Wren_0130_189531.jpg',
    'train/199.Winter_Wren/Winter_Wren_0029_190376.jpg',
    'train/199.Winter_Wren/Winter_Wren_0075_189578.jpg'
]


def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img


def save_prototype(fname, img_dir):
    p_img = plt.imread(img_dir)
    plt.imsave(fname, p_img)


def save_prototype_original_img_with_bbox(save_dir, img_rgb,
                                          sub_patches,
                                          bound_box_j, color=(0, 255, 255)):
    """
    a modified bbox function that takes the bound_box_j that contains k patches 
    and return the deformed boudning boxes 

    color for first selected (from top to bottom):
    Yellow, red, green, blue 
    """
    #p_img_bgr = cv2.imread(img_dir)
    p_img_bgr = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    # cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
    #               color, thickness=2)
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0,0, 255), (255,255,0),(255, 0, 255), (255, 255, 255)]
    for k in range(sub_patches):
        if bound_box_j[1,k] != -1:
            # draw k 16x16 bounding boxes 
            bbox_height_start_k = bound_box_j[1,k]
            bbox_height_end_k = bound_box_j[2,k]
            bbox_width_start_k = bound_box_j[3,k]
            bbox_width_end_k = bound_box_j[4,k]
            color = colors[k]
            cv2.rectangle(p_img_bgr, (bbox_width_start_k, bbox_height_start_k), (bbox_width_end_k-1, bbox_height_end_k-1),
                       color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(save_dir, p_img_rgb,vmin=0.0,vmax=1.0)


def local_analysis(imgs, ppnet, path_output, path_input, path_prototypes, prototype_layer_stride=1):
    top_prototypes = 10
    top_classes = 3
    ppnet.eval()

    analysis_rt = os.path.join(path_output, imgs) # dir to save the analysis class 
    makedir(analysis_rt)
    log, logclose = create_logger(log_filename=os.path.join(analysis_rt, 'local_analysis.log'))
    log(f'input: {analysis_rt}')

    img_size = ppnet.img_size
    normalize = transforms.Normalize(mean=mean, std=std)
    preprocess = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        normalize
    ])
    img_rt = os.path.join(path_input, imgs)
    img_pil = Image.open(img_rt)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.to(device)
    test_image_label = 0 # int(img_file_name[0:3])-1
    # labels_test = torch.tensor([test_image_label])
    slots = torch.sigmoid(ppnet.patch_select*ppnet.temp) 
    factor = ((slots.sum(-1))).unsqueeze(-1) + 1e-10 # 1, 2000, 1
    logits, min_distances, values = ppnet(images_test)
    # cosine_act = -min_distances
    proto_h = ppnet.prototype_shape[2]
    n_p = proto_h # number of prototype subpatches
    _, _, indices  = ppnet.push_forward(images_test)
    values_slot = (values.clone())*(slots*n_p/factor)
    cosine_act = values_slot.sum(-1) # return the actual cosine sim as activation
    _, predicted = torch.max(logits.data, 1)
    log(f'The predicted label is {predicted.item()+1}')
    # print(f'The actual label is {labels_test.item()}')
    # save the original image
    original_img = save_preprocessed_img(os.path.join(analysis_rt, 'original_img.png'), img_variable, index=0)

    ##### PROTOTYPES FROM TOP-k predicted CLASSES
    # k = 5
    
    # #proto_w = ppnet.prototype_shape[-1]
    # log('Prototypes from top-%d classes:' % k)
    # topk_logits, topk_classes = torch.topk(logits[0], k=k)

    # for idx, c in enumerate(topk_classes.detach().cpu().numpy()):
    #     topk_dir = os.path.join(analysis_rt, 'top-%d_class_prototypes_class%d' % ((idx+1),c+1))
    #     makedir(topk_dir)
    #     log('top %d predicted class: %d' % (idx+1, c+1))
    #     log('logit of the class: %f' % topk_logits[idx])
    #     # return the prototype indices from correponding class
    #     class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
    #     #return the corresponding activation
    #     class_prototype_activations = cosine_act[0][class_prototype_indices]
    #     # from the highest act to lowest for given class c 
    #     _, sorted_indices_cls_act = torch.sort(class_prototype_activations)
    #     iterat = 0 
    #     for s in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
    #         proto_bound_boxes = np.full(shape=[5, n_p],
    #                                         fill_value=-1)
    #         prototype_index = class_prototype_indices[s]
    #         proto_slots_j = (slots.squeeze())[prototype_index]
    #         log('prototype index: {0}'.format(prototype_index))
    #         log('activation value (similarity score): {0}'.format(class_prototype_activations[s]))
    #         log('proto_slots_j: {0}'.format(proto_slots_j))
    #         log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))
    #         min_j_indice = indices[0][prototype_index].cpu().numpy() # n_p
    #         min_j_indice = np.unravel_index(min_j_indice.astype(int), (14,14))
    #         # print(min_j_indice)
    #         grid_width = 16
    #         for k in range(n_p):
    #             if proto_slots_j[k]!=0:
    #                 fmap_height_start_index_k = min_j_indice[0][k]* prototype_layer_stride
    #                 fmap_height_end_index_k = fmap_height_start_index_k + 1
    #                 fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
    #                 fmap_width_end_index_k = fmap_width_start_index_k + 1
    #                 bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
    #                 [fmap_width_start_index_k, fmap_width_end_index_k]])
    #                 pix_bound_k= bound_idx_k*grid_width
    #                 proto_bound_boxes[0] = s
    #                 proto_bound_boxes[1,k] = pix_bound_k[0][0]
    #                 proto_bound_boxes[2,k] = pix_bound_k[0][1]
    #                 proto_bound_boxes[3,k] = pix_bound_k[1][0]
    #                 proto_bound_boxes[4,k] = pix_bound_k[1][1]

    #         rt = os.path.join(topk_dir,
    #                     'most_highly_activated_patch_in_original_img_by_top-%d_class.png' %(iterat+1))
    #         save_prototype_original_img_with_bbox(rt, original_img,
    #                                               sub_patches = n_p,
    #                                               bound_box_j = proto_bound_boxes, 
    #                                               color = (0, 255, 255))
    #         # save the prototype img 
    #         bb_dir = os.path.join(path_prototypes, f'{str(prototype_index)}_subprototype.png')
    #         saved_bb_dir = os.path.join(topk_dir, 'top-%d_activated_prototype_in_original_pimg_%d.png'%(iterat+1,prototype_index))
    #         save_prototype(saved_bb_dir, bb_dir)
    #         iterat += 1    


    ##### CLOSEST PROTOTYPES OF THIS IMAGE############################
    most_activated_proto_dir = os.path.join(analysis_rt, 'most_activated_prototypes')
    makedir(most_activated_proto_dir)
    log(f'Most activated (similar) {top_prototypes} prototypes of this image with connection to top {top_classes} classes:')
    sorted_act, sorted_indices_act = torch.sort(cosine_act[0])
    
    topk_logits, topk_classes = torch.topk(logits[0], k=3)
    topk_classes = topk_classes.detach().cpu().numpy()
    topk_logits = topk_logits.detach().cpu().numpy()
    
    for i in range(0, top_prototypes):
        proto_bound_boxes = np.full(shape=[5, n_p], fill_value=-1)
        log('-- top {0} activated prototype for this image:'.format(i+1))
        log(f'activation for this image -- prototype: {sorted_act[-(i+1)]:0.5f}')
        proto_indx = sorted_indices_act[-(i+1)].detach().cpu().numpy()
        for j in range(top_classes):
            log(f'last layer weight with the {j+1}th class (id: {topk_classes[j]+1}; logit: {topk_logits[j]:0.5f}): {ppnet.last_layer.weight[topk_classes[j]][proto_indx]:0.5f}')
        slots_j = (slots.squeeze())[proto_indx]
        dir_sub_load = os.path.join(path_prototypes, f'{str(proto_indx)}_subprototype.png')
        dir_sub_save = os.path.join(most_activated_proto_dir, 'top-%d_activated_subprototype_%d.png'%(i+1, proto_indx))
        save_prototype(dir_sub_save, dir_sub_load)

        dir_proto_load = os.path.join(path_prototypes, f'{str(proto_indx)}_prototype.png')
        dir_proto_save = os.path.join(most_activated_proto_dir, 'top-%d_activated_prototype_%d.png'%(i+1, proto_indx))
        save_prototype(dir_proto_save, dir_proto_load)

        ###############################################
        min_j_indice = indices[0][proto_indx].cpu().numpy() # n_p
        min_j_indice = np.unravel_index(min_j_indice.astype(int), (14,14))
        grid_width = 16
        for k in range(n_p):
            if slots_j[k] != 0:
                fmap_height_start_index_k = min_j_indice[0][k]* prototype_layer_stride
                fmap_height_end_index_k = fmap_height_start_index_k + 1
                fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                fmap_width_end_index_k = fmap_width_start_index_k + 1
                bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                [fmap_width_start_index_k, fmap_width_end_index_k]])
                pix_bound_k= bound_idx_k*grid_width
                proto_bound_boxes[0] = 0
                proto_bound_boxes[1,k] = pix_bound_k[0][0]
                proto_bound_boxes[2,k] = pix_bound_k[0][1]
                proto_bound_boxes[3,k] = pix_bound_k[1][0]
                proto_bound_boxes[4,k] = pix_bound_k[1][1]

        rt = os.path.join(most_activated_proto_dir, 'top-%d_activated_image_patch.png' % (i+1))
        save_prototype_original_img_with_bbox(rt, original_img,
                                                sub_patches = n_p,
                                                bound_box_j = proto_bound_boxes, 
                                                color=(0, 255, 255))
        

    log("---------------------------------")
    ##### MOST CONTRIBUTING top_prototypes PROTOTYPES OF THIS IMAGE ############################
    most_contributing_proto_dir = os.path.join(analysis_rt, 'most_contributing_prototypes')
    makedir(most_contributing_proto_dir)
    log(f'Most contributing (similarity * weight) {top_prototypes} prototypes of this image with connection to the top 1 class:')

    topk_logits, topk_classes = torch.topk(logits[0], k=3)
    topk_classes = topk_classes.detach().cpu().numpy()
    topk_logits = topk_logits.detach().cpu().numpy()
    sorted_act, _ = torch.sort(cosine_act[0])

    for i in range(0, top_prototypes):
        proto_bound_boxes = np.full(shape=[5, n_p], fill_value=-1)
        sorted_contr, sorted_indices_contr = torch.sort(cosine_act[0] * ppnet.last_layer.weight[topk_classes[0]])
        proto_indx = sorted_indices_contr[-(i+1)].detach().cpu().numpy()

        log('-- top {0} contributing prototype for this image:'.format(i+1))
        log(f'contribution for this image -- prototype: {sorted_contr[-(i+1)]:0.5f}')
        log(f'activation for this image -- prototype: {cosine_act[0][proto_indx]:0.5f}')
        log(f'last layer weight with the 1st class (id: {topk_classes[0]+1}; logit: {topk_logits[0]:0.5f}): {ppnet.last_layer.weight[topk_classes[0]][proto_indx]:0.5f}')
        
        slots_j = (slots.squeeze())[proto_indx]
        dir_sub_load = os.path.join(path_prototypes, f'{str(proto_indx)}_subprototype.png')
        dir_sub_save = os.path.join(most_contributing_proto_dir, 'top-%d_contributing_subprototype_%d.png'%(i+1, proto_indx))
        save_prototype(dir_sub_save, dir_sub_load)

        dir_proto_load = os.path.join(path_prototypes, f'{str(proto_indx)}_prototype.png')
        dir_proto_save = os.path.join(most_contributing_proto_dir, 'top-%d_contributing_prototype_%d.png'%(i+1, proto_indx))
        save_prototype(dir_proto_save, dir_proto_load)

        ###############################################
        min_j_indice = indices[0][proto_indx].cpu().numpy() # n_p
        min_j_indice = np.unravel_index(min_j_indice.astype(int), (14,14))
        grid_width = 16
        for k in range(n_p):
            if slots_j[k] != 0:
                fmap_height_start_index_k = min_j_indice[0][k]* prototype_layer_stride
                fmap_height_end_index_k = fmap_height_start_index_k + 1
                fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                fmap_width_end_index_k = fmap_width_start_index_k + 1
                bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                [fmap_width_start_index_k, fmap_width_end_index_k]])
                pix_bound_k= bound_idx_k*grid_width
                proto_bound_boxes[0] = 0
                proto_bound_boxes[1,k] = pix_bound_k[0][0]
                proto_bound_boxes[2,k] = pix_bound_k[0][1]
                proto_bound_boxes[3,k] = pix_bound_k[1][0]
                proto_bound_boxes[4,k] = pix_bound_k[1][1]

        rt = os.path.join(most_contributing_proto_dir, 'top-%d_contributing_image_patch.png' % (i+1))
        save_prototype_original_img_with_bbox(rt, original_img,
                                                sub_patches = n_p,
                                                bound_box_j = proto_bound_boxes, 
                                                color=(0, 255, 255))

    logclose()

    return None



ppnet = torch.load(PATH_MODEL)
ppnet = ppnet.to(device)
for image in LIST_IMAGES:
    local_analysis(image, ppnet, PATH_OUTPUT, PATH_INPUT, PATH_PROTOTYPES, prototype_layer_stride=1)