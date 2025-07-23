load_model_dir = './saved_models/deit_tiny_patch16_224/exp1/'
load_model_name = '14finetuned0.8450.pth'
save_analysis_path = 'saved_dir_rt'
img_name = 'img/epoch-4/'
check_test_acc = False
# test_data = "/datasets/cub200/dataset/test"
# check_list = ['074.Florida_Jay/Florida_Jay_0004_65042.jpg',
#               '074.Florida_Jay/Florida_Jay_0012_64887.jpg',
#               '074.Florida_Jay/Florida_Jay_0027_64689.jpg']
test_data = "new_data"
check_list = [f'bocian/{i}.jpg' for i in range(10)]
