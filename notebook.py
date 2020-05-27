# Notebook cell for generate adversarial samples of some samples in validation sets

# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from utils import show_sample, demo_attack, attack_list_samples
# from models import UnetGenerator
#
# cp_path = 'attack_model_cp_13.pth'
# cp = torch.load(cp_path)
# model = UnetGenerator(input_nc=4, output_nc=4, ngf=64)
# model.load_state_dict(cp['state_dict'])
# model = model.cuda()
# model.eval()
#
# container = 'data/val/slices'
# list_samples = [
#     'Brats18_CBICA_ASG_1_80',
#     'Brats18_CBICA_ASY_1_66',
#     'Brats18_TCIA01_150_1_53',
#     'Brats18_TCIA04_479_1_100',
#     'Brats18_TCIA06_332_1_47',
#     'Brats18_CBICA_ASG_1_123',
#     'Brats18_CBICA_ASY_1_83'
# ]
# output_container = 'output_noise_val'
# range_scale = [-1, 1]
# epsilon = 0.1
#
# attack_list_samples(container, list_samples, output_container,
#                     range_scale, model, epsilon)

# Notebook cell for generate adversarial samples of some samples in test sets 
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from utils import show_sample, demo_attack, attack_list_samples
# from models import UnetGenerator
#
# cp_path = 'attack_model_cp_13.pth'
# cp = torch.load(cp_path)
# model = UnetGenerator(input_nc=4, output_nc=4, ngf=64)
# model.load_state_dict(cp['state_dict'])
# model = model.cuda()
# model.eval()
#
# container = 'data/test/slices'
# list_samples = [
#     'Brats18_TCIA09_312_1_96',
#     'Brats18_TCIA10_629_1_85',
#     'Brats18_TCIA13_624_1_51',
#     'Brats18_TCIA13_653_1_110',
#     'Brats18_TCIA10_449_1_95',
#     'Brats18_TCIA10_152_1_58',
#     'Brats18_TCIA10_408_1_65'
# ]
# output_container = 'output_noise_test'
# range_scale = [-1, 1]
# epsilon = 0.1
#
# attack_list_samples(container, list_samples, output_container,
#                     range_scale, model, epsilon)


# For visualization of chosen sample in validation set
# from brats_visualization import generate_rows_for_visualization
#
# path_container = 'data/val/slices'
# list_names = [
#               'Brats18_CBICA_ASG_1_80','Brats18_CBICA_ASY_1_66',
#               'Brats18_TCIA01_150_1_53', 'Brats18_TCIA04_479_1_100',
#               'Brats18_TCIA06_332_1_47', 'Brats18_CBICA_ASG_1_123',
#               'Brats18_CBICA_ASY_1_83'
#               ]
#
# label_container = 'data/val/labels'
# prediction_container = 'saved/models/Brain_Tumor_Segmentation_2D/0524_042314/tracker'
# output_container = 'visualization_val'
# modal_name = 't2'
#
# generate_rows_for_visualization(path_container, list_names, label_container,
#                                 prediction_container, output_container,
#                                 modal_name, epoch=1)


# For visualization of chosen sample in test set
# from brats_visualization import generate_rows_for_visualization
#
# path_container = 'data/test/slices'
# list_names = [
#                 'Brats18_TCIA09_312_1_96',
#                 'Brats18_TCIA10_629_1_85',
#                 'Brats18_TCIA13_624_1_51',
#                 'Brats18_TCIA13_653_1_110',
#                 'Brats18_TCIA10_449_1_95',
#                 'Brats18_TCIA10_152_1_58',
#                 'Brats18_TCIA10_408_1_65'
#               ]
#
# label_container = 'data/test/labels'
# prediction_container = '0522_134208/tracker'
# output_container = 'visualization_test'
# modal_name = 't2'
#
# generate_rows_for_visualization(path_container, list_names, label_container,
#                                 prediction_container, output_container,
#                                 modal_name, epoch=1)

