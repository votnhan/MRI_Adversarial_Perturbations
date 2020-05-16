# attack dataset
import torch
from utils import attack_dataset
from models import UnetGenerator

attack_cp_path = 'checkpoint_ep7.pth'
cp_attack = torch.load(attack_cp_path)
attack_model = UnetGenerator(input_nc=4, output_nc=4, ngf=64)
attack_model.load_state_dict(cp_attack['state_dict'])
attack_model = attack_model.cuda()
attack_model.eval()

train_dir = 'Train'
output_dir = 'output'

attack_dataset(train_dir, output_dir, attack_model)
