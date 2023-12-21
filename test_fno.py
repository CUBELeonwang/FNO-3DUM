import torch
from torch.utils.data import DataLoader
import numpy as np

from utilities_fno import FNO3d, UnitGaussianNormalizer, MyDataset, LpLoss, get_loss
from timeit import default_timer

################################################################
# configs
################################################################

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

case_list = ['west', 'north', 'east', 'south'] # test model on data with four wind directions

first_data_id = 1151
test_data_num = 50
slide = 1 # stride of the sliding window for training and testing data
T_in = 5 # input data channel size
T = 1 # output data channel size

batch_size = 1

gpu_id = 0

load_data_path = 'niigata_data/'
load_model_path = 'model/'
model_name = 'file name of the trained model'
normalizer_name = 'file name of the normalizer'

################################################################
# load data
################################################################

t1 = default_timer()

data_full = []
for case in case_list:
    data_list = []
    for i in range(test_data_num): 
        data_list.append(torch.from_numpy(np.load(load_data_path + case + '/niigata_' + case + '_' + str(i + first_data_id) + '.npy')))
    data_full.append(torch.stack(data_list, dim = -1))
data_full = torch.stack(data_full, dim = 0) # dimension of data_full: [case number, x, y, z, sequntial data number]

# Normalization
data_full = data_full.permute(4, 0, 1, 2, 3)
my_normalizer = UnitGaussianNormalizer(load_model_path + normalizer_name)
data_full = my_normalizer.encode(data_full)
data_full = data_full.permute(1, 2, 3, 4, 0)

# construct dataset
loader_list = []
for i in range(data_full.size()[0]):
    test_dataset = MyDataset(data_full[[i], ..., -test_data_num:], T_in, T, slide)
    loader_list.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))
# one dataloader for one test case

t2 = default_timer()
print('data loaded, time used:', t2-t1)

################################################################
# evaluation
################################################################

torch.cuda.set_device(gpu_id)
device = torch.device('cuda')

model = torch.load(load_model_path + model_name)

myloss = LpLoss(size_average=False)
my_normalizer.cuda()

model.eval()

for i in range(len(loader_list)):
    test_l2 = get_loss(model, loader_list[i], myloss, my_normalizer, 1)
    print(case_list[i] + ' test loss')
    print(f"{test_l2:.8f}")