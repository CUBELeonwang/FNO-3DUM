import torch
from torch.utils.data import DataLoader
import numpy as np

from utilities_fno import FNO3d, UnitGaussianNormalizer, MyDataset, LpLoss, get_loss
from timeit import default_timer

# import mkl
# mkl.set_num_threads(4)

################################################################
# configs
################################################################

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

# train model on data with only one wind direction
case_list = ['west']

# train model on data of two wind directions
# case_list = ['west', 'north']

first_data_id = 1
train_data_num = 1000
test_data_num = 200
slide = 1 # stride of the sliding window for training and testing data
T_in = 5 # input data channel size
T = 1 # output data channel size

modes = 8 # frequency modes of FNO
width = 20 # higher dimension channels of FNO
layer = 4 # layers of FNO
batch_size = 1
learning_rate = 0.001
epochs = 50

gpu_id = 0

load_data_path = 'niigata_data/'

save_model_path = 'model/'
save_model_name = 'niigata' + '_ncase' + str(len(case_list)) + '_ep' + str(epochs) + '_ntrain' + str(train_data_num) + '_seed' + str(seed)

################################################################
# load data
################################################################

t1 = default_timer()

data_full = []
for case in case_list:
    data_list = []
    for i in range(train_data_num + test_data_num): 
        data_list.append(torch.from_numpy(np.load(load_data_path + case + '/niigata_' + case + '_' + str(i + first_data_id) + '.npy')))
    data_full.append(torch.stack(data_list, dim = -1))
data_full = torch.stack(data_full, dim = 0) # dimension of data_full: [case number, x, y, z, sequntial data number]

# Normalization
data_full = data_full.permute(4, 0, 1, 2, 3)
my_normalizer = UnitGaussianNormalizer(data_full[:train_data_num])
my_normalizer.SaveNormalizer(save_model_path + save_model_name)
data_full = my_normalizer.encode(data_full)
data_full = data_full.permute(1, 2, 3, 4, 0)

# construct dataset
train_dataset = MyDataset(data_full[..., :train_data_num], T_in, T, slide)
test_dataset = MyDataset(data_full[..., -test_data_num:], T_in, T, slide)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
ntrain = len(train_dataset)

t2 = default_timer()
print('data loaded, time used:', t2-t1)

################################################################
# training and evaluation
################################################################

torch.cuda.set_device(gpu_id)
device = torch.device('cuda')

model = FNO3d(modes, modes, modes, width, T_in, T, layer).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
iterations = epochs*(ntrain//batch_size)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
my_normalizer.cuda()

print('Ep  Train time  Test time  Train loss  Test loss')

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        y = my_normalizer.decode(y.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        out = my_normalizer.decode(out.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        l2 = myloss(out, y)
        l2.backward()

        optimizer.step()
        scheduler.step()

    model.eval()
    t2 = default_timer()

    train_l2 = get_loss(model, train_loader, myloss, my_normalizer, 10) # compute train loss on 1/10 of train data to save time
    test_l2 = get_loss(model, test_loader, myloss, my_normalizer, 1)
    t3 = default_timer()

    print(f"{ep} {(t2-t1):.2f} {(t3-t2):.2f} {train_l2:.8f} {test_l2:.8f}")

torch.save(model, save_model_path + save_model_name)

