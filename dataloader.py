from torch.utils.data import DataLoader, distributed as dist
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import utils
import glob

class DiskImageDataset():
    def __init__(self, img_paths, map_fn=None):
        self.img_paths = img_paths
        self.map_fn = map_fn
        
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
        
    def __getitem__(self, i):
        return self.map_fn(self.pil_loader(self.img_paths[i]))
    
    def __len__(self):
        return len(self.img_paths)

class shiftData(object):
    def __call__(self, img):
        is_numpy = type(img) is np.ndarray
        
        h, w = img.shape[:2] if is_numpy else img.shape[1:]
        x = np.array(range(h))
        y = np.array(range(w))

        xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

        xv_flat = np.asarray(xv.flat)
        yv_flat = np.asarray(yv.flat)
        index = np.full_like(xv_flat, -1)**(xv_flat + yv_flat)
        
        index = index.reshape((h, w, -1)) if is_numpy else index.reshape((-1, h, w))
        
        return (img*index) if is_numpy else (img*index).float()
 
def dataloader(dataset, input_size, batch_size, distributed=True, shift_data=False, world_size=-1):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    if shift_data:
        print("!!!!  SHIFTING DATA  !!!!")
        transform = transforms.Compose([transform, shiftData()])

    if dataset == 'cifar10':
        train_dataset = DiskImageDataset(glob.glob('data/cifar10/cifar10' + str(input_size) + '/*.png'), map_fn=transform)
        
    elif dataset == 'lsun-church_outdoor':
        train_dataset = DiskImageDataset(glob.glob('data/lsun/church_outdoor' + str(input_size) + '/*.png'), map_fn=transform)
        
    elif dataset == 'celeba':
        train_dataset = DiskImageDataset(glob.glob('data/celeba/celeba' + str(input_size) + '/*.png'), map_fn=transform)
        
    elif dataset == 'CT':
        train_dataset = DiskImageDataset(glob.glob('data/CT_COVID/CT_COVID' + str(input_size) + '/*.png'), map_fn=transform)
        
    elif dataset == 'VHR10':
        train_dataset = DiskImageDataset(glob.glob('data/VHR10/VHR10' + str(input_size) + '/*.png'), map_fn=transform)
        
    elif dataset == 'chest_xray':
        train_dataset = DiskImageDataset(glob.glob('data/chest_xray/chest_xray' + str(input_size) + '/*.png'), map_fn=transform)
        
    elif dataset == 'DOTA':
        train_dataset = DiskImageDataset(glob.glob('data/DOTA/DOTA' + str(input_size) + '/*.png'), map_fn=transform)
        
    elif dataset == 'maps':
        train_dataset = DiskImageDataset(glob.glob('data/maps/maps' + str(input_size) + '/*.png'), map_fn=transform)
    
    train_sampler = dist.DistributedSampler(train_dataset) if distributed else None
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=20, 
                                pin_memory=False, sampler=train_sampler, drop_last=(len(train_dataset)%(batch_size*world_size)!=0))
    
    return train_sampler, data_loader


# class pre_process():
#     def __init__(self, dataset):
#         if dataset == 'celeba':
#             crop_size = self.ori_size = 128
#             offset_height = (218 - crop_size) // 2
#             offset_width = (178 - crop_size) // 2
#             crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
#             self.transform_celeba = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Lambda(crop),
#                 transforms.ToPILImage()
#             ])
#             self.dataset = dataset
#             self.resize = [32, 64]
#             self.image_path = "/data0/cly/spectralGan/GAN-collections/data/celeba/"
#             self.real_data = DiskImageDataset(glob.glob(self.image_path+'img_align_celeba/' + '*.jpg'), map_fn=self.transform_celeba)
#         elif dataset == 'church_outdoor':
#             self.resize = [32, 64]
#             self.ori_size = 128
#             self.real_data = datasets.LSUN('data/lsun', classes=['church_outdoor_train'], transform=transforms.Resize((self.ori_size, self.ori_size)))
#             self.image_path = "/data0/cly/spectralGan/GAN-collections/data/lsun/"
#             self.dataset = dataset
            
#     def save_resize_imgs(self):
#         ds = self.real_data if self.dataset == 'celeba' else self.real_data.data
#         for index, x_ in enumerate():
#             scipy.misc.imsave(self.image_path + self.dataset + str(self.ori_size) + '/' + self.dataset + str(self.ori_size) + '_' + str(index) + '.png', x_)
#             for size in self.resize:
#                 scipy.misc.imsave(self.image_path + self.dataset + str(size) + '/' + self.dataset + str(size) + '_' + str(index) + '.png', 
#                                 transforms.Resize((size, size))(x_))