import utils, torch, time, os, pickle
import functools
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from dataloader import dataloader, shiftData

from scipy.interpolate import griddata
from torch.autograd import Variable

from torch.nn.parallel import DistributedDataParallel as DDP
import encoder
import radialProfile
import loss
import gradient_penalty as gp
import layers as l
import scipy.misc

from torchvision import datasets, transforms
from PIL import Image

class generator_mix(nn.Module):
    def __init__(self, 
                 input_dim = 128, 
                 output_dim = 3, 
                 input_size = 128,
                 apply_SR = False,
                 dim = 64,
                 norm = 'batch_norm',
                 upsample_layer = 1):
        super().__init__()
        n_upsamplings = {32:3, 64:4, 128:5, 256:6}[input_size]
            
        def upsample_norm_relu(in_dim, out_dim, kernel_size=5, stride=1, padding=2):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                l._get_norm_layer_2d(norm, out_dim),
                nn.ReLU()
            )
            
        def dconv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),
                l._get_norm_layer_2d(norm, out_dim),
                nn.ReLU()
            )
            
        layers = []

        # 1: 1x1 -> 4x4
        d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
        layers.append(dconv_norm_relu(input_dim, d, kernel_size=4, stride=1, padding=0))
    
        # 2: deconv, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_upsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
            if i < n_upsamplings - 1 - (upsample_layer - 1):
                layers.append(dconv_norm_relu(d_last, d, kernel_size=4, stride=2, padding=1))
            else:
                layers.append(upsample_norm_relu(d_last, d, kernel_size=5, stride=1, padding=2))

        if upsample_layer > 0:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, output_dim, kernel_size=5, stride=1, padding=2))
        else:
            layers.append(nn.ConvTranspose2d(d, output_dim, kernel_size=4, stride=2, padding=1))
        
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        
        utils.initialize_weights(self)
        
    def forward(self, z):
        x = self.net(z)
        return x

class generator_upsample(nn.Module):
    def __init__(self, 
                 input_dim = 128, 
                 output_dim = 3, 
                 input_size = 32,
                 apply_SR = False,
                 dim = 64,
                 norm = 'batch_norm'):
        super().__init__()
    
        n_upsamplings = n_downsamplings = {32:3, 64:4, 128:5, 256:6}[input_size]
            
        def upsample_norm_relu(in_dim, out_dim, kernel_size=5, stride=1, padding=2):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False or Norm == l.Identity),
                l._get_norm_layer_2d(norm, out_dim),
                nn.ReLU()
            )
        
        layers = []

        # 1: 1x1 -> 4x4
        d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
        layers.append(nn.Sequential(
                nn.Conv2d(input_dim, d, kernel_size=4, stride=1, padding=3, bias=False or Norm == l.Identity),
                l._get_norm_layer_2d(norm, out_dim),
                nn.ReLU()
            ))
        
        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_upsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
            layers.append(upsample_norm_relu(d_last, d, kernel_size=5, stride=1, padding=2))

        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Conv2d(d, output_dim, kernel_size=5, stride=1, padding=2))
        
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        
        # utils.initialize_weights(self)
        
    def forward(self, z):
        x = self.net(z)
        return x

class generator(nn.Module):
    def __init__(self, 
                 input_dim = 128, 
                 output_dim = 3, 
                 input_size = 32,
                 apply_SR = False,
                 dim = 64,
                 norm = 'batch_norm'):
        super().__init__()
        
        n_upsamplings = n_downsamplings = {32:3, 64:4, 128:5, 256:6}[input_size]
        
        def dconv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),
                l._get_norm_layer_2d(norm, out_dim),
                nn.ReLU()
            )

        layers = []

        # 1: 1x1 -> 4x4
        d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
        layers.append(dconv_norm_relu(input_dim, d, kernel_size=4, stride=1, padding=0))

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_upsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
            layers.append(dconv_norm_relu(d_last, d, kernel_size=4, stride=2, padding=1))
           
        if apply_SR:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, output_dim, kernel_size=5, stride=1, padding=2))
        else:
            layers.append(nn.ConvTranspose2d(d, output_dim, kernel_size=4, stride=2, padding=1))
            
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        
        utils.initialize_weights(self)
        
    def forward(self, z):
        x = self.net(z)
        return x

class discriminator(nn.Module):
    def __init__(self, 
                 input_dim = 3, 
                 output_dim = 1, 
                 input_size = 32,
                 dim = 64,
                 norm = 'layer_norm'):
        super().__init__()
        
        n_upsamplings = n_downsamplings = {32:3, 64:4, 128:5, 256:6}[input_size]

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),
                l._get_norm_layer_2d(norm, out_dim),
                nn.LeakyReLU(0.2)
            )

        layers = []

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_dim, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 8)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))

        # 2: logit
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))

        self.net = nn.Sequential(*layers)

        utils.initialize_weights(self)
        
    def forward(self, x):
        y = self.net(x)
        return y  
    
class GAN(object):
    def __init__(self, args, mode_dict):
        # parameters
        self.epoch = args.epoch
        self.sample_num = args.sample_num
        self.sample_times = args.sample_times
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.model_name = args.gan_type
        self.input_size = args.input_size
        
        self.stage = args.stage
        self.shift_data = args.shift_data
        self.SR = args.SR 
        self.R = args.R 
        self.SR_lambda_ = args.SR_lambda_
        self.exp_name = args.exp_name
        self.distributed = args.distributed
        self.gen_type = args.gen_type
        if self.gen_type == 'mix':
            self.upsample_layer = args.upsample_layer
        
        if self.SR:
            self.model_name += '_SR' 
        if self.R:
            self.model_name += '_Relativistic' 
        if self.exp_name is not None:
            self.model_name += '_' + self.exp_name
        self.model_name += '_' + str(self.input_size)
        self.save_model_name = self.model_name + '_' + str(self.epoch)
            
        self.lrG, self.lrD, self.beta1, self.beta2 = args.lrG, args.lrD, args.beta1, args.beta2
        
        self.z_dim = 128
        self.lambda_ = 10
           
        self.adversarial_loss_mode = mode_dict['adversarial_loss_mode']
        self.gradient_penalty_mode = mode_dict['gradient_penalty_mode']
        self.gradient_penalty_sample_mode =  mode_dict['gradient_penalty_sample_mode']
        self.n_critic = mode_dict['n_critic']
        
    def setup_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
        torch.cuda.set_device(self.rank)

        # initialize the process group
        torch.distributed.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        
        self.sample_z_ = self.sample_z_.cuda()
        self.G = DDP(self.G.cuda(), device_ids=[self.rank])
        self.D = DDP(self.D.cuda(), device_ids=[self.rank])
        self.E = DDP(self.E.cuda(), device_ids=[self.rank])
    
    def setup_gan(self):
        # networks init
        # setup the normalization function for discriminator
        # if self.gradient_penalty_mode == 'none':
        #     d_norm = 'batch_norm'
        # else:  # cannot use batch normalization with gradient penalty
        d_norm = 'layer_norm'
        
        if(self.gen_type == 'deconv'):
            self.G = generator(input_dim=self.z_dim, output_dim=3, input_size=self.input_size, apply_SR=self.SR)
        elif(self.gen_type == 'upsample'):
            self.G = generator_upsample(input_dim=self.z_dim, output_dim=3, input_size=self.input_size, apply_SR=self.SR)
        elif(self.gen_type == 'mix'):
            self.G = generator_mix(input_dim=self.z_dim, output_dim=3, input_size=self.input_size, upsample_layer=self.upsample_layer)
            
        self.D = discriminator(input_dim=3, output_dim=1, input_size=self.input_size, norm=d_norm)
        self.E = encoder.resnet56()
            
        # optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))
        self.E_optimizer = optim.Adam(self.E.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))
        
        # adversarial_loss_functions
        self.d_loss_fn, self.g_loss_fn = loss.get_adversarial_losses_fn(self.adversarial_loss_mode)

        # fixed noise
        self.sample_z_ = torch.randn((self.batch_size, self.z_dim, 1, 1))
        
    def train(self, rank=-1, world_size=-1):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        if self.SR:
            self.train_hist['psd1D_freq_loss'] = []
            self.train_hist['phase1D_freq_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        
        print('--SR', self.SR)
        
        self.setup_gan()
        
        if rank == 0:
            print('---------- Networks architecture -------------')
            utils.print_network(self.G)
            utils.print_network(self.D)
            # utils.print_network(self.E)
            print('-----------------------------------------------') 
        
        if self.distributed: 
            self.setup_distributed(rank, world_size)
            
        # load dataset
        self.train_sampler, self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size, self.distributed, shift_data=self.shift_data, world_size=self.world_size)
        
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        
        if self.distributed:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            self.train_sampler.set_epoch(epoch)
            
            for iter, x_ in enumerate(self.data_loader):
                z_ = torch.randn((self.batch_size, self.z_dim, 1, 1))
                if self.distributed:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                
                G_ = self.G(z_).detach()
                D_fake = self.D(G_)
                
                D_real_loss, D_fake_loss = self.d_loss_fn(D_real, D_fake)

                # gradient penalty
                gradient_penalty_ = gp.gradient_penalty(functools.partial(self.D), x_, G_, 
                                                        gp_mode=self.gradient_penalty_mode, 
                                                        sample_mode=self.gradient_penalty_sample_mode)
                

                D_loss = D_real_loss + D_fake_loss + self.lambda_ * gradient_penalty_
                
                if self.R:
                    D_relativistic_loss = (torch.mean((D_real - torch.mean(D_fake) - torch.ones_like(D_real)) ** 2) + torch.mean((D_fake - torch.mean(D_real) + torch.ones_like(D_real)) ** 2))/20
                    D_loss += D_relativistic_loss

                D_loss.backward()
                self.D_optimizer.step()

                if ((iter+1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()
                    
                    G_ = self.G(z_)
                    
                    if self.R:
                        D_real = self.D(x_)
                        D_fake = self.D(G_.detach())
                    else:
                        D_fake = self.D(G_)
                        
                    
                    g_loss = self.g_loss_fn(D_fake)
                    
                    ##############################################################################
                    if self.SR:
                        psd1D_img = Variable(self.get_spectrum1D(G_, 'magnitude'), requires_grad=True).cuda()
                        psd1D_rec = Variable(self.get_spectrum1D(x_, 'magnitude'), requires_grad=True).cuda()

                        phase1D_img = Variable(self.get_spectrum1D(G_, 'phase'), requires_grad=True).cuda()
                        phase1D_rec = Variable(self.get_spectrum1D(x_, 'phase'), requires_grad=True).cuda()

                        psd1D_loss_freq = nn.MSELoss()(psd1D_rec, psd1D_img.detach())
                        phase1D_loss_freq = nn.MSELoss()(phase1D_rec, phase1D_img.detach())
                        # psd1D_loss_freq *= g_loss
                        # phase1D_loss_freq *= g_loss
                    
                        G_loss = g_loss + 0.45*self.SR_lambda_*psd1D_loss_freq + 0.55*self.SR_lambda_*phase1D_loss_freq
                        # self.SR_lambda_*psd1D_loss_freq
                        # self.SR_lambda_*phase1D_loss_freq
                        
                        self.train_hist['psd1D_freq_loss'].append(psd1D_loss_freq.item())
                        self.train_hist['phase1D_freq_loss'].append(phase1D_loss_freq.item())
                    else:
                        G_loss = g_loss
                        
                    # if self.R:
                    #     G_relativistic_loss = (torch.mean((D_real - torch.mean(D_fake) + torch.ones_like(D_real)) ** 2) + torch.mean((D_fake - torch.mean(D_real) - torch.ones_like(D_real)) ** 2))/20
                    #     G_loss += G_relativistic_loss
                    ##############################################################################

                    self.train_hist['G_loss'].append(g_loss.item())
                    G_loss.backward()
                    self.G_optimizer.step()

                    self.train_hist['D_loss'].append(D_loss.item())

                if self.rank == 0:
                    if ((iter + 1) % 100) == 0:
                        if self.SR:
                            print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, psd1D_loss_freq: %.8f, phase1D_loss_freq: %.8f" %
                            ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), g_loss.item(), psd1D_loss_freq.item(), phase1D_loss_freq.item()))
                        else:
                            print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                            ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
            
            if self.rank == 0:
                ##############save every 10 epochs#####################
                if (epoch+1)%10==0: 
                    self.save(str(epoch+1))
               
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                with torch.no_grad():
                    self.visualize_results((epoch+1))
                    
        if self.rank == 0:
            self.train_hist['total_time'].append(time.time() - start_time)
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                self.epoch, self.train_hist['total_time'][0]))
            print("Training finish!... save training results")

        # rank 0 process
        if self.rank == 0:
            ##############save the final model#####################
            self.save(str(self.epoch))
        
            utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.save_model_name + '/' + self.save_model_name,
                                    self.epoch)
            utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.save_model_name), self.save_model_name)

    def test(self, rank, world_size):
        
        self.setup_gan()
        
        if self.distributed: 
            self.setup_distributed(rank, world_size)
            
        self.load(str(self.epoch))
        self.visualize_results(self.epoch, fix=False, flat=True)

    def visualize_results(self, epoch, fix=True, flat=False, show=False):
        self.G.eval()

        if self.stage == 'train':
            tot_num_samples = self.batch_size
        else:
            tot_num_samples = self.sample_num
            
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        for cnt in range(self.sample_times):
            if fix:
                """ fixed noise """
                samples = self.G(self.sample_z_)
            else:
                """ random noise """
                sample_z_ = torch.randn((tot_num_samples, self.z_dim, 1, 1))
                if self.distributed:
                    sample_z_ = sample_z_.cuda()

                samples = self.G(sample_z_)

            if self.distributed:
                samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
            else:
                samples = samples.data.numpy().transpose(0, 2, 3, 1)

            if flat:
                if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.save_model_name + '_evaluate'):
                    os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.save_model_name + '_evaluate')
                utils.save_images_flat(samples, 
                                self.result_dir + '/' + self.dataset + '/' + self.save_model_name + '_evaluate' + '/' + self.save_model_name, self.shift_data, str(self.rank + 4 * cnt))
                
            elif show:
                if self.rank == 0:
                    if not os.path.exists('./figure/SupFig3'): 
                        os.makedirs('./figure/SupFig3')
                    if self.shift_data:
                        for i in range(16):
                            samples[i, :, :, :] = shiftData()(samples[i, :, :, :])
                        utils.save_images(samples[:16, :, :, :], [4, 4], './figure/SupFig3/' + self.model_name + '_S-' + self.dataset + '_' + self.gen_type+ '.png')
                    else:
                        utils.save_images(samples[:16, :, :, :], [4, 4], './figure/SupFig3/' + self.model_name + '_' + self.dataset + '_' + self.gen_type+ '.png')
            
            else:
                if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.save_model_name):
                    os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.save_model_name)
                if(self.shift_data):
                    for i in range(image_frame_dim * image_frame_dim):
                        samples[i, :, :, :] = shiftData()(samples[i, :, :, :])
                utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                            self.result_dir + '/' + self.dataset + '/' + self.save_model_name + '/' + self.save_model_name + '_epoch%03d' % epoch + '.png')


    def get_spectrum1D(self, img, flag='magnitude'):
        # fake image 1d power spectrum
        if self.input_size == 256:
            N = 179
        elif self.input_size == 128:
            N = 88
        elif self.input_size == 64:
            N = 43
        elif self.input_size == 32:
            N = 20
        epsilon = 1e-8
        rgb_weights = [0.2989, 0.5870, 0.1140]
        psd1D_img = np.zeros([img.shape[0], N])
        
        for t in range(img.shape[0]):
            gen_imgs = img.permute(0,2,3,1)
            img_numpy = gen_imgs[t,:,:,:].cpu().detach().numpy()
            img_gray = np.dot(img_numpy, rgb_weights)
            fft = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(fft)
            fshift += epsilon
            
            if flag == 'magnitude':
                spectrum = 20*np.log(np.abs(fshift))
            else:
                spectrum = 10*np.log(np.abs(np.angle(fshift)) + epsilon)
                
            psd1D = radialProfile.azimuthalAverage(spectrum)
            psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))
            psd1D_img[t,:] = psd1D
        
        return torch.from_numpy(psd1D_img).float()

    def save(self, suffix='', save_E = False):
        save_model_name = self.model_name + '_' + suffix
        save_dir = os.path.join(self.save_dir, self.dataset, save_model_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if save_E:
            torch.save(self.E.state_dict(), os.path.join(save_dir, save_model_name + '_E.pkl'))
        else:
            torch.save(self.G.state_dict(), os.path.join(save_dir, save_model_name + '_G.pkl'))
            torch.save(self.D.state_dict(), os.path.join(save_dir, save_model_name + '_D.pkl'))

        with open(os.path.join(save_dir, save_model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self, suffix='', load_G = True, load_D = False, load_E = False):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        load_model_name = self.model_name + '_' + suffix
        
        save_dir = os.path.join(self.save_dir, self.dataset, load_model_name)
        
        if load_D == True:
            self.D.load_state_dict(torch.load(os.path.join(save_dir, load_model_name + '_D.pkl'), map_location=map_location))
        if load_E == True:
            self.E.load_state_dict(torch.load(os.path.join(save_dir, load_model_name + '_E.pkl'), map_location=map_location))
        if load_G == True:
            self.G.load_state_dict(torch.load(os.path.join(save_dir, load_model_name + '_G.pkl'), map_location=map_location))
        
    def inverse_train(self, load = True):
        self.E.train()
        print('training start!!')
        self.train_hist = {}
        self.train_hist['total_time'] = []
        start_time = time.time()
        
        # Training Encoder
        if not load:
            for epoch in range(5000):
                self.E_optimizer.zero_grad()
                z_ = torch.rand((self.batch_size, self.z_dim, 1, 1))
                if self.distributed:
                    z_ = z_.cuda()
                G_ = self.G(z_)
                z_inverse = self.E(G_).unsqueeze(2).unsqueeze(2)
                z_loss = torch.sum(torch.dist(z_inverse, z_, p=2))
                z_loss.backward()
                self.E_optimizer.step()
                if self.rank == 0:
                    if not ((epoch+1) % 500):
                        print("Inverse Encoder Training Iteration: [%2d] Inverse Encoder_loss: %.8f" %
                            ((epoch + 1), z_loss.item()))
            if self.rank == 0:
                self.save(str(self.epoch), save_E = True)
        else:
            self.load(str(self.epoch), load_G = False, load_E = True)
        
        # Finetuning G
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size, self.distributed, shift_data=self.shift_data)
        data = iter(self.data_loader)
        x_ = next(data)
        k = 1
        # path = './data/lsun/church_outdoor128/church_outdoor128_497.png'
        # path = './data/chest_xray/chest_xray256/090.png'
        # with open(path, 'rb') as f:
            # x_[0] = transforms.ToTensor()(np.array(Image.open(f).convert('RGB')))
        
        if self.distributed:
            x_ = x_.cuda()
            
        for epoch in range(800):
            self.G_optimizer.zero_grad()
            z_ = self.E(x_).unsqueeze(2).unsqueeze(2)
            x_rec = self.G(z_)

            x_loss = torch.sum(torch.dist(x_rec, x_, p=2))
                
            x_loss.backward()
            
            self.G_optimizer.step()
            
            if self.rank == 0:
                if not ((epoch+1) % 50):
                    print("Finetuning Iteration: [%2d] iE_loss: %.8f" % ((epoch + 1), x_loss.item()))
                
        if self.rank == 0:
            self.train_hist['total_time'].append(time.time() - start_time)
            print("Total %d epochs time: %.2f" % (self.epoch, self.train_hist['total_time'][0]))
            print("Training finish!... save training results")

            if self.distributed:
                x_rec = x_rec.cpu().data.numpy().transpose(0, 2, 3, 1)
                x_ = x_.cpu().data.numpy().transpose(0, 2, 3, 1)
            else:
                x_rec = x_rec.data.numpy().transpose(0, 2, 3, 1)
                x_ = x_.data.numpy().transpose(0, 2, 3, 1)
            
            
            scipy.misc.imsave( './inversion/' + self.dataset + '_' + self.save_model_name + '_fig1_inverse_rec.png', x_rec[k])
            scipy.misc.imsave( './inversion/' + self.dataset + '_' + self.save_model_name + '_fig1_inverse_real.png', x_[k])
        
    def invert(self, rank, world_size):
            
        self.setup_gan()
        
        if self.distributed: 
            self.setup_distributed(rank, world_size)
            
        self.load(str(self.epoch))
        self.inverse_train()

    def show(self, rank, world_size):
        
        self.setup_gan()
        
        if self.distributed: 
            self.setup_distributed(rank, world_size)
            
        self.load(str(self.epoch))
        self.visualize_results(self.epoch, fix=False, flat=False, show=True)