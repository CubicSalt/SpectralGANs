import argparse, os, torch
from networks import GAN
import os
import torch.multiprocessing as mp
import numpy as np

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['WGAN_GP', 'DRAGAN', 'LSGAN', 'DCGAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['cifar10', 'lsun-church_outdoor', 'celeba', 'CT', 'VHR10', 'chest_xray', 'DOTA', 'maps'],
                        help='The name of dataset')
    
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=32, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', 
                        help='Directory name to save the generated images')
    
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--SR_lambda_', type=float, default=2.5)
    
    parser.add_argument('--stage', type=str, default='train', 
                        choices=['train', 'test', 'invert', 'show'])
    parser.add_argument('--shift_data', default=False, action="store_true")
    parser.add_argument('--SR', default=False, action="store_true")
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--sample_num', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--radius', type=int, default=0)
    parser.add_argument('--gen_type', type=str, default='mix', choices=['deconv', 'upsample', 'mix'])
    parser.add_argument('--upsample_layer', type=int, default=1, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--sample_times', type=int, default=1)
    parser.add_argument('--R', default=False, action="store_true")
    
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args
      

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # distributed
    torch.backends.cudnn.benchmark = True
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12500'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5'
    world_size = 6

    # loss state dic
    mode_dict = {
        'adversarial_loss_mode': 'gan',
        'gradient_penalty_mode': 'none',
        'gradient_penalty_sample_mode': 'line',
        'n_critic': 1
    }
    if args.gan_type == 'WGAN_GP':
        mode_dict['adversarial_loss_mode'] = 'wgan'
        mode_dict['gradient_penalty_mode'] = '1-gp'
        mode_dict['gradient_penalty_sample_mode'] = 'line'
        mode_dict['n_critic'] = 5
    elif args.gan_type == 'DRAGAN':
        mode_dict['adversarial_loss_mode'] = 'gan'
        mode_dict['gradient_penalty_mode'] = '1-gp'
        mode_dict['gradient_penalty_sample_mode'] = 'dragan'
        mode_dict['n_critic'] = 1
    elif args.gan_type == 'LSGAN':
        mode_dict['adversarial_loss_mode'] = 'lsgan'
        mode_dict['n_critic'] = 1
    elif args.gan_type == 'DCGAN':
        mode_dict['adversarial_loss_mode'] = 'gan'
        mode_dict['n_critic'] = 1
    
    # declare instance for GAN
    gan = GAN(args, mode_dict)
    
    if args.stage == 'train':
        print(" [*] Training start!")
        # multiprocessing lunch
        mp.spawn(gan.train,
            args=(world_size,),
            nprocs=world_size,
            join=True)
        # gan.load()
        print(" [*] Training finished!")
    elif args.stage == 'test':
        print(" [*] Testing start!")
        mp.spawn(gan.test,
            args=(world_size,),
            nprocs=world_size,
            join=True)
        print(" [*] Testing finished!")
    elif args.stage == 'invert':
        # gan.load()
        # gan.inverse_train()
        # print(" [*] Reconstruction finished!")
        print(" [*] Reconstructing start!")
        mp.spawn(gan.invert,
            args=(world_size,),
            nprocs=world_size,
            join=True)
        print(" [*] Reconstructing finished!")
    elif args.stage == 'show':
        print(" [*] Showing start!")
        mp.spawn(gan.show,
            args=(world_size,),
            nprocs=world_size,
            join=True)
        print(" [*] Showing finished!")

if __name__ == '__main__':
    main()
