# train/sample(8000) unshift-WGAN-GP
python main.py --gan_type WGAN_GP --dataset lsun-church_outdoor --epoch 50 --input_size 128 --save_dir 'models/unshift/bs_256' --result_dir 'results/unshift/bs_256' --stage train --batch_size 256
python main.py --gan_type WGAN_GP --dataset lsun-bed --epoch 25 --input_size 128 --save_dir 'models/unshift' --result_dir 'results/unshift' --stage test

# train/sample(8000) unshift-WGAN-GP-SR
python main.py --gan_type WGAN_GP --dataset lsun-church_outdoor --epoch 25 --input_size 128 --save_dir 'models/unshift' --result_dir 'results/unshift' --stage train --SR
python main.py --gan_type WGAN_GP --dataset lsun-church_outdoor --epoch 25 --input_size 128 --save_dir 'models/unshift' --result_dir 'results/unshift' --stage test

# train/sample(8000) shift-WGAN-GP
python main.py --gan_type WGAN_GP --dataset lsun-bed --epoch 25 --input_size 128 --save_dir 'models/shift' --result_dir 'results/shift' --stage train --shift_data
python main.py --gan_type WGAN_GP --dataset lsun-bed --epoch 25 --input_size 128 --save_dir 'models/shift' --result_dir 'results/shift' --stage test

# train/sample(8000) shift-WGAN-GP-SR
python main.py --gan_type WGAN_GP --dataset lsun-church_outdoor --epoch 50 --input_size 128 --save_dir 'models/unshift/up_4' --result_dir 'results/unshift/up_4' --stage train --batch_size 128 --gen_type 'mix' --upsample_layer 4
python main.py --gan_type WGAN_GP --dataset lsun-bed --epoch 25 --input_size 128 --save_dir 'models/shift' --result_dir 'results/shift' --stage test --SR

# get unshift output from shift-GANs
python main.py --gan_type WGAN_GP --dataset celeba --epoch 50 --input_size 32 --save_dir 'models/shift' --result_dir 'results/shift' --stage test --shift_data

# fid
python -m pytorch_fid '/data0/cly/spectralGan/GAN-collections/results/shift/cifar10/WGAN_GP_evaluate' '/data0/cly/spectralGan/GAN-collections/data/fid-stats/fid_stats_cifar10_train.npz'

# Reconstruction
python main.py --gan_type WGAN_GP --dataset lsun-church_outdoor --epoch 100 --input_size 128 --save_dir 'models_deconv/unshift' --result_dir 'results_deconv/unshift' --stage invert --use_deconv
python main.py --gan_type WGAN_GP --dataset lsun-church_outdoor --epoch 50 --input_size 128 --save_dir 'models/unshift' --result_dir 'results/unshift' --stage invert

python main.py --gan_type DRAGAN --dataset chest_xray --epoch 50 --input_size 256 --save_dir 'models_deconv/unshift/deconv_50' --result_dir 'results_deconv/unshift/deconv_50' --stage invert --batch_size 32 --gen_type deconv --SR
python main.py --gan_type DRAGAN --dataset lsun-church_outdoor --epoch 50 --input_size 128 --save_dir 'models/unshift/deconv_50' --result_dir 'results_deconv/unshift/deconv_50' --stage invert --batch_size 32 --gen_type deconv --SR
python main.py --gan_type DRAGAN --dataset celeba --epoch 50 --input_size 128 --save_dir 'models/unshift/deconv_50' --result_dir 'results_deconv/unshift/deconv_50' --stage invert --batch_size 32 --gen_type deconv --SR
python main.py --gan_type DRAGAN --dataset lsun-church_outdoor --epoch 100 --input_size 128 --save_dir 'models_deconv/unshift' --result_dir 'results_deconv/unshift/deconv_50' --stage invert --batch_size 32 --gen_type deconv
# SupFig3
python main.py --gan_type WGAN_GP --dataset lsun-church_outdoor --epoch 100 --input_size 128 --save_dir 'models_deconv/unshift' --result_dir 'results_deconv/unshift' --gen_type deconv --stage show

# Relativistic
python main.py --gan_type LSGAN --dataset lsun-church_outdoor --epoch 50 --input_size 128 --save_dir 'models/relativistic' --result_dir 'results/relativistic' --stage train --batch_size 128 --R