from gan import GAN
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of F2Pnet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_name', type=str, default='F2Pnet', help='model name')
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--datasetA_type', type=str, default='PFD', help='datasetA_type')
    parser.add_argument('--datasetB_type', type=str, default='CelebA', help='datasetB_type')
    parser.add_argument('--datasetC_type', type=str, default='RafD', help='datasetB_type')
    parser.add_argument('--datasetA_img_type', type=str, default='png', help='datasetA_image_type')
    parser.add_argument('--datasetB_img_type', type=str, default='jpg', help='datasetB_image_type')
    parser.add_argument('--datasetC_img_type', type=str, default='jpg', help='datasetB_image_type')
    parser.add_argument('--datasetA_path', type=str, default='./AIRS-PFD/', help='datasetA_path')
    parser.add_argument('--datasetB_path', type=str, default='./CelebA/', help='datasetB_path')
    parser.add_argument('--datasetC_path', type=str, default='./RafD-front/', help='datasetC_path')
    parser.add_argument('--test_path', type=str, default=None, help='None for CelebA, or ./test')
    
    parser.add_argument('--augment_flag', type=bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--iteration', type=int, default=50000, help='The number of training iterations')

    parser.add_argument('--decay_flag', type=str2bool, default=False, help='The decay_flag')
    parser.add_argument('--decay_iter', type=int, default=50000, help='decay start iteration')

    parser.add_argument('--batch_size', type=int, default=16, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')

    parser.add_argument('--gan_type', type=str, default='hinge', help='[gan / lsgan / hinge / wgan-gp / wgan-div / dragan]')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')
    parser.add_argument('--rito', type=int, default=1, help='D and G update rito')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--adv_weight', type=float, default=1, help='weight of adversarial loss')
    parser.add_argument('--rec_weight', type=float, default=10.0, help='weight of image reconstraction loss')
    parser.add_argument('--cyc_weight', type=float, default=10.0, help='weight of cycle loss')
    parser.add_argument('--cls_weight', type=float, default=10.0, help='weight of classfication loss')
    
    parser.add_argument('--img_size', type=int, default=64, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--ch', type=int, default=64, help='using depthwise conv')
    parser.add_argument('--dw', type=int, default=False, help='base channel number per layer')
    parser.add_argument('--label_size', type=int, default=10, help='The label size of discriminator predict')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.iteration >= 1
    except:
        print('number of iterations must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():

    args = parse_args()
    # automatic_gpu_usage()

    gan = GAN(args)
    gan.build_model()    #zjf

    # build graph
    if args.phase == 'train' :
        gan.train()  # zjf 
        print(" [*] Training finished!")

    else :
        gan.test()
        print(" [*] Test finished!")


if __name__ == '__main__':
    main()
