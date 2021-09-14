from utils import *
import time

import numpy as np
from glob import glob
from tqdm import tqdm
import seaborn as sns
import pandas as pd


from networks import *

automatic_gpu_usage()

class GAN() :
    def __init__(self, args):
        super(GAN, self).__init__()

        self.model_name = args.model_name
        self.phase = args.phase
        self.checkpoint_dir_original = args.checkpoint_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir_original = args.result_dir
        self.result_dir = args.result_dir
        self.log_dir_original = args.log_dir #
        self.log_dir = args.log_dir
        self.sample_dir_original = args.sample_dir
        self.sample_dir = args.sample_dir
        self.augment_flag = args.augment_flag

        self.decay_flag = args.decay_flag
        self.decay_iter = args.decay_iter
        self.iteration = args.iteration
        self.gan_type = args.gan_type
        self.rito = args.rito

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.img_height = args.img_size
        self.img_width = args.img_size
        self.img_ch = args.img_ch
        self.dw = args.dw
        self.img_shape = [self.img_height, self.img_width, self.img_ch]
        self.pixle_shape = [self.img_height, self.img_width, 1]
        self.mask_shape = [self.img_height, self.img_width, 2]

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.rec_weight = args.rec_weight
        self.cyc_weight = args.cyc_weight
        self.cls_weight = args.cls_weight

        """ Discriminator """
        self.label_size = args.label_size
        self.sn = args.sn
        
        """ Dataset """
        self.trainA_dataset_type = args.datasetA_type
        self.trainB_dataset_type = args.datasetB_type
        self.trainC_dataset_type = args.datasetC_type
        self.trainA_dataset_img_type = args.datasetA_img_type
        self.trainB_dataset_img_type = args.datasetB_img_type
        self.trainC_dataset_img_type = args.datasetC_img_type
        self.trainA_dataset_path = args.datasetA_path
        self.trainB_dataset_path = args.datasetB_path
        self.trainC_dataset_path = args.datasetC_path
        self.test_path = args.test_path

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print("# spectral normalization : ", self.sn)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train' :
            """ Input Image - data augment"""
            self.Image_Data_A = ImageData(self.trainA_dataset_path, img_shape = self.img_shape, augment_flag = self.augment_flag,
                                    data_type = self.trainA_dataset_type, img_type = self.trainA_dataset_img_type, label_size=self.label_size, channels=1)
            self.Image_Data_B = ImageData(self.trainB_dataset_path, img_shape = self.img_shape, augment_flag = self.augment_flag, 
                                    data_type = self.trainB_dataset_type, img_type = self.trainB_dataset_img_type, label_size=self.label_size, channels=3)
            self.Image_Data_C = ImageData(self.trainC_dataset_path, img_shape = self.img_shape, augment_flag = self.augment_flag, 
                                    data_type = self.trainC_dataset_type, img_type = self.trainB_dataset_img_type, label_size=self.label_size, channels=3)
            
            img_slice_A = tf.data.Dataset.from_tensor_slices((self.Image_Data_A.train_dataset, self.Image_Data_A.train_label))
            img_slice_B = tf.data.Dataset.from_tensor_slices((self.Image_Data_B.train_dataset, self.Image_Data_B.train_label))
            img_slice_C = tf.data.Dataset.from_tensor_slices((self.Image_Data_C.train_dataset, self.Image_Data_C.train_label))
            
            self.dataset_num = max(len(self.Image_Data_A.train_dataset), len(self.Image_Data_B.train_dataset), len(self.Image_Data_C.train_dataset))

            print("Dataset number : ", self.dataset_num)
            
            img_slice_A = img_slice_A.shuffle(self.dataset_num).repeat().map(self.Image_Data_A.PFD_image_processing, num_parallel_calls=4).batch(self.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            img_slice_B = img_slice_B.shuffle(self.dataset_num).repeat().map(self.Image_Data_B.image_processing, num_parallel_calls=4).batch(self.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            img_slice_C = img_slice_C.shuffle(self.dataset_num).repeat().map(self.Image_Data_C.RafD_image_processing, num_parallel_calls=4).batch(self.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            
            self.dataset_A_iter = iter(img_slice_A)
            self.dataset_B_iter = iter(img_slice_B)
            self.dataset_C_iter = iter(img_slice_C)
            
            """ Network """
            self.unet = Unet(self.img_shape, self.ch, output_channel=2, relord_path='unet.h5', name='unet')
            self.Gf = AE(self.img_shape, self.ch, output_channel=1, separable=self.dw, name='Gf')
            self.Gp = AE(self.pixle_shape, self.ch//2, output_channel=2, name='Gp')
            self.Gm = AE(self.mask_shape, self.ch//2, output_channel=1, name='Gm')
            self.dis = Discriminator([25,23,1], self.ch, self.label_size, self.sn, name='dis')

            """ Optimizer """
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
            self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(Gf=self.Gf, Gp=self.Gp, Gm=self.Gm, dis=self.dis, 
                                            g_optimizer=self.g_optimizer, d_optimizer=self.d_optimizer)
            
            self.sample_dir = os.path.join(self.sample_dir_original, self.model_dir)
            check_folder(self.sample_dir)

            self.checkpoint_dir = os.path.join(self.checkpoint_dir_original, self.model_dir)
            check_folder(self.checkpoint_dir)

            self.log_dir = os.path.join(self.log_dir_original, self.model_dir)
            check_folder(self.log_dir)

            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint)
                self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

        else :
            """ Test """
            if self.test_path == None:
                self.batch_size = 10
                self.Image_Data_B = ImageData(self.trainB_dataset_path, img_shape = self.img_shape, augment_flag = False, 
                                            data_type = self.trainB_dataset_type, img_type = self.trainB_dataset_img_type, channels=3)
                
                img_slice_B = tf.data.Dataset.from_tensor_slices((self.Image_Data_B.test_dataset, self.Image_Data_B.test_label))
                
                self.dataset_num = len(self.Image_Data_B.test_dataset)

                print("Dataset number : ", self.dataset_num)
                
                img_slice_B = img_slice_B.map(self.Image_Data_B.image_processing, num_parallel_calls=8).batch(self.batch_size, drop_remainder=True)
                
                self.dataset_B_iter = iter(img_slice_B)

            """ Network """
            self.Gf = AE(self.img_shape, self.ch, output_channel=1, separable=self.dw, name='Gf')

            """ Checkpoint """
            self.checkpoint_dir = os.path.join(self.checkpoint_dir_original, self.model_dir)
            check_folder(self.checkpoint_dir)

            self.result_dir = os.path.join(self.result_dir_original, self.model_dir)
            check_folder(self.result_dir)

            self.ckpt = tf.train.Checkpoint(Gf=self.Gf)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')

    @tf.function
    def g_train_step(self, real_face, face_label, radf_face, radf_label):

        face = tf.concat([real_face, radf_face], axis=0)
        # get mask
        m = tf.cast(tf.greater_equal(tf.sigmoid(self.unet(face)), 0.5), tf.float32)

        with tf.GradientTape() as g_tape:

            # face to pixl
            f2p = tf.sigmoid(self.Gf(face))
            # 64*64 to 25*23
            f2p = crop_resize(f2p)
            # dis
            f2p_logits, f2p_label = self.dis(f2p)
            f2p_label_1, f2p_label_2 = tf.split(f2p_label, 2, axis=0)
            # 25*23 to 64*64
            f2p= resize_pad(f2p)
            # cyc of pixl
            p2m_logit = self.Gp(f2p)
            p2m = tf.sigmoid(p2m_logit)
            pmp = tf.sigmoid(self.Gm(p2m))
            # dis
            pmp_logits, pmp_label = self.dis(crop_resize(pmp))
            pmp_label_1, pmp_label_2 = tf.split(pmp_label, 2, axis=0)
            # cyc of mask
            m2p = tf.sigmoid(self.Gm(m))
            mpm_logit = self.Gp(m2p)
            # dis
            m2p_logits, m2p_label = self.dis(crop_resize(m2p))
            m2p_label_1, m2p_label_2 = tf.split(m2p_label, 2, axis=0)

            g_adv_loss = gen_loss(self.gan_type, [f2p_logits, pmp_logits, m2p_logits])

            g_cls_loss = tf.reduce_mean([
                cls_loss(logits=f2p_label_1, labels=face_label, type=1, label_size=self.label_size),
                cls_loss(logits=f2p_label_2, labels=radf_label, type=2, label_size=self.label_size),
                cls_loss(logits=pmp_label_1, labels=face_label, type=1, label_size=self.label_size),
                cls_loss(logits=pmp_label_2, labels=radf_label, type=2, label_size=self.label_size),
                cls_loss(logits=m2p_label_1, labels=face_label, type=1, label_size=self.label_size),
                cls_loss(logits=m2p_label_2, labels=radf_label, type=2, label_size=self.label_size)])

            g_rec_loss = tf.reduce_mean([
                CE_loss(logits=p2m_logit, labels=m),
                L1_loss(f2p, m2p)])

            g_cyc_loss = tf.reduce_mean([
                L1_loss(f2p, pmp),
                CE_loss(logits=mpm_logit, labels=m)])

            g_loss = self.adv_weight * g_adv_loss \
                   + self.cls_weight * g_cls_loss \
                   + self.rec_weight * g_rec_loss \
                   + self.cyc_weight * g_cyc_loss \

        g_train_variable = self.Gf.trainable_variables\
                         + self.Gp.trainable_variables\
                         + self.Gm.trainable_variables\
                         
        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))
        f2p_1, f2p_2 = tf.split(f2p, 2, axis=0)
        m_1, m_2 = tf.split(m, 2, axis=0)
        p2m_1, p2m_2 = tf.split(p2m, 2, axis=0)

        return [g_loss, g_adv_loss, g_cls_loss, g_rec_loss, g_cyc_loss], [f2p_1, m_1, p2m_1, f2p_2, m_2, p2m_2]

    @tf.function
    def d_train_step(self, real_face, radf_face, real_pixl, pixl_label):
            
        face = tf.concat([real_face, radf_face], axis=0)
        m = tf.cast(tf.greater_equal(tf.sigmoid(self.unet(face)), 0.5), tf.float32)
        f2p = crop_resize(tf.sigmoid(self.Gf(face)))
        pmp = crop_resize(tf.sigmoid(self.Gm(tf.sigmoid(self.Gp(resize_pad(f2p))))))
        m2p = crop_resize(tf.sigmoid(self.Gm(m)))

        with tf.GradientTape() as d_tape:
            
            real_logits, label_logits = self.dis(real_pixl)
            
            f2p_logits, _ = self.dis(f2p)
            pmp_logits, _ = self.dis(pmp)
            m2p_logits, _ = self.dis(m2p)

            d_adv_loss = dis_loss(self.gan_type, real_logits, [f2p_logits, pmp_logits, m2p_logits])

            d_cls_loss = cls_loss(logits=label_logits, labels=pixl_label, type=0, label_size=self.label_size)\

            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
                grad_pen_t = gradient_penalty(self.dis, tf.concat([real_pixl, real_pixl], axis=0), [f2p, pmp, m2p], gan_type=self.gan_type)
            else :
                grad_pen_t = 0

            d_regular_loss = regularization_loss(self.dis)

            d_loss = self.adv_weight * d_adv_loss \
                   + self.cls_weight * d_cls_loss \
                   + grad_pen_t + d_regular_loss

        d_train_variable = self.dis.trainable_variables\

        d_gradient = d_tape.gradient(d_loss, d_train_variable)
        self.d_optimizer.apply_gradients(zip(d_gradient, d_train_variable))

        return d_loss, d_adv_loss, d_cls_loss, d_regular_loss

    def train(self):

        # loop for epoch
        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        for idx in range(self.start_iteration, self.iteration):
            current_step = idx
            if self.decay_flag :
                total_step = self.iteration
                decay_start_step = self.decay_iter

                if current_step >= decay_start_step :
                    lr = self.init_lr * (total_step - current_step) / (total_step - decay_start_step)
                    self.g_optimizer.learning_rate = lr
                    self.d_optimizer.learning_rate = lr
            
            real_pixl, pixl_label = next(self.dataset_A_iter)
            real_face, face_label = next(self.dataset_B_iter)
            radf_face, radf_label = next(self.dataset_C_iter)

            # update discriminator
            d_loss = self.d_train_step(real_face, radf_face, real_pixl, pixl_label)
            [self.d_loss, self.d_adv_loss, self.d_cls_loss, self.d_regular_loss] = d_loss

            # update generator
            if np.mod(idx+1, self.rito)==0:
                g_loss, img = self.g_train_step(real_face, face_label, radf_face, radf_label)
                [self.g_loss, self.g_adv_loss, self.g_cls_loss, self.g_rec_loss, self.g_cyc_loss] = g_loss
                [f2p_1, m_1, p2m_1, f2p_2, m_2, p2m_2] = img
                m_11, m_12 = tf.split(m_1, 2, axis=-1)
                m_21, m_22 = tf.split(m_2, 2, axis=-1)
                p2m_11, p2m_12 = tf.split(p2m_1, 2, axis=-1)
                p2m_21, p2m_22 = tf.split(p2m_2, 2, axis=-1)

                # save to tensorboard
                with train_summary_writer.as_default():
                    with tf.name_scope('g_loss') :
                        tf.summary.scalar('g_adv_loss', self.g_adv_loss, step=idx)
                        tf.summary.scalar('g_cls_loss', self.g_cls_loss, step=idx)
                        tf.summary.scalar('g_rec_loss', self.g_rec_loss, step=idx)
                        tf.summary.scalar('g_cyc_loss', self.g_cyc_loss, step=idx)
                        tf.summary.scalar('g_loss', self.g_loss, step=idx)
                    with tf.name_scope('d_loss') :
                        tf.summary.scalar('d_adv_loss', self.d_adv_loss, step=idx)
                        tf.summary.scalar('d_cls_loss', self.d_cls_loss, step=idx)
                        tf.summary.scalar('d_regular_loss', self.d_regular_loss, step=idx)
                        tf.summary.scalar('d_loss', self.d_loss, step=idx)

            # save every self.save_freq
            if np.mod(idx+1, self.save_freq) == 0:
                self.manager.save(checkpoint_number=idx+1)

            # save every self.print_freq
            if np.mod(idx+1, self.print_freq) == 0:
                merge = []
                merge.append(np.expand_dims(return_images(real_face, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(1-f2p_1, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(m_11, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(m_12, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(p2m_11, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(p2m_12, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(radf_face, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(1-f2p_2, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(m_21, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(m_22, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(p2m_21, [self.batch_size, 1]), axis=0))
                merge.append(np.expand_dims(return_images(p2m_22, [self.batch_size, 1]), axis=0))

                merge_img = np.concatenate(merge, axis=0)
                save_images(merge_img, [1, len(merge)],
                            './{}/merge_{:07d}.jpg'.format(self.sample_dir, idx + 1), range_type=1, show=False)

                print("iter: [%6d/%6d] time: %4.4f" % \
                        (idx+1, self.iteration, time.time() - start_time))
                start_time = time.time()


        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

    @property
    def model_dir(self):

        sn = 'sn' if self.sn else ''
        dw = 'dw' if self.dw else ''

        return "{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.gan_type, sn, dw, 
                            self.adv_weight, self.cls_weight, self.rec_weight)

    def test(self):

        if self.test_path == None:

            for idx in tqdm(range(0,self.dataset_num,self.batch_size), ncols=50):

                real_face, _ = next(self.dataset_B_iter)
                f2p = tf.sigmoid(self.Gf(real_face))
                f2p = crop_resize(f2p)
                f2p = resize_pad(f2p)

                merge_1 = np.expand_dims(return_images(real_face, [self.batch_size, 1]), axis=0)
                merge_2 = np.expand_dims(return_images(1-f2p, [self.batch_size, 1]), axis=0)
                merge_img = np.concatenate([merge_1, merge_2], axis=0)
                save_images(merge_img, [1, 2],
                            './{}/merge_{:07d}.jpg'.format(self.result_dir, idx + 1), range_type=1, show=False)
                for j, img in enumerate(f2p):
                    save_images(np.expand_dims(1-img, axis=0), [1, 1],
                            './{}/z_{:04d}_{:02d}.jpg'.format(self.result_dir, idx + 1, j), range_type=1, show=False)

        else:
            for img_path in tqdm(sorted(glob( self.test_path + os.sep + '*.*' ))):
                test_img = load_test_data(img_path)
                f2p = tf.sigmoid(self.Gf(test_img))
                f2p = crop_resize(f2p)
                f2p = resize_pad(f2p)
                test_img = np.expand_dims(return_images(test_img, [1, 1]), axis=0)
                f2p = 1-np.expand_dims(return_images(f2p, [1, 1]), axis=0)
                save_name = os.path.basename(img_path)
                save_images(f2p, [1, 1],
                            './{}/f2p_{}'.format(self.result_dir, save_name), range_type=1, show=False)
                save_images(np.concatenate([test_img, f2p], axis=0), [1, 2],
                            './{}/m_{}'.format(self.result_dir, save_name), range_type=1, show=False)
        
        return 0