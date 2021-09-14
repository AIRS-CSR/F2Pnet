import tensorflow as tf
from scipy import misc
import os, random
import numpy as np
from glob import glob
import cv2

    
class ImageData:

    def __init__(self, data_path, img_shape=(64,64,1), augment_flag=False, data_type='None', img_type='jpg', pad_flag=False, label_size=3, channels=1):
        self.data_path = data_path
        self.data_type = data_type
        self.img_shape = img_shape
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]
        self.channels = channels
        self.augment_flag = augment_flag
        self.img_type = img_type
        self.pad_flag = pad_flag
        self.label_size = label_size
        if self.data_type == 'PFD':
            self.PFD = PFD_Data(self.data_path, self.label_size)
            self.train_dataset, self.train_label = self.PFD.Data_Label()
        elif self.data_type == 'CelebA':
            self.CelebaA = CelebaA_Data(self.data_path, self.label_size)
            self.train_dataset, self.train_label = self.CelebaA.Train_Data_Label()
            self.test_dataset, self.test_label = self.CelebaA.Test_Data_Label()
        elif self.data_type == 'RafD':
            self.RafD = RafD_Data(self.data_path, self.label_size)
            self.train_dataset, self.train_label = self.RafD.Data_Label()
        else:
            self.train_dataset = glob(os.path.join(os.getcwd(), self.data_path, '*.'+img_type))
            self.train_label = np.zeros(len(self.train_dataset))

    def PFD_image_processing(self, filename, label):
        img = self.image_read(filename, self.img_type, self.channels, gt_flag = 1)
        img = tf.reshape(img, [25, 23, self.channels])
        img = 1 - (tf.cast(img, tf.float32) / 255.)

        if self.augment_flag :
            img = tf.cond(pred=tf.greater_equal(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), 0.5),
                          true_fn=lambda: PFD_augmentation(img), false_fn=lambda: img)
        
        return img, label
    
    def RafD_image_processing(self, filename, label):
        img = self.image_read(filename, self.img_type, self.channels, gt_flag = 1)
        img = tf.image.resize(img, [self.img_h, self.img_w])
        img = tf.reshape(img, [self.img_h, self.img_w, self.channels])
        img = tf.cast(img, tf.float32) / 255
        if self.augment_flag :
            img = tf.cond(pred=tf.greater_equal(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), 0.5),
                          true_fn=lambda: RafD_augmentation(img),  false_fn=lambda: img)
                
        return img, label
    
    def image_processing(self, filename, label):
        img = self.image_read(filename, self.img_type, self.channels, gt_flag = 1)
        img = tf.image.resize(img, [self.img_h, self.img_w])
        img = tf.reshape(img, [self.img_h, self.img_w, self.channels])
        img = tf.cast(img, tf.float32) / 255
        if self.augment_flag :
            img = tf.cond(pred=tf.greater_equal(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), 0.5),
                          true_fn=lambda: augmentation(img), false_fn=lambda: img)
                
        return img, label

    def image_read(self, filename, img_type, channel, gt_flag = 0):
        x = tf.io.read_file(filename)
        if img_type == 'jpg':
            x_decode = tf.image.decode_jpeg(x, channels=channel)
        if img_type == 'png':
            x_decode = tf.image.decode_png(x, channels=channel)
        if img_type == 'bmp':
            x_decode = tf.image.decode_bmp(x)
            if channel == 1 :
                x_decode = tf.image.rgb_to_grayscale(x_decode)
        
        return x_decode
        
 
    def set_value(self, matrix, x, y, val):
        w = int(matrix.get_shape()[0])
        h = int(matrix.get_shape()[1])
        mult_matrix = tf.compat.v1.sparse_tensor_to_dense(tf.SparseTensor(indices=[[x, y]], values=[0.], dense_shape=[w, h])) + 1.0
        diff_matrix = tf.compat.v1.sparse_tensor_to_dense(tf.SparseTensor(indices=[[x, y]], values=[val], dense_shape=[w, h]))
        matrix = tf.multiply(matrix, mult_matrix) 
        matrix = matrix + diff_matrix
        return matrix

 
class PFD_Data:

    def __init__(self, data_path, label_size) :
        self.data_path = os.path.join(data_path, 'Pixel')
        self.lines = open(os.path.join(data_path, 'label.txt'), 'r').readlines()
        self.label_size = label_size
    
    def Data_Label(self):
        head_label=[0.0]*(self.label_size-8)
        Motion_label = [0.0]*8
        images = []
        labels = []
        for i, line in enumerate(self.lines) :
            head_label=[0.0]*(self.label_size-8)
            Motion_label = [0.0]*8
            split = line.split()
            filename = os.path.join(self.data_path, split[0])
            Mouth_value = int(split[1])
            Motion_value = int(split[2])
            Glasses_value = int(split[3])
            images.append(filename)
            head_label[0] = Mouth_value
            head_label[1] = Glasses_value
            Motion_label[Motion_value] = 1.
            labels.append(head_label+Motion_label)
        return images, labels

class CelebaA_Data:

    def __init__(self, data_path, label_size) :
        self.data_path = data_path
        self.lines = open(os.path.join(data_path, 'Anno/list_attr_celeba_front.txt'), 'r').readlines()
        self.label_size = label_size
        self.attr = ['Mouth_Slightly_Open','Eyeglasses']
        all_attr_names = self.lines[1].split()
        self.attr2idx = {}
        self.idx2attr = {}
        for i, attr_name in enumerate(all_attr_names) :
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        self.lines = self.lines[2:]
        self.idx = []
        for l in self.attr:
            self.idx.append(self.attr2idx[l])
    
    def Train_Data_Label(self):
        images = []
        labels = []
        for i, line in enumerate(self.lines[:-2000]) :
            label=[0.0]*self.label_size
            split = line.split()
            ig_path = os.path.join(self.data_path, 'celeba_64',split[0])
            images.append(ig_path)
            label_value = split[1:]
            label[0] = (int(label_value[int(self.idx[0])])+1)*0.5
            label[1] = (int(label_value[int(self.idx[1])])+1)*0.5
            labels.append(label)
        return images, labels
    
    def Test_Data_Label(self):
        images = []
        labels = []
        for i, line in enumerate(self.lines[-2000:]) :
            label=[0.0]*self.label_size
            split = line.split()
            ig_path = os.path.join(self.data_path, 'celeba_64',split[0])
            images.append(ig_path)
            label_value = split[1:]
            label[0] = (int(label_value[int(self.idx[0])])+1)*0.5
            label[1] = (int(label_value[int(self.idx[1])])+1)*0.5
            labels.append(label)
        return images, labels

class RafD_Data:

    def __init__(self, data_path, label_size) :
        assert label_size>=8, 'wrong label_size'
        self.img_dataset=glob(os.path.join(os.getcwd(), data_path, '*.*'))
        self.label_size = label_size
        self.Motion_Mode={'neutral':0,
                       'angry':1,
                       'contemptuous':2,
                       'disgusted':3,
                       'fearful':4,
                       'happy':5,
                       'sad':6,
                       'surprised':7}#'frontal','left','right'
    
    def Data_Label(self):
        head_label=[0.0]*(self.label_size-8)
        Motion_label = [0.0]*8
        labels=[]
        for image in self.img_dataset:
            Motion_label = [0.0]*8  
            S=image.split('_')
            GenderStr=S[3]
            MotionStr=S[4]
            EyeFrontStr=S[5].split('-')[0]
            Motion_idx = self.Motion_Mode[MotionStr]
            Motion_label[Motion_idx] = 1.
            labels.append(head_label+Motion_label)
        return self.img_dataset, labels
    
def one_hot(batch_size, mask_size, location):
    l = tf.constant([location])
    m = tf.one_hot(l,mask_size,1.,0.)
    m = tf.tile(m,[batch_size,1])
    return m
    
def load_test_data(image_path, size_h=64, size_w=64):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    img = cv2.resize(img, (size_h, size_w))
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/255. # 0 ~ 1
    return x

def augmentation(image):
    seed = random.randint(0, 2 ** 31 - 1)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_brightness(image,max_delta=0.2)
    image = tf.image.random_contrast(image, 0.5, 1.5)
    image = tf.image.random_saturation(image, 0, 1)
    image = tf.clip_by_value(image,0.,1.)
    return image

def RafD_augmentation(image):
    seed = random.randint(0, 2 ** 31 - 1)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_brightness(image,max_delta=0.2)
    image = tf.image.random_contrast(image, 0.5, 1.5)
    image = tf.image.random_saturation(image, 0, 1)
    image = tf.clip_by_value(image,0.,1.)
    return image

def MaskHQ_augmentation(image):
    image = tf.image.random_brightness(image,max_delta=0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0, 1)
    image = tf.clip_by_value(image,0.,1.)
    return image

def PFD_augmentation(image):
    seed = random.randint(0, 2 ** 31 - 1)
    image = tf.image.random_flip_left_right(image, seed=seed)
    return image

def save_images(images, size, image_path, range_type=2, show=True):
    return imsave(inverse_transform(images, range_type), size, image_path, show=show)

def inverse_transform(images, range_type):
    if range_type==1:
        return images * 255.0
    if range_type==2:
        return ((images+1.) / 2) * 255.0


def imsave(images, size, path, show=False):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)
    if show:
        cv2.imshow("image",images)
        cv2.waitKey(1)

    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def return_images(images, size) :
    x = merge(images, size)

    return x

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')

def automatic_gpu_usage() :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)