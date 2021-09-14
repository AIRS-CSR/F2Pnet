import tensorflow as tf


##################################################################################
# Pooling & Resize
##################################################################################

def crop_resize(x, size=[7, 9, 50, 46]):
    x = tf.image.crop_to_bounding_box(x, size[0], size[1], size[2], size[3])
    x = avg_poolling(x)
    return x

def resize_pad(x, h=[7, 7], w=[9, 9]):
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.pad(x, [[0, 0], h, w, [0,0]])
    return x

def max_poolling(x, size=2):
    ksize=[1, size, size, 1]
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='VALID')

def avg_poolling(x, size=2):
    ksize=[1, size, size, 1]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID')

##################################################################################
# Loss Function
##################################################################################

def cls_loss(logits, labels, type, label_size):
    logit1, logit2 = tf.split(logits, [label_size-8, 8], axis=-1)
    label1, label2 = tf.split(labels, [label_size-8, 8], axis=-1)
    if type==0:
        loss = CE_loss(labels=labels, logits=logits)
    if type==1:
        loss = CE_loss(labels=label1, logits=logit1)
    if type==2:
        loss = CE_loss(labels=label2, logits=logit2)

    return loss


def CE_loss(logits, labels):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    return loss


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss


def regularization_loss(model):
    loss = tf.nn.scale_regularization_loss(model.losses)

    return loss


##################################################################################
# GAN Loss Function
##################################################################################
def dis_loss(gan_type, real_logit, fake_logit):
    real_loss = 0
    fake_loss = 0
    if isinstance(fake_logit, list):
        fake_logit = tf.concat(fake_logit, axis=0)

    if gan_type == 'lsgan':
        real_loss = tf.reduce_mean(tf.square(real_logit - 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake_logit))

    if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

    if gan_type == 'hinge':
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))
        
    if gan_type.__contains__('wgan'):
        real_loss = -tf.reduce_mean(real_logit)
        fake_loss = tf.reduce_mean(fake_logit)
    
    if gan_type == 'nsl':
        real_loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(real_logit)))
        fake_loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(-fake_logit)))

    loss = real_loss + fake_loss

    return loss


def gen_loss(gan_type, fake_logit):
    fake_loss = 0
    real_loss = 0
    if isinstance(fake_logit, list):
        fake_logit = tf.concat(fake_logit, axis=0)

    if gan_type == 'lsgan':
        fake_loss = tf.reduce_mean(tf.square(fake_logit - 1.0))

    if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

    if gan_type == 'hinge':
        fake_loss = -tf.reduce_mean(fake_logit)

    if gan_type.__contains__('wgan'):
        fake_loss = -tf.reduce_mean(fake_logit)

    if gan_type == 'nsl':
        fake_loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(fake_logit)))

    loss = fake_loss + real_loss

    return loss

def gradient_penalty(discriminator, real_images, fake_images, lambda_val=10, gan_type='wgan-gp'):
    assert gan_type in ['wgan-gp', 'wgan-lp', 'dragan', 'wgan-div']

    if gan_type == 'dragan':
        eps = tf.random.uniform(shape=tf.shape(real_images), minval=0.0, maxval=1.0)
        _, x_var = tf.nn.moments(real_images, axes=[0, 1, 2, 3])
        x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region


        fake_images = real_images + 0.5 * x_std * eps

    if isinstance(fake_images,list):
        interpolated_images = []
        for f in fake_images:
            alpha = tf.random.uniform(shape=[tf.shape(real_images)[0], 1, 1, 1], minval=0.0, maxval=1.0)
            interpolated_images.append(real_images if gan_type == 'wgan-zc' else alpha * real_images + (1 - alpha)*f)
        interpolated_images = tf.concat(interpolated_images, axis=0)
    else:
        alpha = tf.random.uniform(shape=[tf.shape(real_images)[0], 1, 1, 1], minval=0.0, maxval=1.0)
        interpolated_images = real_images if gan_type == 'wgan-zc' else alpha * real_images + (1 - alpha)*fake_images

    with tf.GradientTape() as t:
        t.watch(interpolated_images)
        logit = discriminator(interpolated_images)[0]

    grad = t.gradient(logit, interpolated_images)
    grad_norm = tf.norm(tf.keras.layers.Flatten()(grad), axis=1) # l2 norm

    if gan_type == 'wgan-lp':
        gp = lambda_val * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.0)))
    elif gan_type == 'wgan-div':
        gp = 2 * tf.reduce_mean(tf.pow(grad_norm, 3))
    elif gan_type == 'wgan-zc':
        gp = lambda_val * tf.reduce_mean(tf.square(grad_norm))
    else :
        gp = lambda_val * tf.reduce_mean(tf.square(grad_norm - 1.0))

    return gp

##################################################################################
# Class function
##################################################################################

class get_weight(tf.keras.layers.Layer):
    def __init__(self, w_shape, w_init, w_regular, w_trainable):
        super(get_weight, self).__init__()

        self.w_shape = w_shape
        self.w_init = w_init
        self.w_regular = w_regular
        self.w_trainable = w_trainable
        # self.w_name = w_name

    def call(self, inputs=None, training=None, mask=None):
        return self.add_weight(shape=self.w_shape, dtype=tf.float32,
                               initializer=self.w_init, regularizer=self.w_regular,
                               trainable=self.w_trainable)


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name=self.name + '_u',
                                 dtype=tf.float32, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        # self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = None

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)

        self.layer.kernel = self.w / sigma

    def restore_weights(self):

        self.layer.kernel = self.w
