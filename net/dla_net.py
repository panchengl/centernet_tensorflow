import tensorflow as tf
import numpy as np
from net.layers import _conv
import cfg

class DLAnet():
    def __init__(self, is_training=False, use_bn=True):
        self.is_training = is_training
        self.use_bn = use_bn
        self.inplanes = 64
        self.batch_size = cfg.batch_size
        self.data_shape = [cfg.input_image_w, cfg.input_image_h, 3]
        self._conv_bn_activation = _conv
        self.data_format = 'channels_last'

    def _dla_generator(self, inputs, filters, levels, stack_block_fn):
        if levels == 1:
            block1 = stack_block_fn(inputs, filters)
            block2 = stack_block_fn(block1, filters)
            aggregation = block1 + block2
            # aggregation = self._conv_bn_activation(aggregation, filters, 3, 1, is_training=self.is_training, use_bn=self.use_bn)
            aggregation = self._conv_bn_activation(aggregation, filters, 3, 1, is_training=self.is_training, use_bn=self.use_bn)
        else:
            block1 = self._dla_generator(inputs, filters, levels - 1, stack_block_fn)
            block2 = self._dla_generator(block1, filters, levels - 1, stack_block_fn)
            aggregation = block1 + block2
            aggregation = self._conv_bn_activation(aggregation, filters, 3, 1, is_training=self.is_training, use_bn=self.use_bn)
        return aggregation

    def _max_pooling(self, bottom, pool_size, strides, name=None):
        return tf.layers.max_pooling2d(inputs=bottom, pool_size=pool_size, strides=strides, padding='same', name=name)

    def forward(self, inputs): # layers and channels  [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512]
        with tf.variable_scope('backone'):
            with tf.variable_scope('base_layer'):
                base_layer = self._conv_bn_activation(inputs=inputs, filters=16, kernel_size=7, strides=1, is_training=self.is_training, use_bn=self.use_bn)

            with tf.variable_scope('level0'):
                level0 = self._conv_bn_activation(inputs=base_layer, filters=16, kernel_size=3, strides=1, is_training=self.is_training, use_bn=self.use_bn)

            with tf.variable_scope('level1'):
                level1 = self._conv_bn_activation(inputs=level0, filters=32, kernel_size=3,strides=2, is_training=self.is_training, use_bn=self.use_bn)

            with tf.variable_scope('level2'):
                dla_stage3 = self._dla_generator(level1, 64, 1, self._basic_block)
                dla_stage3 = self._max_pooling(dla_stage3, 2, 2)

            with tf.variable_scope('level3'):
                dla_stage4 = self._dla_generator(dla_stage3, 128, 2, self._basic_block)
                residual = self._conv_bn_activation(dla_stage3, 128, 1, 1, is_training=self.is_training, use_bn=self.use_bn)
                residual = self._avg_pooling(residual, 2, 2)
                dla_stage4 = self._max_pooling(dla_stage4, 2, 2)
                dla_stage4 = dla_stage4 + residual

            with tf.variable_scope('level4'):
                dla_stage5 = self._dla_generator(dla_stage4, 256, 2, self._basic_block)
                residual = self._conv_bn_activation(dla_stage4, 256, 1, 1, is_training=self.is_training, use_bn=self.use_bn)
                residual = self._avg_pooling(residual, 2, 2)
                dla_stage5 = self._max_pooling(dla_stage5, 2, 2)
                dla_stage5 = dla_stage5 + residual

            with tf.variable_scope('level5'):
                dla_stage6 = self._dla_generator(dla_stage5, 512, 1, self._basic_block)
                residual = self._conv_bn_activation(dla_stage5, 512, 1, 1,is_training=self.is_training, use_bn=self.use_bn)
                residual = self._avg_pooling(residual, 2, 2, )
                dla_stage6 = self._max_pooling(dla_stage6, 2, 2)
                dla_stage6 = dla_stage6 + residual

            # with tf.variable_scope('upsampling'):
                dla_stage6 = self._conv_bn_activation(dla_stage6, 256, 1, 1, is_training=self.is_training, use_bn=self.use_bn)
                dla_stage6_5 = self._dconv_bn_activation(dla_stage6, 256, 4, 2)
                dla_stage6_4 = self._dconv_bn_activation(dla_stage6_5, 256, 4, 2)
                dla_stage6_3 = self._dconv_bn_activation(dla_stage6_4, 256, 4, 2)

                dla_stage5 = self._conv_bn_activation(dla_stage5, 256, 1, 1, is_training=self.is_training, use_bn=self.use_bn)
                dla_stage5_4 = self._conv_bn_activation(dla_stage5+dla_stage6_5, 256, 3, 1, is_training=self.is_training, use_bn=self.use_bn)
                dla_stage5_4 = self._dconv_bn_activation(dla_stage5_4, 256, 4, 2)
                dla_stage5_3 = self._dconv_bn_activation(dla_stage5_4, 256, 4, 2)

                dla_stage4 = self._conv_bn_activation(dla_stage4, 256, 1, 1, is_training=self.is_training, use_bn=self.use_bn)
                dla_stage4_3 = self._conv_bn_activation(dla_stage4+dla_stage5_4+dla_stage6_4, 256, 3, 1, is_training=self.is_training, use_bn=self.use_bn)
                dla_stage4_3 = self._dconv_bn_activation(dla_stage4_3, 256, 4, 2)

                features = self._conv_bn_activation(dla_stage6_3+dla_stage5_3+dla_stage4_3, 256, 3, 1, is_training=self.is_training, use_bn=self.use_bn)
                features = self._conv_bn_activation(features, 256, 1, 1, is_training=self.is_training, use_bn=self.use_bn)
        return features

    def _basic_block(self, bottom, filters):
        conv = self._conv_bn_activation(bottom, filters, 3, 1, is_training=self.is_training, use_bn=self.use_bn)
        conv = self._conv_bn_activation(conv, filters, 3, 1, is_training=self.is_training, use_bn=self.use_bn)
        axis = 3 if self.data_format == 'channels_last' else 1
        input_channels = tf.shape(bottom)[axis]
        shutcut = tf.cond(
            tf.equal(input_channels, filters),
            lambda: bottom,
            lambda: self._conv_bn_activation(bottom, filters, 1, 1, is_training=self.is_training, use_bn=self.use_bn)
        )
        return conv + shutcut

    def _dconv_bn_activation(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        conv = tf.layers.conv2d_transpose( inputs=bottom, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', data_format=self.data_format)
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    # def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
    #     conv = tf.layers.conv2d(
    #         inputs=bottom,
    #         filters=filters,
    #         kernel_size=kernel_size,
    #         strides=strides,
    #         padding='same',
    #         data_format=self.data_format
    #     )
    #     bn = self._bn(conv)
    #     if activation is not None:
    #         return activation(bn)
    #     else:
    #         return bn

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _avg_pooling(self, bottom, pool_size, strides, name=None):
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

def load_weights(sess, path):
    pretrained = np.load(path, allow_pickle=True).item()
    for variable in tf.trainable_variables():
        for key in pretrained.keys():
            key2 = variable.name.rstrip(':0')
            if (key == key2):
                sess.run(tf.assign(variable, pretrained[key]))


def _dla34(**kwargs):
    model = DLAnet(**kwargs)
    return model


def dla_34(**kwargs):
    return _dla34( **kwargs)


if __name__ == '__main__':
    inputs = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32)
    net = dla_34(is_training=True).forward(inputs)
    for variable in tf.trainable_variables():
        print(variable.name, variable.shape)