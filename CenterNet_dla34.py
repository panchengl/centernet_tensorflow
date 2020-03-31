import tensorflow as tf
import cfg
import loss
from net import resnet
from net.layers import _conv, upsampling
from net import dla_net
import numpy as np


class CenterNet():
    def __init__(self, inputs, is_training, net_name):
        self.is_training = is_training
        self.net_name = net_name
        try:
            self.pred_hm, self.pred_wh, self.pred_reg = self._build_model(inputs)
        except:
            raise NotImplementedError("Can not build up centernet network!")

    def _build_model(self, inputs):
        with tf.variable_scope('dla'):
            assert self.net_name in ["dla_34"]
            if self.net_name == "dla_34":
                features = dla_net.DLAnet(is_training=self.is_training).forward(inputs)
        with tf.variable_scope('detector'):
            hm = _conv(features, 64, [3, 3], is_training=self.is_training)
            hm = tf.layers.conv2d(hm, cfg.num_classes, 1, 1, padding='valid', activation=tf.nn.sigmoid,
                                  bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm')

            wh = _conv(features, 64, [3, 3], is_training=self.is_training)
            wh = tf.layers.conv2d(wh, 2, 1, 1, padding='valid', activation=None, name='wh')

            reg = _conv(features, 64, [3, 3], is_training=self.is_training)
            reg = tf.layers.conv2d(reg, 2, 1, 1, padding='valid', activation=None, name='reg')
        return hm, wh, reg

    def compute_loss(self, true_hm, true_wh, true_reg, reg_mask, ind):
        hm_loss = loss.focal_loss(self.pred_hm, true_hm)
        wg_loss = 0.05 * loss.reg_l1_loss(self.pred_wh, true_wh, ind, reg_mask)
        reg_loss = loss.reg_l1_loss(self.pred_reg, true_reg, ind, reg_mask)
        return hm_loss, wg_loss, reg_loss

    def compute_loss_pcl(self, true_hm, true_wh, true_reg, reg_mask, ind):
        hm_loss = loss.focal_loss_pcl(self.pred_hm, true_hm)
        wg_loss = 0.05 * loss.reg_l1_loss(self.pred_wh, true_wh, ind, reg_mask)
        reg_loss = loss.reg_l1_loss(self.pred_reg, true_reg, ind, reg_mask)
        return hm_loss, wg_loss, reg_loss