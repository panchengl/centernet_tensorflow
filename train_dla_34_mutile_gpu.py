import os
import numpy as np
import tensorflow as tf
import cv2
import math
import time
import shutil
import mutile_gpu_cfg as cfg
from tqdm import tqdm
from CenterNet_dla34 import CenterNet
from utils.generator_mgpu import get_data
from utils.utils import AverageMeter, parse_gt_rec, post_process, get_preds_gpu
from utils.decode import decode
from net.resnet import load_weights



def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def sum_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    sum_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_sum(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        sum_grads.append(grad_and_var)
    return sum_grads



def train():
    # define dataset
    num_train_imgs = len(open(cfg.train_data_file, 'r').readlines())
    num_train_batch = int(math.ceil(float(num_train_imgs) / cfg.batch_size))
    num_test_imgs = len(open(cfg.test_data_file, 'r').readlines())
    num_test_batch = int(math.ceil(float(num_test_imgs) / 1))

    train_dataset = tf.data.TextLineDataset(cfg.train_data_file)
    train_dataset = train_dataset.shuffle(num_train_imgs)
    train_dataset = train_dataset.batch(cfg.batch_size)
    train_dataset = train_dataset.map(lambda x: tf.py_func(get_data, inp=[x, True],
                                                           Tout=[tf.float32, tf.float32, tf.float32, tf.float32,
                                                                 tf.float32, tf.float32, tf.int32, tf.int32]),
                                      num_parallel_calls=6)
    train_dataset = train_dataset.prefetch(3)

    test_dataset = tf.data.TextLineDataset(cfg.test_data_file)
    test_dataset = test_dataset.batch(1)
    test_dataset = test_dataset.map(lambda x: tf.py_func(get_data, inp=[x, False],
                                                         Tout=[tf.float32, tf.float32, tf.float32, tf.float32,
                                                               tf.float32, tf.float32, tf.int32, tf.int32]),
                                    num_parallel_calls=1)
    test_dataset = test_dataset.prefetch(1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    trainset_init_op = iterator.make_initializer(train_dataset)
    testset_init_op = iterator.make_initializer(test_dataset)

    input_data, hm, wh, reg, reg_mask, ind, img_size, id = iterator.get_next()

    batch_input_data = []
    batch_hm  = []
    batch_wh = []
    batch_reg = []
    batch_reg_mask = []
    batch_ind = []
    batch_img_size = []
    batch_id= []

    for i in range(cfg.NUM_GPU):
        start = i * (cfg.batch_size // cfg.NUM_GPU)
        end = (i + 1) * (cfg.batch_size // cfg.NUM_GPU)

        single_input_data= input_data[start:end, :, :, :]
        single_hm = hm[start:end, :, :, :]
        single_wh = wh[start:end, :, :]
        single_reg = reg[start:end, :, :]
        single_reg_mask = reg_mask[start:end, :]
        single_ind = ind[start:end, : ]
        single_img_size = img_size[start:end:, :]
        single_id = id[start:end, : ]

        batch_input_data.append(single_input_data)
        batch_hm.append(single_hm)
        batch_wh.append(single_wh)
        batch_reg.append(single_reg)
        batch_reg_mask.append(single_reg_mask)
        batch_ind.append(single_ind)
        batch_img_size.append(single_img_size)
        batch_id.append(single_id)

        batch_input_data[i].set_shape([None, None, None, 3])
        batch_hm[i].set_shape([None, None, None, None])
        batch_wh[i].set_shape([None, None, None])
        batch_reg[i].set_shape([None, None, None])
        batch_reg_mask[i].set_shape([None, None])
        batch_ind[i].set_shape([None, None])
        batch_img_size[i].set_shape([None, None])
        batch_id[i].set_shape([None])

    # difine model and loss
    with tf.device('/cpu:0'):
        tower_grads = []
        hm_loss = []
        wh_loss = []
        reg_loss = []
        total_loss = []
        hm_pred_list = []
        wh_pred_list = []
        reg_pred_list = []
        pred_det = []
        # training flag
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        # with tf.variable_scope(tf.get_variable_scope()):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
            for i in range(cfg.NUM_GPU):
                print("current gpu is", i)
                with tf.device('/gpu:%d' % i):
                    model = CenterNet(batch_input_data[i], is_training, "dla_34")
                    hm_pred = model.pred_hm
                    wh_pred = model.pred_wh
                    reg_pred = model.pred_reg
                    hm_pred_list.append(hm_pred)
                    wh_pred_list.append(wh_pred)
                    reg_pred_list.append(reg_pred)
                    det = decode(hm_pred_list[i], wh_pred_list[i], reg_pred_list[i], K=cfg.max_objs)
                    pred_det.append(det)
                    with tf.variable_scope('loss'):
                        l2_loss = tf.losses.get_regularization_loss()
                        # hm_loss[i], wh_loss[i], reg_loss[i] = model.compute_loss(batch_hm[i], batch_wh[i], batch_reg[i], batch_reg_mask[i], batch_ind[i])
                        hm_loss_single, wh_loss_single, reg_loss_single = model.compute_loss(batch_hm[i], batch_wh[i], batch_reg[i], batch_reg_mask[i], batch_ind[i])
                        hm_loss.append(hm_loss_single)
                        wh_loss.append(wh_loss_single)
                        reg_loss.append(reg_loss_single)
                        total_loss_single = hm_loss[i] + wh_loss[i] + reg_loss[i] + l2_loss
                        total_loss.append(total_loss_single)
                    # define train op
                    global_step = tf.Variable(0, trainable=False)
                    if cfg.lr_type == "exponential":
                        learning_rate = tf.train.exponential_decay(cfg.lr,
                                                                   global_step,
                                                                   cfg.lr_decay_steps,
                                                                   cfg.lr_decay_rate,
                                                                   staircase=True)
                    elif cfg.lr_type == "piecewise":
                        learning_rate = tf.train.piecewise_constant(global_step, cfg.lr_boundaries, cfg.lr_piecewise)
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    with tf.control_dependencies(update_ops):
                        grads = optimizer.compute_gradients(total_loss[i])
                        clip_grad_var = [gv if gv[0] is None else [
                            tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in grads]
                        tower_grads.append(clip_grad_var)
            last_loss = tf.reduce_mean(total_loss)
            if len(tower_grads) > 1:
                clip_grad_var = sum_gradients(tower_grads)
            else:
                clip_grad_var = tower_grads[0]
            train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                with tf.name_scope('summary'):
                    tf.summary.scalar("learning_rate", learning_rate)
                    tf.summary.scalar("hm_loss", tf.reduce_mean(hm_loss))
                    tf.summary.scalar("wh_loss", tf.reduce_mean(wh_loss))
                    tf.summary.scalar("reg_loss", tf.reduce_mean(reg_loss))
                    tf.summary.scalar("total_loss", tf.reduce_mean(total_loss))

                    logdir = "./log_dla/"
                    if os.path.exists(logdir): shutil.rmtree(logdir)
                    os.mkdir(logdir)
                    write_op = tf.summary.merge_all()
                    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
                sess.run(tf.global_variables_initializer())
                if cfg.pre_train:
                    saver.restore(sess, './checkpoint/centernet_train_epoch_loss=313.7357.ckpt-6')
                for epoch in range(1, 1 + cfg.epochs):
                    pbar = tqdm(range(num_train_batch))
                    train_epoch_loss, test_epoch_loss = [], []
                    sess.run(trainset_init_op)
                    for i in pbar:
                        _, summary, train_step_loss, global_step_val = sess.run(
                            [train_op, write_op, last_loss, global_step], feed_dict={is_training: True})
                        train_epoch_loss.append(train_step_loss)
                        summary_writer.add_summary(summary, global_step_val)
                        pbar.set_description("train loss: %.2f" % train_step_loss)
                    # sess.run(testset_init_op)
                    # for j in range(num_test_batch ):
                    #     test_step_loss = sess.run(last_loss, feed_dict={is_training: False})
                    #     test_epoch_loss.append(test_step_loss)
                    # train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
                    train_epoch_loss = np.mean(train_epoch_loss)
                    ckpt_file = "./checkpoint/centernet_train_epoch_loss=%.4f.ckpt" % train_epoch_loss
                    log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    print("=> Epoch: %2d Time: %s Train loss: %.2f  Saving %s ..."
                          % (epoch, log_time, train_epoch_loss, ckpt_file))
                    saver.save(sess, ckpt_file, global_step=epoch)

                    if epoch % cfg.eval_epoch == 0 and epoch > 0 :
                        print("begining test")
                        sess.run(testset_init_op)
                        val_preds = []
                        for j in tqdm(range(num_test_batch)):
                            detections, _batch_img_size, _batch_id = sess.run([pred_det[0], img_size, id ], feed_dict={is_training: False})
                            # print("detecttiion is", detections)
                            # print("_batch_img_size is", _batch_img_size)
                            # print("id is", _batch_id)
                            ori_h = _batch_img_size[0][1]
                            ori_w = _batch_img_size[0][0]
                            detect_post = post_process(detections, (ori_h, ori_w), [cfg.input_image_h, cfg.input_image_w],
                                                       cfg.down_ratio, cfg.score_threshold)
                            id_test = _batch_id[0]
                            detect_per_img = get_preds_gpu(detect_post, id_test)
                            val_preds.extend(detect_per_img)
                        rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
                        info = ""
                        gt_dict = parse_gt_rec(cfg.test_data_file, [cfg.input_image_h, cfg.input_image_w], cfg.letterbox_resize)
                        for ii in range(cfg.num_classes):
                            from utils.utils import voc_eval
                            npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=cfg.score_threshold,
                                                               use_07_metric=cfg.use_voc_07_metric)
                            info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
                            rec_total.update(rec, npos)
                            prec_total.update(prec, nd)
                            ap_total.update(ap, 1)
                        mAP = ap_total.average
                        info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total.average,
                                                                                               prec_total.average, mAP)
                        print(info)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
    train()
