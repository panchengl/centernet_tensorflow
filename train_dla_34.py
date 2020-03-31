import os
import numpy as np
import tensorflow as tf
import cv2
import math
import time
import shutil
import cfg
from tqdm import tqdm
from CenterNet_dla34 import CenterNet
from utils.generator import get_data
from utils.utils import AverageMeter, parse_gt_rec, post_process, get_preds_gpu
from net.resnet import load_weights


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

    input_data, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind, batch_img_size, batch_id = iterator.get_next()
    input_data.set_shape([None, None, None, 3])
    batch_hm.set_shape([None, None, None, None])
    batch_wh.set_shape([None, None, None])
    batch_reg.set_shape([None, None, None])
    batch_reg_mask.set_shape([None, None])
    batch_ind.set_shape([None, None])
    batch_img_size.set_shape([None, None])
    batch_id.set_shape([None])

    # training flag
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    # difine model and loss
    model = CenterNet(input_data, is_training, "dla_34")
    hm = model.pred_hm
    wh = model.pred_wh
    reg = model.pred_reg
    from utils.decode import decode
    det = decode(hm, wh, reg, K=cfg.max_objs)

    with tf.variable_scope('loss'):
        # hm_loss, wh_loss, reg_loss = model.compute_loss(batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind)
        hm_loss, wh_loss, reg_loss = model.compute_loss_pcl(batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind)
        total_loss = hm_loss + wh_loss + reg_loss

    # define train op
    if cfg.lr_type == "CosineAnnealing":
        learning_rate = 0.0001
        global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
        # warmup_steps = tf.constant(cfg.warm_up_epochs * num_train_batch, dtype=tf.float64, name='warmup_steps')
        # train_steps = tf.constant(cfg.epochs * num_train_batch, dtype=tf.float64, name='train_steps')
        # learning_rate = tf.cond(
        #     pred=global_step < warmup_steps,
        #     true_fn=lambda: global_step / warmup_steps * cfg.init_lr,
        #     false_fn=lambda: cfg.end_lr + 0.5 * (cfg.init_lr - cfg.end_lr) *
        #                      (1 + tf.cos(
        #                          (global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
        # )
        global_step_update = tf.assign_add(global_step, 1.0)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([optimizer, global_step_update]):
                train_op = tf.no_op()

    else:
        global_step = tf.Variable(0, trainable=False)
        if cfg.lr_type == "exponential":
            learning_rate = 0.0001
            # learning_rate = tf.train.exponential_decay(cfg.lr,
            #                                            global_step,
            #                                            cfg.lr_decay_steps,
            #                                            cfg.lr_decay_rate,
            #                                            staircase=True)
        elif cfg.lr_type == "piecewise":
            learning_rate = 0.0001
            # learning_rate = tf.train.piecewise_constant(global_step, cfg.lr_boundaries, cfg.lr_piecewise)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    with tf.Session() as sess:
        with tf.name_scope('summary'):
            tf.summary.scalar("learning_rate", learning_rate)
            tf.summary.scalar("hm_loss", hm_loss)
            tf.summary.scalar("wh_loss", wh_loss)
            tf.summary.scalar("reg_loss", reg_loss)
            tf.summary.scalar("total_loss", total_loss)

            logdir = "./log_dla/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            write_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

        # train
        sess.run(tf.global_variables_initializer())
        # if cfg.dla_pretrain:
        #     saver.restore(sess, './checkpoint/centernet_test_loss=3.1386.ckpt-79')
        for epoch in range(1, 1 + cfg.epochs):
            pbar = tqdm(range(num_train_batch))
            train_epoch_loss, test_epoch_loss = [], []
            sess.run(trainset_init_op)
            for i in pbar:
                _, summary, train_step_loss, global_step_val, _hm_loss, _wh_loss, _reg_loss = sess.run(
                    [train_op, write_op, total_loss, global_step, hm_loss, wh_loss, reg_loss],
                    feed_dict={is_training: True})
                train_epoch_loss.append(train_step_loss)
                summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" % train_step_loss)
                if i % 20 == 0:
                    print("train loss: %.2f hm_loss: %.2f wh_loss:%2f reg_loss:%2f learning_rate:%f" % (train_step_loss, _hm_loss, _wh_loss, _reg_loss, learning_rate) )
            print("begining test")
            sess.run(testset_init_op)
            val_preds = []
            # for j in range(num_test_batch ):
            #     test_step_loss = sess.run(total_loss, feed_dict={is_training: False})
            #     test_epoch_loss.append(test_step_loss)
            train_epoch_loss = np.mean(train_epoch_loss)
            # train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/centernet_train_loss=%.4f.ckpt" % train_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, ckpt_file))
            saver.save(sess, ckpt_file, global_step=epoch)

            if epoch % cfg.eval_epoch == 0 and epoch > 0:
                sess.run(testset_init_op)
                for j in range(num_test_batch):
                    detections, _batch_img_size, _batch_id = sess.run([det, batch_img_size, batch_id], feed_dict={is_training: False})
                    ori_h = _batch_img_size[0][1]
                    ori_w = _batch_img_size[0][0]
                    detect_post = post_process(detections, (ori_h, ori_w), [cfg.input_image_h, cfg.input_image_w],
                                               cfg.down_ratio, cfg.score_threshold)
                    id = _batch_id[0]
                    detect_per_img = get_preds_gpu(detect_post, id)
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
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    train()
