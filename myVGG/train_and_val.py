import os
import tensorflow as tf
import numpy as np
import math

import input_data
import model
import tools

from tensorflow.python.framework.graph_util import convert_variables_to_constants

IMG_W, IMG_H, IMG_CHANNELS, NUM_CLASSES, BATCH_SIZE = 32, 32, 3, 10, 32
learning_rate, MAX_STEP, IS_PRETRAIN = 0.001, 15000, True


def train(retrain = False):
    data_dir = '/home/rong/something_for_deep/cifar-10-batches-bin'
    npy_dir = '/home/rong/something_for_deep/vgg16.npy'
    train_log_dir = './logs/train'
    val_log_dir = './logs/val'

    train_image_batch, train_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=True,
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=True)
    val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                               is_train=False,
                                                               batch_size=BATCH_SIZE,
                                                               shuffle=False)

    #宣布图片batch和标签batch的占位符
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, IMG_CHANNELS])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, NUM_CLASSES])

    #宣布VGG16类型的变量
    vgg = model.VGG16()

    #宣布损失,精确度等关键节点
    logits = vgg.build(x, NUM_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)
    train_op2 = tools.optimize2(loss, learning_rate)

    saver = tf.train.Saver() #括号那个参数不知道是干什么的
    summary_op = tf.summary.merge_all()

    #初始化所有的variable,之前我看过另外一种写法,那种写法好像废弃了
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #从npy文件加载除了全连接之外,其他层的权重
    tools.load_with_skip(npy_dir, sess, ['fc6', 'fc7', 'fc8'])

    saver.restore(sess, './logs/train/model.ckpt-6000')
    output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['fc8/relu'])

    with tf.gfile.FastGFile('vgg_6000.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
    '''
    #下面的和多线程有关,暂时不懂
    coord = tf.train.Coordinator() #宣布线程管理器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) #线程负责把文件加入队列(input_data那个file队列)

    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    '''
    '''
    if retrain == False:
        print('Reading checkpoints')
        ckpt = tf.train.get_checkpoint_state(train_log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, './logs/train/model.ckpt-10000')
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            return
    
    saver.restore(sess, './logs/train/model.ckpt-10000')


    for step in range(50):
        train_images, train_labels = sess.run([train_image_batch, train_label_batch])
        _, train_loss, train_acc = sess.run([train_op2, loss, accuracy],
                                            feed_dict={x: train_images, y_: train_labels})
        print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
   
    saver.restore(sess, './logs/train/model.ckpt-14999')
    '''
    '''
    #下面的try语句可以当做模板使用
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            #运行计算节点,从计算节点中得到真实的image,label
            train_images, train_labels = sess.run([train_image_batch, train_label_batch])

            #运行损失, 精确度计算节点, 得到具体数值
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: train_images, y_: train_labels})

            #每到50步或者最后一步就当前batch的损失值大小和准确度大小
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                #summary_str = sess.run(summary_op)
                #tra_summary_writer.add_summary(summary_str, step)

            #每到200步或者最后一步就从测试集取一个batch, 计算损失值大小和准确度
            if step % 200 == 0 or (step + 1) == MAX_STEP:

                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x: val_images, y_: val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))

                #summary_str = sess.run(summary_op)
                #val_summary_writer.add_summary(summary_str, step)

            #每到2000步就保存一次
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                if step == 0:
                    continue
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    '''
    sess.close()


def val():
    num_test = 10000
    data_dir = '/home/rong/something_for_deep/cifar-10-batches-bin'
    train_log_dir = './logs/train'

    image_batch, label_batch = input_data.read_cifar10(data_dir,
                                                       is_train=False,
                                                       batch_size=BATCH_SIZE,
                                                        shuffle=False)
    vgg16 = model.VGG16()
    logits = vgg16.build(image_batch, NUM_CLASSES, False)
    saver = tf.train.Saver()
    correct_per_batch = tools.num_correct_prediction(logits, label_batch)

    with tf.Session() as sess:
        print('Reading checkpoints')
        ckpt = tf.train.get_checkpoint_state(train_log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            return

        saver.restore(sess, './logs/train/model.ckpt-8000')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            print('\nEvaluating......')
            num_step = int(math.floor(num_test / BATCH_SIZE))
            num_sample = num_step * BATCH_SIZE
            step = 0
            total_correct = 0
            while step < num_step and not coord.should_stop():
                batch_correct = sess.run(correct_per_batch)
                total_correct += np.sum(batch_correct)
                step += 1
                if step % 10 == 0:
                    print('Testing samples: %d' % (step*BATCH_SIZE))
                    print('Correct predictions: %d' % total_correct)
                    print('Average accuracy: %.2f%%' % (total_correct / (step*BATCH_SIZE)))
            print('Total testing samples: %d' % num_sample)
            print('Total correct predictions: %d' % total_correct)
            print('Average accuracy: %.2f%%' % (100 * total_correct / num_sample))
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def train2(retrain=False):
    data_dir = '/home/rong/something_for_deep/cifar-10-batches-bin'

    train_image_batch, train_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=True,
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=True)


    # 宣布图片batch和标签batch的占位符
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, IMG_CHANNELS], name='X')
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, NUM_CLASSES])


    with open('vgg_6000.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        logits = tf.import_graph_def(graph_def,
                                     input_map={'X': x},
                                     return_elements=['fc8/relu:0'])



    # 宣布损失,精确度等关键节点
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)


    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator() #宣布线程管理器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) #线程负责把文件加入队列(input_data那个file队列)

    train_images, train_labels = sess.run([train_image_batch, train_label_batch])
    loss2, accuracy = sess.run([loss, accuracy], feed_dict={x:train_images, y_:train_labels})
    print(loss2, accuracy)

    coord.request_stop()
    coord.join(threads)

    '''
    if retrain == False:
        print('Reading checkpoints')
        ckpt = tf.train.get_checkpoint_state(train_log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, './logs/train/model.ckpt-10000')
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            return

    saver.restore(sess, './logs/train/model.ckpt-10000')


    for step in range(50):
        train_images, train_labels = sess.run([train_image_batch, train_label_batch])
        _, train_loss, train_acc = sess.run([train_op2, loss, accuracy],
                                            feed_dict={x: train_images, y_: train_labels})
        print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))

    saver.restore(sess, './logs/train/model.ckpt-14999')
    '''
    '''
    #下面的try语句可以当做模板使用
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            #运行计算节点,从计算节点中得到真实的image,label
            train_images, train_labels = sess.run([train_image_batch, train_label_batch])

            #运行损失, 精确度计算节点, 得到具体数值
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: train_images, y_: train_labels})

            #每到50步或者最后一步就当前batch的损失值大小和准确度大小
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                #summary_str = sess.run(summary_op)
                #tra_summary_writer.add_summary(summary_str, step)

            #每到200步或者最后一步就从测试集取一个batch, 计算损失值大小和准确度
            if step % 200 == 0 or (step + 1) == MAX_STEP:

                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x: val_images, y_: val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))

                #summary_str = sess.run(summary_op)
                #val_summary_writer.add_summary(summary_str, step)

            #每到2000步就保存一次
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                if step == 0:
                    continue
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    '''
    sess.close()

def main():
    train2()

main()
