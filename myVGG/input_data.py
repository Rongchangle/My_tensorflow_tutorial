import tensorflow as tf
import numpy as np
import os


#从bin文件读取图片和标签的batch, csv等其他格式文件的读取也可以参考这里
#从网上下载的cifar10数据集读取图片的batch
def read_cifar10(data_dir, is_train, batch_size, shuffle):
    '''
    :param data_dir:
    :param is_train:
    :param batch_size:
    :param shuffle:
    :return:
    '''
    img_width, img_height, img_channels = 32, 32, 3
    label_bytes, image_bytes = 1, img_width*img_height*img_channels

    #训练集or测试集
    if is_train:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin'%i)for i in range(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]

    #filenames是文件名的list, 下面函数把全部文件打包为tf内部的queue类型, 之后的reader就从这个queue取目录
    filename_queue = tf.train.string_input_producer(filenames)

    #对bin(二进制)类型的文件, 定义下面的reader较好, 每次读取一段固定长度
    reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)

    #从文件名的queue取一个, 打开它进行一次读取, 注意返回的是tensor, 在run之前, 这些都只不过是花graph而已, 和定义神经网络节点是一样的
    key, value = reader.read(filename_queue)


    #value还是字符串, 转换为uint8的张量
    record_bytes = tf.decode_raw(value, tf.uint8)

    #cifar数据集格式是标签图片混在一起, 先标签后图片
    #函数原型 tf.slice(inputs,begin,size,name='')

    #取标签
    label = tf.slice(record_bytes, [0], [label_bytes])
    label = tf.cast(label, tf.int32)

    #取图片
    image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
    image_raw = tf.reshape(image_raw, [img_channels, img_height, img_width])
    image = tf.transpose(image_raw, (1, 2, 0))  # convert from D/H/W to H/W/D
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)  # 标准化(不是归一化), 有点像减去均值除以标准差(不过除数稍微有些不同,为了防止标准差为0?)

    #得到图片的batch和普通格式标签batch
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=64,
            capacity=20000,
            min_after_dequeue=3000)
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=64,
            capacity=2000)

    ##把普通格式标签batch转为ONE-HOT格式
    n_classes = 10
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch

