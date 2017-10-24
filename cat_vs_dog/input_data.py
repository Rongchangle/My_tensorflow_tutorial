import tensorflow as tf
import numpy as np
import os


def get_files(file_dir):
    '''
    函数get_files(file_dir)
    输入保存数据库文件的路径
    以list(python)格式输出所有图像的文件路径和标签
    输出结果的list格式如下(顺序是打乱的):
        [['path3', 0],
            ....
         ['path5', 1],
         ['path2, 0]]


    代码函数学习:
        1.python的list可以用append操作加入一个元素,但是numpy的array好像没有什么好办法,所以开始使用list便于操作
        2.numpy的tranpose()转置,把[ ['path1','path2',...'path10'], [0,1,...1]] 这种array转换成: [['path1',0],['path2',1],...['path10',1]]
        3.numpy的random.shuffle可以打乱顺序
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


# %%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    '''
    输入所有训练图像路径和标签,输出指定图片大小和制定batch大小的batch
    主要是input_queue = tf.train.slice_input_producer([image, label]),这里的image其实还是图片路径
    input_queue[0]部分经过一系列处理,还有input_queue[1](标签)最后用tf.train.batch就可以返回训练的batch(因为slice_input_producer的功劳.不需打乱图片顺序)
    补充就是因为有tf的resize函数,所以这里还可以输出指定的图片大小
    用tf.train.batch函数去除的batch可以用tf的函数(比如reshape,cast)来把这些tensor转换成神经网络可以接受的格式和形状
    '''
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label]) #

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    # you can also use shuffle_batch
    #    image_batch, label_batch = tf.train.shuffle_batch([image,label],
    #                                                      batch_size=BATCH_SIZE,
    #                                                      num_threads=64,
    #                                                      capacity=CAPACITY,
    #                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


