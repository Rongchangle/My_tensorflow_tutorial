import tensorflow as tf
import numpy as np


class VGG16:
    #输入图片的batch, 输出最后分类的权重
    def build(self, x, num_classes, is_pretrain=True):
        '''
        参考论文设计的VGG16的网络结构, 这个网络暂时没有什么drop_out, 有机会我补上
        :param x: 输入tensor,一般情况是[batch_size, height, width, channels]
        :param num_classes: 分类数目
        :param is_pretrain: 这个暂时不懂
        :return: 返回图片为每个类的权重(这里的权重还没有经过softmax), 一般情况[batch_size, num_classes]
        '''
        self.conv1_1 = self.conv('conv1_1', x, out_channels=64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.conv1_2 = self.conv('conv1_2', self.conv1_1, out_channels=64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.pool1 = self.pool('pool1', self.conv1_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv2_1 = self.conv('conv2_1', self.pool1, out_channels=128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.conv2_2 = self.conv('conv2_2', self.conv2_1, out_channels=128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.pool2 = self.pool('pool2', self.conv2_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv3_1 = self.conv('conv3_1', self.pool2, out_channels=256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.conv3_2 = self.conv('conv3_2', self.conv3_1, out_channels=256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.conv3_3 = self.conv('conv3_3', self.conv3_2, out_channels=256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.pool3 = self.pool('pool3', self.conv3_3, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv4_1 = self.conv('conv4_1', self.pool3, out_channels=512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.conv4_2 = self.conv('conv4_2', self.conv4_1, out_channels=512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.conv4_3 = self.conv('conv4_3', self.conv4_2, out_channels=512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.pool4 = self.pool('pool4', self.conv4_3, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv5_1 = self.conv('conv5_1', self.pool4, out_channels=512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.conv5_2 = self.conv('conv5_2', self.conv5_1, out_channels=512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.conv5_3 = self.conv('conv5_3', self.conv5_2, out_channels=512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        self.pool5 = self.pool('pool5', self.conv5_3, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.fc6 = self.FC_layer('fc6', self.pool5, out_nodes=4096)
        self.fc7 = self.FC_layer('fc7', self.fc6, out_nodes=4096)
        self.fc8 = self.FC_layer('fc8', self.fc7, out_nodes=num_classes)

        return self.fc8

    #定义个通用的卷积层,激活函数relu
    def conv(self, layer_name, x, out_channels, kernel_size=[3,3],stride=[1,1,1,1], is_pretrain=True):
        '''
        :param layer_name: 当前网络的名字,比如"conv1","conv2"
        :param x: 输入的tensor, [batch, height, width, channels]
        :param out_channels: 输出的通道数
        :param kernel_size: 卷积核大小,默认3*3
        :param stride: 卷积窗口步长, 默认1
        :param is_pretrain: 参数是否可以训练, 要知道, 有时候某些层参数是会冻结的
        :return: 输出的tensor, [batch, height, width, channels]
        '''
        in_channels = x.get_shape()[-1] #得到输入tensor的通道数
        with tf.variable_scope(layer_name): #创建
            w = tf.get_variable(name='weights',
                                trainable=is_pretrain,
                                shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                trainable=is_pretrain,
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=w, strides=stride,padding='SAME',name='conv_x')
            x = tf.nn.bias_add(x, b, name='bias_x')
            x = tf.nn.relu(x, name='relu_x')
            return x

    #定义一个通用的池化层
    def pool(self, layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
        '''
        :param layer_name: 当前网络的名字,比如"pool1","pool2"
        :param x: 输入的tensor, [batch, height, width, channels]
        :param kernel: 池化窗口大小,默认2*2
        :param stride: 池化窗口步长, 默认2
        :param is_max_pool: 如果是,使用max_pool, 否则使用average_pool
        :return: 输出的tensor, [batch, height, width, channels]
        '''
        if is_max_pool:
            x = tf.nn.max_pool(x, kernel, stride, padding='SAME', name=layer_name)
        else:
            x = tf.nn.avg_pool(x, kernel, stride, padding='SAME', name=layer_name)
        return x

    #定义一个通用全连接层,激活函数relu
    def FC_layer(self, laryer_name, x, out_nodes):
        '''
        :param laryer_name: 全连接层名字, 比如'FC1','FC2'
        :param x: 输入的tensor
        :param out_nodes: 输出层的点数
        :return:
        '''
        shape = x.get_shape()
        if len(shape) == 4: #刚从卷积或池化层出来
            in_nodes = shape[1].value*shape[2].value*shape[3].value
        else: #上个也是全连接层
            in_nodes = shape[-1].value

        with tf.variable_scope(laryer_name):
            w = tf.get_variable(name='weights',
                                shape=[in_nodes, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))

            reshape_x = tf.reshape(x,[-1, in_nodes]) #转换成[batch_size, in_nodes]大小的tensor,不过[batch_size不在函数输入里面,就写成-1了

            x = tf.nn.bias_add(tf.matmul(reshape_x, w), b)
            x = tf.nn.relu(x,name='relu')
            return x


