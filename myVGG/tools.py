import tensorflow as tf
import numpy as np

#定义损失, 交叉熵损失
def loss(logits, labels):
    '''

    :param logits: 格式[batch_size, num_classes], 这里的logits没有经过softmax处理
    :param labels: 格式[batch_size, num_classes], one-hot标签
    :return:
    '''
    with tf.name_scope('cross_entropy_loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar(scope, loss)
        return loss


#计算输入的batch预测准确的百分比
def accuracy(logits, labels):
  """.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor,
  """
  with tf.name_scope('accuracy') as scope:
    #tf.arg_max返回数组指定维数中(这里是1)最大的下标(注意是下标), tf.equal判断是两个数组各个元素是否相等,然后返回类似[True, False,...]的数组, tf.cast把输入数据转换成指定类型, 这里可以把前面的布尔数组转换为[1.0, 0...]
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1)), tf.float32))*100
    tf.summary.scalar(scope, accuracy)
    return accuracy


#定义优化方法
def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) #先使用最常用的梯度下降法
        train_op = optimize.minimize(loss, global_step=global_step)
        return train_op

#
def optimize2(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) #先使用最常用的梯度下降法
        train_op = optimize.minimize(loss)
        return train_op


#先抄下, 这个应该是查看下载的npy文件各个参数的shape
def test_load():
    data_path = '/home/rong/something_for_deep/vgg16.npy'

    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)


#除了几个全连接层,其他层的参数从npy文件中载入
def load_with_skip(data_path, session, skip_layer):
    # python3默认编码是'UTF-8',所以设置encoding(python2是ASCII)
    #np.load返回一个dict, item()专门用于python的字典,返回可遍历的(键,值)元组数组 [(key1,value1),....(keyn,valuen)], 在这里是key是名称'conv1'(包括weight和bias), value是参数
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data)) #assign就是直接赋值


#判断一个batch有多少个数据是预测正确的
def num_correct_prediction(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Return:
      the number of correct predictions
  """
  correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
  correct = tf.cast(correct, tf.int32)
  n_correct = tf.reduce_sum(correct)
  return n_correct
# %%


def print_all_variables(train_only=True):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)

    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try:  # TF1.0
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

        # %%