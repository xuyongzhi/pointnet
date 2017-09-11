import tensorflow as tf
import os
import sys
sys.path.append('/home/x/Research/tensorflow/tensorflow/python/tools')
import inspect_checkpoint  as inscp

LOGPATH = 'LOG_QI/log1'


def inspect_checkpoint(logpath):
    inscp.print_tensors_in_checkpoint_file(os.path.join(logpath,'model.ckpt'),\
                                           tensor_name=False,\
                                           all_tensors=True)

def show_check_point_1():

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    sess = tf.Session(config = config)

    new_saver = tf.train.import_meta_graph(os.path.join(logpath,'model.ckpt.meta'))
    print( tf.train.latest_checkpoint('./'))
    what=new_saver.restore(sess, os.path.join(logpath,'model.ckpt'))
    print(what)
    all_vars = tf.get_collection('train_op')
    #all_vars = tf.get_collection(what.GraphKeys.GLOBAL_VARIABLES)
    print(all_vars)
    for v_ in all_vars:
            v_ = sess.run(v_)
            print(v_)


def Rename_moments_ExponentialMovingAverage(old_log_path):
    '''
    In tf1.1:
        conv1/bn/conv1/bn/moments/moments_1/mean/ExponentialMovingAverage
        conv1/bn/conv1/bn/moments/moments_1/variance/ExponentialMovingAverage
    After tf1.1:
        conv1/bn/conv1/bn/moments/Squeeze/ExponentialMovingAverage
        conv1/bn/conv1/bn/moments/Squeeze_1/ExponentialMovingAverage
    '''
    new_log_path = old_log_path+'_updated'

    OLD_CHECKPOINT_FILE = os.path.join(old_log_path,'model.ckpt')
    NEW_CHECKPOINT_FILE = os.path.join(new_log_path,'model.ckpt')

    vars_to_rename = {
        "/moments/moments_1/mean/":"/moments/Squeeze/",
        "/moments/moments_1/variance/":"/moments/Squeeze_1/",
    }
    new_checkpoint_vars = {}
    tf.reset_default_graph()
    reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
    for old_name in reader.get_variable_to_shape_map():
        #print(old_name)
        for var in  vars_to_rename:
            new_name = old_name.replace(var,vars_to_rename[var])
            if new_name != old_name:
                print(var)
                print('%s -> \n%s'%(old_name,new_name))
            new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)
    #tf.get_default_graph().finalize()

    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, NEW_CHECKPOINT_FILE)
        #saver.save(sess, NEW_CHECKPOINT_FILE,write_meta_graph=False)
    print('\n%s -> \n%s'%(OLD_CHECKPOINT_FILE,NEW_CHECKPOINT_FILE))
    return new_log_path


if __name__ == '__main__':
    for i in range(1,7):
        LOGPATH = 'LOG_QI/log'+str(i)
        new_log_path = Rename_moments_ExponentialMovingAverage(LOGPATH)

    #inspect_checkpoint(LOGPATH)
print('OK')
