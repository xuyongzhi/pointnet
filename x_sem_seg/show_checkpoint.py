import tensorflow as tf
import os
#import tensorflow.tools as tf_tools
#from tensorflow.tools import  inspect_checkpoint


logpath = 'log5_4096_bs32'
#tf_tools.inspect_checkpoint.print_tensors_in_checkpoint_file(checkpoint_fn)
#tf.tools.inspect_checkpoint.print_tensors_in_checkpoint_file(checkpoint_fn)


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

print('OK')
