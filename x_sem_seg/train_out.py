import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
UPER_DIR = os.path.dirname(ROOT_DIR)
#ETH_DATASET_DIR = os.path.join(UPER_DIR,'Dataset/ETH_Semantic3D_Dataset')
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'x_utils'))
import provider
import tf_util
from model import *
from outdor_data_prep_util import Normed_H5f,Net_Provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log_outdoor', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--no_color_1norm',action='store_true',help='set true when do trian with color_1norm ')
parser.add_argument('--no_intensity_1norm',action='store_true',help='set true when do trian with intensity_1norm ')
#parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')

parser.add_argument('--only_evaluate',action='store_true',help='do not train')

FLAGS = parser.parse_args()
FLAGS.no_intensity_1norm = True
FLAGS.model_path = os.path.join(FLAGS.log_dir,'model.ckpt')
MODEL_PATH = FLAGS.model_path

BATCH_SIZE = FLAGS.batch_size
if FLAGS.only_evaluate:
    MAX_EPOCH = 1
else:
    MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
if not FLAGS.only_evaluate:
    log_name = 'log_train.txt'
else:
    log_name = 'log_test.txt'
LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = Normed_H5f.NUM_CLASSES

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

ALL_FILES = provider.getDataFiles(os.path.join('ETH3D_sem_seg_hdf5_data/all_files.txt') )



# Load ALL data
all_files_f = os.path.join('ETH3D_sem_seg_hdf5_data/all_files.txt')
ALL_FILES = [line.rstrip() for line in open(all_files_f)]
if not FLAGS.only_evaluate:
    train_file_list = ALL_FILES[0:1]
    eval_file_list = ALL_FILES[-1:]
    train_num_block_rate = 0.8
    eval_num_block_rate = 0.2
else:
    train_file_list = []
    eval_file_list = ALL_FILES
    train_num_block_rate = 0
    eval_num_block_rate = 1

net_provider = Net_Provider( train_file_list,eval_file_list,NUM_POINT,\
                            only_evaluate = FLAGS.only_evaluate,\
                            no_color_1norm = FLAGS.no_color_1norm,\
                            no_intensity_1norm = FLAGS.no_intensity_1norm,\
                            train_num_block_rate = train_num_block_rate,\
                            eval_num_block_rate = eval_num_block_rate)
NUM_CHANNELS = net_provider.num_channels

    #data_batch_list = []
    #label_batch_list = []
    #for i,h5_filename in enumerate(ALL_FILES):
    #    data_batch, label_batch = provider.loadDataFile(h5_filename)
    #    data_batch_list.append(data_batch)
    #    label_batch_list.append(label_batch)
    #    if  i>0:
    #        break
    #
    #data_batches = np.concatenate(data_batch_list, 0)
    #label_batches = np.concatenate(label_batch_list, 0)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

    ## split dataset to train and test
    ##test_area = 'Area_'+str(FLAGS.test_area)
    #train_idxs = []
    #test_idxs = []
    #n_train = int(data_batches.shape[0] * 0.85)
    #train_idxs = range(0,n_train)
    #test_idxs = range(n_train,data_batches.shape[0])

    ## extract the data types to be trained
    ## xyz_1norm xyz_midnorm color_1norm intensity_1norm
    #COLOR_IDXS = Normed_H5f.elements_idxs['color_1norm']
    #INTENSITY_IDX = Normed_H5f.elements_idxs['intensity_1norm']
    #delete_idxs = []
    #if FLAGS.no_color_1norm:
    #    delete_idxs += COLOR_IDXS
    #if FLAGS.no_intensity_1norm:
    #    delete_idxs += INTENSITY_IDX
    #data_batches = np.delete(data_batches,delete_idxs,2)

    ## randomly sample the points within one batch if not what all
    #num_point_in = data_batches.shape[1]
    #if NUM_POINT < num_point_in:
    #    log_string('sample data batches from %d to %d points'%(num_point_in,NUM_POINT))
    #    sample_choice = provider.sample(num_point_in,NUM_POINT)
    #    data_batches = data_batches[:,sample_choice,:]
    #    label_batches = label_batches[:,sample_choice]
    #
    #train_data = data_batches[train_idxs,...]
    #train_label = label_batches[train_idxs]
    #test_data = data_batches[test_idxs,...]
    #test_label = label_batches[test_idxs]




def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT,NUM_CHANNELS)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(reshape=False)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        if not FLAGS.only_evaluate:
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                    sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
        else:
            test_writer = None

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        log_string('\ntrain data shape: %s'%(str(net_provider.train_data_shape)) )
        log_string('test data shape: %s\n'%(str(net_provider.eval_data_shape)) )

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            if not FLAGS.only_evaluate:
                train_one_epoch(sess, ops, train_writer,epoch)
            else:
                saver.restore(sess,MODEL_PATH)
                log_string('restored model from: \n\t%s'%MODEL_PATH)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if (not FLAGS.only_evaluate) and (epoch % 10 == 0 or epoch == MAX_EPOCH-1):
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("epoch %d, Model saved in file: %s" % ( epoch,save_path) )



def train_one_epoch(sess, ops, train_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string('----')
    #current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label[:,0:NUM_POINT])

    train_num_blocks = net_provider.train_num_blocks
    num_batches = train_num_blocks // BATCH_SIZE
    #num_batches = int(math.ceil(1.0*train_num_blocks/BATCH_SIZE))

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    print('total batch num = ',num_batches)
    def log_train():
        log_string('epoch %d batch %d \ttrain \tmean loss: %f   \taccuracy: %f' % \
                   (epoch,batch_idx,loss_sum / float(num_batches),total_correct / float(total_seen)  ))

    for batch_idx in range(num_batches):
        #if batch_idx % 100 == 0:
            #print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx+1) * BATCH_SIZE,train_num_blocks)


        batch_data,batch_label = net_provider.get_train_batch(start_idx,end_idx)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']:      batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

        if (epoch == 0 and batch_idx <= 100) or batch_idx%100==0:
            log_train()
    log_string('\n')
    log_train()


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')

    eval_num_blocks = net_provider.eval_num_blocks
    #num_batches = int(math.ceil( 1.0 * eval_num_blocks / BATCH_SIZE ))
    num_batches = eval_num_blocks // BATCH_SIZE


    def log_eval():
        log_string('\neval \tmean loss: %f  \taccuracy: %f' % (loss_sum / float(total_seen/NUM_POINT),\
                    total_correct / float(total_seen) ))
        class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
        log_string('eval class accuracies: %s' % (np.array2string(class_accuracies,formatter={'float':lambda x: "%f"%x})))


    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min( (batch_idx+1) * BATCH_SIZE, eval_num_blocks )

        batch_data,batch_label = net_provider.get_eval_batch(start_idx,end_idx)
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']:      batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        if test_writer != None:
            test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        accuracy = float(correct) / (BATCH_SIZE*NUM_POINT)
        print('accu = ',accuracy)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(0,batch_label.shape[0]):
            for j in range(NUM_POINT):
                l = batch_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i, j] == l)

        if  batch_idx%10==0:
            log_eval()
    log_eval()



if __name__ == "__main__":
    train()
    LOG_FOUT.close()
