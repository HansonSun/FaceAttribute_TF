from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("/home/hanson/facetools/lib")
import faceutils as fu
import numpy as np
import tensorflow as tf
from  input_dataset import *
import importlib
import config
import tensorflow.contrib.slim as slim
import time
from datetime import datetime
import shutil
import vgg



def run_training():

    #1.create log and model saved dir according to the datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    models_dir = os.path.join("saved_models", subdir, "models")
    if not os.path.isdir(models_dir):  # Create the model directory if it doesn't exist
        os.makedirs(models_dir)
    logs_dir = os.path.join("saved_models", subdir, "logs")
    if not os.path.isdir(logs_dir):  # Create the log directory if it doesn't exist
        os.makedirs(logs_dir)
    topn_models_dir = os.path.join("saved_models", subdir, "topn")#topn dir used for save top accuracy model
    if not os.path.isdir(topn_models_dir):  # Create the topn model directory if it doesn't exist
        os.makedirs(topn_models_dir)
    topn_file=open(os.path.join(topn_models_dir,"topn_acc.txt"),"a+")
    topn_file.close()


    #2.load dataset and define placeholder
    conf=config.get_config()
    demo=TFRecordDataset( conf )
    train_iterator,train_next_element=demo.generateDataset(tfrecord_path='tfrecord_dataset/train.tfrecords',test_mode=0,batch_size=256)


    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    images_placeholder = tf.placeholder(name='input', shape=[None, conf.input_img_height,conf.input_img_width, 3], dtype=tf.float32)
    binned_pose_placeholder = tf.placeholder(name='binned_pose', shape=[None,3 ], dtype=tf.int64)
    cont_labels_placeholder = tf.placeholder(name='cont_labels', shape=[None,3 ], dtype=tf.float32)

    yaw,pitch,roll = vgg.inference(images_placeholder,phase_train=phase_train_placeholder)

    loss_yaw   = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yaw,labels=binned_pose_placeholder[:,0])
    loss_pitch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pitch,labels=binned_pose_placeholder[:,1])
    loss_roll  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=roll,labels=binned_pose_placeholder[:,2])


    softmax_yaw=tf.nn.softmax(yaw)
    softmax_pitch=tf.nn.softmax(pitch)
    softmax_roll=tf.nn.softmax(roll)

    yaw_predicted   =  tf.math.reduce_sum( (softmax_yaw * tf.linspace(0.0,67.0,68) ) )* 3 - 99
    pitch_predicted =  tf.math.reduce_sum( (softmax_pitch * tf.linspace(0.0,67.0,68) ) )* 3 - 99
    roll_predicted  =  tf.math.reduce_sum( (softmax_roll * tf.linspace(0.0,67.0,68) ) )* 3 - 99


    yaw_mse_loss   = tf.reduce_mean(tf.square(yaw_predicted - cont_labels_placeholder[:,0]))
    pitch_mse_loss = tf.reduce_mean(tf.square(pitch_predicted - cont_labels_placeholder[:,1]))
    roll_mse_loss  = tf.reduce_mean(tf.square(roll_predicted - cont_labels_placeholder[:,2]))

    # # Total loss
    #loss_yaw   += 0.0001 * yaw_mse_loss
    #loss_pitch += 0.0001 * pitch_mse_loss
    #loss_roll  += 0.0001 * roll_mse_loss

    #reg_loss=tf.reduce_mean(0.0001 * yaw_mse_loss+ 0.0001 * pitch_mse_loss+ 0.0001 * roll_mse_loss)
    #softmax_loss=tf.reduce_mean(yaw_predicted+pitch_predicted+roll_predicted)

    total_loss=(loss_yaw + loss_pitch + loss_roll)
    #total_loss=0.0001 * yaw_mse_loss#+ 0.0001 * pitch_mse_loss+ 0.0001 * roll_mse_loss
    total_loss= tf.reduce_mean(loss_pitch) 
    print(total_loss)
    #total_loss=reg_loss+softmax_loss

    #correct_prediction = tf.equal(tf.argmax(predictions,1),labels_placeholder )
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #adjust learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001,global_step,100000,0.98,staircase=True)



    #optimize loss and update
    #optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    grads=optimizer.compute_gradients(total_loss)

    #with tf.name_scope('clip_grads'):
        #grads = slim.learning.clip_gradient_norms(grads, 2 )


    train_op=optimizer.apply_gradients(grads,global_step=global_step)
    #train_op=tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(total_loss,global_step=global_step)

    saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=5)

    sess=fu.session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(conf.max_nrof_epochs):
        sess.run(train_iterator.initializer)
        while True:
            use_time=0
            try:
                images_train, binned_pose,cont_labels = sess.run(train_next_element)
                start_time=time.time()
                input_dict={phase_train_placeholder:True,images_placeholder:images_train,binned_pose_placeholder:binned_pose,cont_labels_placeholder:cont_labels}
                step,lr,train_loss,_ = sess.run([global_step,
                                                        learning_rate,
                                                        total_loss,
                                                        train_op],
                                                         feed_dict=input_dict)

                end_time=time.time()
                use_time+=(end_time-start_time)

                #display train result
                if(step%conf.display_iter==0):
                    print ("step:%d lr:%f time:%.3f total_loss:%.3f  epoch:%d"%(step,lr,use_time,float(train_loss),epoch) )
                    use_time=0

            except tf.errors.OutOfRangeError:
                print("End of epoch ")
                break


run_training()
