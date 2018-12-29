from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("TrainingNets")
sys.path.append("lossfunc")
import numpy as np
import tensorflow as tf
import input_dataset
import importlib
import config
import tensorflow.contrib.slim as slim
import time
from datetime import datetime
import shutil
import vgg
import mobilenet_v1


def evaluate(model_dir):
    ckpt=tf.train.latest_checkpoint(model_dir)
    meta=ckpt+".meta"
    saver=tf.train.import_meta_graph(meta)
    input = tf.get_default_graph().get_tensor_by_name("input:0")
    lable = tf.get_default_graph().get_tensor_by_name("labels:0")
    ouput = tf.get_default_graph().get_tensor_by_name("output:0")
    phase_train=tf.get_default_graph().get_tensor_by_name("phase_train:0")
    valid_iterator,valid_next_element = input_data.input_tfrecord_data( config.valid_tfrecord_path,config.batch_size)


    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        sess.run(valid_iterator.initializer)
        total_acc=0
        test_cnt=0
        while True:
            try:
                test_cnt+=1
                valid_img,valid_label=(sess.run(valid_next_element) )
                fd={input:valid_img,lable:valid_label,phase_train: False}
                predict_label=sess.run(ouput,feed_dict=fd)
                result=np.equal(valid_label,predict_label)
                result=result.astype(np.float32)
                total_acc+= result.mean()
            except tf.errors.OutOfRangeError:
                print ("test accuracy %.4f"%(total_acc*1.0/test_cnt) )
                break



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
    train_dataset=input_dataset.TFRecordDataset(conf)
    train_iterator,train_next_element = train_dataset.generateDataset( dataset_path=conf.train_dataset_path,batch_suize=conf.batch_size)
    test_dataset=input_dataset.TFRecordDataset(conf)
    test_iterator,test_next_element = test_dataset.generateDataset( dataset_path=conf.test_dataset_path,batch_size=conf.batch_size,test_mode=1)

    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    images_placeholder = tf.placeholder(name='input', shape=[None, conf.input_img_height,conf.input_img_width, 3], dtype=tf.float32)
    labels_placeholder = tf.placeholder(name='labels', shape=[None, ], dtype=tf.int64)

    # Create the model.
    #with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(batch_norm_updates_collections=None)):
        #predictions, end_points = mobilenet_v1.mobilenet_v1(images_placeholder,is_training=phase_train_placeholder,num_classes=3,prediction_fn=False)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        predictions, end_points = vgg.vgg_a(images_placeholder,num_classes=3,is_training=phase_train_placeholder)

    output=tf.argmax(predictions,1,name="output")
    
    softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=labels_placeholder),name="loss")
    tf.add_to_collection('losses', softmax_loss)


    correct_prediction = tf.equal(tf.argmax(predictions,1),labels_placeholder )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #adjust learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(conf.learning_rate,global_step,conf.learning_rate_decay_step,conf.learning_rate_decay_rate,staircase=True)


    custom_loss=tf.get_collection("losses")
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss=tf.add_n(custom_loss+regularization_losses,name='total_loss')


    #optimize loss and update
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(total_loss,global_step=global_step)
    #train_op=tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(total_loss,global_step=global_step)

    saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(conf.max_nrof_epochs):
            sess.run(train_iterator.initializer)
            while True:
                use_time=0
                try:
                    images_train, labels_train = sess.run(train_next_element)

                    start_time=time.time()
                    input_dict={phase_train_placeholder:True,images_placeholder:images_train,labels_placeholder:labels_train}
                    step,lr,train_loss,_,train_accuracy = sess.run([global_step,
                                                                            learning_rate,
                                                                            total_loss,
                                                                            train_op,
                                                                            accuracy],
                                                                          feed_dict=input_dict)

                    end_time=time.time()
                    use_time+=(end_time-start_time)

                    #display train result
                    if(step%conf.display_iter==0):
                        print ("step:%d lr:%f time:%.3f total_loss:%.3f acc:%.3f epoch:%d"%(step,lr,use_time,train_loss,train_accuracy,epoch) )
                        use_time=0
                    if (step%conf.test_save_iter==0):
                        filename_cpkt = os.path.join(models_dir, "%d.ckpt"%step)
                        saver.save(sess, filename_cpkt)
                        #evaluate(models_dir)
                        sess.run(test_iterator.initializer)
                        total_acc=0
                        test_cnt=0
                        
                        while True:
                            try:
                                test_cnt+=1
                                test_img,test_label=(sess.run(test_next_element) )
                                fd={images_placeholder:test_img,labels_placeholder:test_label,phase_train_placeholder: False}
                                acc=sess.run(accuracy,feed_dict=fd)
                                total_acc+=acc
                            except tf.errors.OutOfRangeError:
                                valid_acc=(total_acc*1.0/test_cnt)*100
                                print ("test accuracy %.2f"%valid_acc )
                                with open(os.path.join(topn_models_dir,"topn_acc.txt"),"a+")as tmp_f:
                                        tmp_f.write("step : %d  accuracy : %f\n"%(step,valid_acc) )
                                if valid_acc>conf.topn_threshold:
                                    shutil.copyfile(os.path.join(models_dir, "%d.ckpt.meta"%step),os.path.join(topn_models_dir, "%d.ckpt.meta"%step))
                                    shutil.copyfile(os.path.join(models_dir, "%d.ckpt.index"%step),os.path.join(topn_models_dir, "%d.ckpt.index"%step))
                                    shutil.copyfile(os.path.join(models_dir, "%d.ckpt.data-00000-of-00001"%step),os.path.join(topn_models_dir, "%d.ckpt.data-00000-of-00001"%step))
                                break
                        
                except tf.errors.OutOfRangeError:
                    print("End of epoch ")
                    break


run_training()
