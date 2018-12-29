import cv2
import itertools
import os
import tensorflow as tf
import PIL.Image
import numpy as np
import random

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_example(img_file, label):
    img = PIL.Image.open(img_file)
    img = img.resize((128, 128))
    img_raw = img.tobytes() 

    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': bytes_feature(img_raw),
        'label': int64_feature(label),
    }))
    return example



if __name__ == '__main__':
    train_path_list=[]
    train_label_list=[]

    valid_path_list=[]
    valid_label_list=[]

    rootdirname="dataset_facecrop"
    classes_name_list=["%s/no_glasses"%rootdirname,"%s/normal_glasses"%rootdirname,"%s/sun_glasses"%rootdirname]

    for index,classdir in enumerate(classes_name_list):
        classs_path_list=[os.path.join(classdir,p) for p in os.listdir(classdir)]
        class_len=len(classs_path_list)
        valid_class_len=int(0.1*class_len)
        classs_label_list=[ index for i in range(class_len)]


        train_path_list+=classs_path_list[valid_class_len:]
        train_label_list+=classs_label_list[valid_class_len:]
        valid_path_list+=classs_path_list[:valid_class_len]
        valid_label_list+=classs_label_list[:valid_class_len]


    train_path_label=zip(train_path_list,train_label_list)
    random.shuffle(train_path_label)

    valid_path_label=zip(valid_path_list,valid_label_list)
    random.shuffle(valid_path_label)


    
    with tf.python_io.TFRecordWriter("tfrecord_dataset/train.tfrecords") as writer:
        for imgpath,label in train_path_label:
            tf_example = create_example(imgpath,label)
            writer.write(tf_example.SerializeToString())


    with tf.python_io.TFRecordWriter("tfrecord_dataset/valid.tfrecords") as writer:
        for imgpath,label in valid_path_label:
            tf_example = create_example(imgpath,label)
            writer.write(tf_example.SerializeToString())



