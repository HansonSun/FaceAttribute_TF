from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import config
import sys
import faceutils as fu 
import numpy as np
import cv2
import tensorflow as tf
import os,sys


class TFRecordDataset(fu.TFRecordDataset):
    def __init__(self,easyconfig):
        self.conf=easyconfig 
        self._nrof_classes=-1
                
    def data_parse_function(self,example_proto):
        features = {'img_raw': tf.FixedLenFeature([], tf.string),
                    'label'  : tf.FixedLenFeature([], tf.int64)
                    }

        features = tf.parse_single_example(example_proto, features)
        img = tf.decode_raw(features['img_raw'],tf.uint8)
        img = tf.reshape(img, shape=(128,128,3))
        img=self.miximgprocess(img)
        label = tf.cast(features['label'], tf.int64)
        return img,label



def read_tfrecord_test():
    demo=TFRecordDataset( config.get_config() )
    iterator,next_element=demo.generateDataset(dataset_path='tfrecord_dataset/test.tfrecords',test_mode=0,display_mode=1,batch_size=1)
    sess = tf.Session()

    # begin iteration
    for i in range(1000):
        sess.run(iterator.initializer)
        while True:
            try:
                images, label= sess.run(next_element)
                print (images)
                resultimg= images[0]
                resultimg=cv2.cvtColor(resultimg,cv2.COLOR_RGB2BGR)
                cv2.imshow('test', resultimg)
                cv2.waitKey(0)
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break





if __name__ == '__main__':
	read_tfrecord_test()
