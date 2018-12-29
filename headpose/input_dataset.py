import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import config
import tensorflow as tf
import cv2
import sys
sys.path.append("/home/hanson/facetools/lib")
import faceutils as fu
import math

class TFRecordDataset(fu.TFBaseDataset):
    def __init__(self,easyconfig):
        self.conf=easyconfig 
        self._nrof_classes=-1
                
    def data_parse_function(self,example_proto):
        features = {'img_raw': tf.FixedLenFeature([], tf.string),
                    'img_width':tf.FixedLenFeature([], tf.int64),
                    'img_height':tf.FixedLenFeature([], tf.int64),
                    'binned_pose': tf.FixedLenFeature([3], tf.int64),
                    'cont_labels': tf.FixedLenFeature([3], tf.float32)
                    }

        features = tf.parse_single_example(example_proto, features)
        img_width=tf.cast(features['img_width'], tf.int64)
        img_height=tf.cast(features['img_height'], tf.int64)
        img = tf.decode_raw(features['img_raw'],tf.uint8)
        img = tf.reshape(img, shape=(img_height,img_width,3))
        img=self.miximgprocess(img)
        binned_pose = tf.cast(features['binned_pose'], tf.int64)
        cont_labels = tf.cast(features['cont_labels'], tf.float32)
        return img,binned_pose,cont_labels



def read_tfrecord_test():
    demo=TFRecordDataset( config.get_config() )
    iterator,next_element=demo.generateDataset(tfrecord_path='tfrecord_dataset/train.tfrecords',test_mode=1,batch_size=1)
    sess = tf.Session()

    # begin iteration
    for i in range(1000):
        sess.run(iterator.initializer)
        while True:
            try:
                images, binned_pose,cont_labels= sess.run(next_element)
                print (images.dtype)
                print(binned_pose,cont_labels)
                resultimg= images[0]
                resultimg=cv2.cvtColor(resultimg,cv2.COLOR_RGB2BGR)
                cv2.imshow('test', resultimg)
                cv2.waitKey(0)
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break


if __name__ == '__main__':
    read_tfrecord_test()