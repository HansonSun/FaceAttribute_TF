import sys
sys.path.append("/home/hanson/facetools/lib")
import faceutils as fu
import cv2
import itertools
import os
import tensorflow as tf
import PIL.Image
import numpy as np
import random

def create_example(img_file, binned_pose,cont_labels):
    img = PIL.Image.open(img_file)
    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': fu.tf_bytes_feature(img.tobytes() ),
        'img_width':fu.tf_int_feature(img.width),
        'img_height':fu.tf_int_feature(img.height),
        'binned_pose': fu.tf_int_feature(binned_pose),
        'cont_labels': fu.tf_float_feature(cont_labels),
    }))
    return example



if __name__ == '__main__':

    with tf.python_io.TFRecordWriter("tfrecord_dataset/train.tfrecords") as writer:
        for infor in fu.getFaceinfors("/home/hanson/dataset/CelebA/Img/img_celeba.7z/img_celeba_fc_72x72_zoom","img_celeba_facepp_label.csv"):

            yaw, pitch, roll=infor['faceinfor'].yaw,infor['faceinfor'].pitch,infor['faceinfor'].roll 

            # Bin values
            bins = np.array(range(-99, 102, 3))
            binned_pose = list(np.digitize([yaw, pitch, roll], bins))
            cont_labels = [yaw, pitch, roll]

            tf_example = create_example(infor['imgpath'],binned_pose,cont_labels)
            writer.write(tf_example.SerializeToString())
