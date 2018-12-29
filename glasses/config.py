import sys
sys.path.append("/home/hanson/facetools/lib")
from easydict import EasyDict as edict  

def get_config( ):
    conf = edict()

    ##-----------------train process parameter-----------------------##
    #training dataset path list,if the input dataset is image dataset ,you needn't set the nrof_classes
    conf.train_dataset_path = "tfrecord_dataset/train.tfrecords"

    conf.test_dataset_path = "tfrecord_dataset/test.tfrecords"

    conf.train_dataset_img_width=72
    conf.train_dataset_img_height=72

    conf.nrof_classes=3  #the code can auto infernce from dataset path
    conf.batch_size=64
    conf.display_iter=10
    conf.test_save_iter=1000
    conf.max_nrof_epochs=300
    conf.models_dir="saved_models/"
    conf.model_def="squeezenet"
    conf.topn_threshold=97.0

    ##--------------------hyper parameter---------------------------##
    lr_type_dict={0:'exponential_decay',1:'piecewise_constant',2:'manual_modify'}
    conf.lr_type=lr_type_dict[1]
    conf.learning_rate=0.04  #if learning_rate is -1,use learning_rate schedule file
    #expontial decay
    conf.learning_rate_decay_step=1000
    conf.learning_rate_decay_rate=0.98
    #piecewise constant
    conf.boundaries = [10000, 100000,500000] #the num means iters
    conf.values = [0.1, 0.01, 0.001,0.0001]  #the number means learning rate
    #manual_modify
    conf.modify_step=100

    # optimizer func
    optimizer_dict={0:'ADAGRAD',1:'ADADELTA',2:'ADAM',3:'RMSPROP',4:'MOM'}
    conf.optimizer=optimizer_dict[2]
    conf.moving_average_decay=0.9999
    conf.weight_decay=5e-4
    conf.gpu_memory_fraction=1

    ##---------------------Data Augment-----------------------------##
    #open random crop,crop image size must less than dataset image size
    conf.random_crop=1
    conf.crop_img_width=60
    conf.crop_img_height=60
    conf.input_img_width  = conf.crop_img_width  if conf.random_crop else  conf.train_dataset_img_width
    conf.input_img_height = conf.crop_img_height if conf.random_crop else  conf.train_dataset_img_height
    #random rotate
    conf.random_rotate=1
    conf.rotate_angle_range=[-10,10]
    #random flip
    conf.random_flip=1
    #blur image
    conf.blur_image=0
    conf.blur_ratio_range=[0.5,1]  #0 to 1.0
    #random brigtness
    conf.random_color_brightness=0
    conf.brightness_range=[-0.1,0.5]  #0.0 to 1.0
    #random hue
    conf.random_color_hue=0
    conf.hue_range=[0,0.05]              #0 to 0.5
    #random contrast
    conf.random_color_contrast=0
    conf.contrast_range=[0.8,1.5]
    #random saturation
    conf.random_color_saturation=0
    conf.saturaton_range=[0.6,1.5]
    #image preprocess type
    # 0: image=(image-mean)/std
    # 1: image=(image-127.5)/128.0
    # 2: image=image/255.0
    conf.img_preprocess_type=1


    return conf


conf=get_config
if __name__=="__main__":
    print (get_config())