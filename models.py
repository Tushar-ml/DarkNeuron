""" This Python Script Contains all Predefined ImageNet Models with Their Implementations

which will be directed towards Classification File

Author : Tushar Goel

"""
import tensorflow as tf
import os
import wget
import sys


class Models:
    """
    This Class contains functions of Different models which will return model
    
    """
    def __init__(self,working_directory,image_shape=(224,224,3),train=False):
        
        self.working_directory = working_directory      # Working Directory
        self.train = train                              # Train or Predicting
        self.image_shape = image_shape                  # Image Shape
    
    def MobileNetV2(self):
        """
        Initiatisation of Mobile Net V2 Model
        
        """
        
        if self.train:
            weights_no_top = os.path.join(self.working_directory,'mobilenet_notop_model.h5')
            
            if not os.path.exists(weights_no_top):
                print('Downloading MobileNetV2_Notop_weights...')
                url = 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.MobileNetV2(include_top=False,input_shape=self.image_shape,weights=weights_no_top)
            
        else:
            weights_top = os.path.join(self.working_directory,'mobilenet_model.h5')
            
            if not os.path.exists(weights_top):
                print('Downloading MobileNetV2_weights...')
                url = 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.MobileNetV2(include_top=True,weights=weights_top)
        return model
    
    def InceptionV3(self):
        """
        Initiatisation of InceptionV3 Model
        
        """
        
        if self.train:
            weights_no_top = os.path.join(self.working_directory,'inceptionv3_notop_model.h5')
            
            if not os.path.exists(weights_no_top):
                print('Downloading InceptionV3_Notop_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.InceptionV3(include_top=False,input_shape=self.image_shape,weights=weights_no_top)
            
        else:
            weights_top = os.path.join(self.working_directory,'inceptionv3_model.h5')
            
            if not os.path.exists(weights_top):
                print('Downloading InceptionV3_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.InceptionV3(include_top=True,weights=weights_top)
        return model
        
    def ResNet50(self):
        """
        Intialisation of Resnet50 Model
        
        """
        if self.train:
            weights_no_top = os.path.join(self.working_directory,'ResNet50_notop_model.h5')
            
            if not os.path.exists(weights_no_top):
                print('Downloading Resnet50_Notop_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.ResNet50(include_top=False,input_shape=self.image_shape,weights=weights_no_top)
            
        else:
            weights_top = os.path.join(self.working_directory,'ResNet50_model.h5')
            
            if not os.path.exists(weights_top):
                print('Downloading Resnet50_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.ResNet50(include_top=True,weights=weights_top)
        return model
    
    def Xception(self):
        """
        Initialisation of Xception Model
        
        """
        if self.train:
            weights_no_top = os.path.join(self.working_directory,'Xception_notop_model.h5')
            
            if not os.path.exists(weights_no_top):
                print('Downloading Xception_Notop_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.Xception(include_top=False,input_shape=self.image_shape,weights=weights_no_top)
            
        else:
            weights_top = os.path.join(self.working_directory,'Xception_model.h5')
            
            if not os.path.exists(weights_top):
                print('Downloading Xception_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.Xception(include_top=True,weights=weights_top)
        return model
    
    def VGG16(self):
        """
        Initialisation of VGG16 Model
        
        """
        if self.train:
            weights_no_top = os.path.join(self.working_directory,'VGG16_notop_model.h5')
            
            if not os.path.exists(weights_no_top):
                print('Downloading VGG16_Notop_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.VGG16(include_top=False,input_shape=self.image_shape,weights=weights_no_top)
            
        else:
            weights_top = os.path.join(self.working_directory,'VGG16_model.h5')
            
            if not os.path.exists(weights_top):
                print('Downloading VGG16_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.VGG16(include_top=True,weights=weights_top)
        return model

    def VGG19(self):
        """
        Initialisation of VGG16 Model
        
        """
        if self.train:
            weights_no_top = os.path.join(self.working_directory,'VGG19_notop_model.h5')
            
            if not os.path.exists(weights_no_top):
                print('Downloading VGG19_Notop_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.VGG19(include_top=False,input_shape=self.image_shape,weights=weights_no_top)
            
        else:
            weights_top = os.path.join(self.working_directory,'VGG19_model.h5')
            
            if not os.path.exists(weights_top):
                print('Downloading VGG19_weights...')
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.VGG19(include_top=True,weights=weights_top)
        return model



def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()
        