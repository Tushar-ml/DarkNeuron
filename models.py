""" This Python Script Contains all Predefined ImageNet Models with Their Implementations

which will be directed towards Classification File

Author : Tushar Goel

"""
import tensorflow as tf
import os
import wget
import sys


def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

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
                print('Downloading MobileNetV2_Notop_weights...')
                url = 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5'
                wget.download(url,weights_no_top,bar=bar_progress)
                print('\n---------Weight Downloaded-------------')
            print('\nWeights Loaded\n')
            model = tf.keras.applications.MobileNetV2(include_top=True,weights=weights_top)
            
        return model
    

        
            


        