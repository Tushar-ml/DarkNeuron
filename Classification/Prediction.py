"""
This Script is For Prediction . In this, Prediction will be filter out 

from Bunch of Raw data

Author: Tushar Goel

"""
import tensorflow as tf
from .models import Create_Model
from tensorflow.keras.preprocessing import image
import numpy as np


class Prediction:
    """
    In this Object, we will predict the Untested Objects and Provide the Filtered Predictions
    
    Attributes:
        method: Whether they are providing Dircetory,dataframe,or Single Images
        model_name: to Predict directly from the imagenet models
        
    """
    def __init__(self,method,working_directory,labels=None,model_name=None,user_model=None,img=None,directory = None,top=None,image_directory=None):
        
        self.labels = labels
        self.model_name = model_name
        self.user_model = user_model
        self.working_directory = working_directory
        self.top = top
        self.image_directory=image_directory
        print('\n\t\t-------------Prediction Phase--------------')
        
    def target_shape_generator(self):
        
        model_init = Create_Model(working_directory = self.working_directory)
        if self.user_model is not None:
            target_shape = self.user_model.input_shape[1:3]
            print('\n\t\t----------Model for Prediction Ready----------')
            return target_shape
        
        
        else:
            if self.model_name is None:
                raise ValueError('Provide atleast user model or model_name')
            # Defining Model based on Model name:
            if self.model_name in ['mobilenetv2','MobileNetV2','mobilenet_v2','MobileNet_V2']:
    
                print('\n\t\t-------MobilenetV2 For Prediction Ready----------\n')
                
                return model_init.MobileNetV2().input_shape[1:3]
                
            if self.model_name in ['inceptionv3','InceptionV3','inception_v3','Inception_V3']:
                
                
                print('\n\t\t-------InceptiontV3 for Prediction Ready----------\n')
                model = model_init.InceptionV3()                     
                return model.input_shape[1:3]
                
            if self.model_name in ['resnet50','ResNet50','Resnet50']:
    
                print('\n\t\t-------Resnet50 Model for Prediction Ready----------\n')                     
                return model_init.ResNet50().input_shape[1:3]
            
            if self.model_name in ['Xception','xception']:
    
                print('\n\t\t-------Xception Model for Prediction Ready----------\n')                     
                return model_init.Xception().input_shape[1:3]
            
            if self.model_name in ['VGG16','Vgg16','vgg16']:
                
    
                print('\n\t\t-------VGG16 Model for Prediction Ready----------\n')                     
                return model_init.VGG16().input_shape[1:3]
        
            if self.model_name in ['VGG19','Vgg19','vgg19']:
                        
            
                print('\n\t\t-------VGG19 Model for Prediction Ready----------\n')                     
                return model_init.VGG19()
        
        
        
    def prediction(self,method,model,img=None,data_generator=None):
        
        if self.model_name is None:
            #model = tf.keras.models.load_model(model)        
            if method in ['directory','dataframe','point']:
                if data_generator is None:
                    raise ValueError('Provide Image Generators in generator argument')
                prediction = model.predict_generator(data_generator,steps=len(data_generator),verbose=1)
                predicted_indices = np.argmax(prediction,axis=1)
                return predicted_indices,prediction
            
            elif method in ['image']:
                prediction = model.predict(img)
                predicted_indices = np.argmax(prediction,axis=1)
                return predicted_indices,prediction
        
        else:
            if self.model_name in ['mobilenetv2','MobileNetV2','mobilenet_v2','MobileNet_V2']:
    
                print('\n\t\t-------Getting Label For MobileNetV2----------\n')
                
                decode_predictions = tf.keras.applications.mobilenet_v2
                
            if self.model_name in ['inceptionv3','InceptionV3','inception_v3','Inception_V3']:
                
                
               print('\n\t\t-------Getting Label For InceptionV3----------\n')
                
               decode_predictions = tf.keras.applications.inception_v3
                
            if self.model_name in ['resnet50','ResNet50','Resnet50']:
    
                print('\n\t\t-------Getting Label For ResNet50----------\n')
                
                decode_predictions = tf.keras.applications.resnet50
            
            if self.model_name in ['Xception','xception']:
    
                print('\n\t\t-------Getting Label For Xception----------\n')
                
                decode_predictions = tf.keras.applications.xception
            
            if self.model_name in ['VGG16','Vgg16','vgg16']:
                
    
                print('\n\t\t-------Getting Label For VGG16----------\n')
                
                decode_predictions = tf.keras.applications.vgg16
        
            if self.model_name in ['VGG19','Vgg19','vgg19']:
                        
            
                print('\n\t\t-------Getting Label For MobileNetV2----------\n')
                
                decode_predictions = tf.keras.applications.vgg19
            if method in ['directory','dataframe','point']:
                if data_generator is None:
                    raise ValueError('Provide Image Generators in generator argument')
                prediction = model.predict_generator(data_generator,steps=len(data_generator),verbose=1)
                predicted_indices = decode_predictions.decode_predictions(prediction,top=self.top)
                return predicted_indices,prediction
            
            elif method in ['image']:
                prediction = model.predict(img)
                predicted_indices = decode_predictions.decode_predictions(prediction,top=self.top)
                return predicted_indices,prediction   
            
    def label_class_provider(self,predictions,predicted_indices,label=None,generator=None,img=None):
        
        if type(label) is list:
            class_names = sorted(label) # Sorting them
            label = dict(zip(class_names, range(len(class_names))))
        print("Following are the Predictions made\n")
        print(predicted_indices)
        if self.model_name is None:
            if generator is not None:
            
                labels_score = []
                for i in range(len(predicted_indices)):#Generating Predictions Labels Scores with their Labels
                    for key,value in label.items():
                        if value == predicted_indices[i]:
                            labels_score.append([generator.filenames[i],key,predictions[i][predicted_indices[i]]])
                return labels_score
            
            else:
                if img is None:
                    raise ValueError('Provide Data_Generator or File name ')
                labels_score = []
                
                for key,value in label.items():
                    
                    if value == predicted_indices:
                        labels_score.append([self.image_directory,key,max(predictions)[0]])
                return labels_score
        else:
             if generator is not None:
            
                labels_score = []
                for i in range(len(predicted_indices)):#Generating Predictions Labels Scores with their Labels
  
                            labels_score.append([generator.filenames[i],predicted_indices[i][0][1],predicted_indices[i][0][2]])
                return labels_score
            
             else:
                if img is None:
                    raise ValueError('Provide Data_Generator or File name ')
                labels_score = []
     
                labels_score.append([self.image_directory,predicted_indices[0][0][1],predicted_indices[0][0][2]])
                return labels_score
        
        
        
            
        