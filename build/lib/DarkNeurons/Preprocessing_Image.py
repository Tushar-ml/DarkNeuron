"""
This Script contains Preprocessing Functions according to different Pre-Trained

architecture.

Different Architectures : Xception, MobileNetV2, InceptionV3, Resnet50

Author: Tushar Goel

"""
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils.np_utils import to_categorical
from .Prediction import Prediction

class Preprocess_Image:
    """
    This Class will be Preprocessing Feature based on differnt Different Models and Differnt Methods for custom image 
    
    training
    
    Attributes:
    -----------
            model_name --> Differnt Architecture Models based On ImageNet 
            num_classes --> 2(Default) Number of Classes or Number of Objects to Process 
             
    Methods:
    --------
           Flow_from_Directory --> Taking Images from Data Directory
           Flow_from_DataFrame --> Taking Filename and Classes from DataFrame **Filename = Image_location
           Flow                --> Taking values x_train,y_train
           
    """
    
    def __init__(self,model_name=None,user_model = None,num_classes=2,batch_size=32,target_image_size=(224,224,3),training=False,method='directory',working_directory=None):
        
        self.user_model=user_model
        self.model_name=model_name
        self.method_name = method
        self.training = training
        if num_classes==2:
            self.class_mode='binary'                    # Creating Class Mode for Image Data Processing
        else:
            self.class_mode = 'categorical'
        self.batch_size = batch_size
        self.target_image_size = target_image_size
            
        print('\t\t----------------------------\n')
        print('\t\t Image Preprocessing Phase\n')
        print('\t\t----------------------------')
        
        self.pred = Prediction(method = self.method_name,user_model=self.user_model,model_name=self.model_name,working_directory = working_directory)
        
        
    def preprocess_architecture_function(self):
        """
        This Function will define preprocess function depends on the basis of the Model Architecture
        
        Arguments:
            None
        
        Output:
            Preprocess_function
        
        """
        #Based on Different Architecture of Models:
        if self.user_model is not None:
            pass
        else:
            if self.model_name is None:
                raise ValueError('Provide Model name or Provide user model')
            if self.model_name in ['mobilenetv2','MobileNetV2','mobilenet_v2','MobileNet_V2']:
                preprocess_function = tf.keras.applications.mobilenet_v2.preprocess_input
                return preprocess_function
            
            elif self.model_name in ['resnet50','ResNet50','Resnet50']:
                preprocess_function = tf.keras.applications.resnet50.preprocess_input
                return preprocess_function
            
            elif self.model_name in ['inceptionv3','InceptionV3','inception_v3','Inception_V3']:
                preprocess_function = tf.keras.applications.inception_v3.preprocess_input
                return preprocess_function
            
            elif self.model_name in ['VGG16','Vgg16','vgg16']:
                preprocess_function = tf.keras.applications.vgg16.preprocess_input
                return preprocess_function
            
            elif self.model_name in ['Xception','xception']:
                preprocess_function = tf.keras.applications.xception.preprocess_input
                return preprocess_function
            
            elif self.model_name in ['VGG19','Vgg19','vgg19']:
                preprocess_function = tf.keras.applications.vgg19.preprocess_input
                return preprocess_function
            else:
                raise ValueError('Invalid Model Name')
        
    def Get_Images_from_Directory(self,training_images_directory=None,validation_images_directory=None,test_image_directory=None):
        
        """
        This Function will Take images from image Directory and convert them to desired
        
        shape and Configurations.
        
        Arguments:
            
            training_images_directory --> Directory Containing Images from Training Folder for Preprocessing
            
            Validation_images_directory --> None (Defalult)--No Validation will be occur.
            
            batch_size --> Batch Size for Data Generator
            
            target_image_size --> Contains Target Size for conversion of Model including 3 channel
            
        Output:
            
            Training_Data_Generator,Validation_data_Generator
            
        """
        
        target_image_size = self.target_image_size
        if len(target_image_size)!=3:
            raise ValueError('Required 3 Arguments --- {} are given'.format(len(target_image_size)))
            
        #Defining Preprocessing Function
        if self.user_model is not None:
            preprocessing_function = None
        else:
            preprocessing_function = self.preprocess_architecture_function()
        # Data Generator Function for preprocessing Function
        data_generator = ImageDataGenerator(preprocessing_function = preprocessing_function,
                                                rescale=1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True)
        
        if self.training:
            if training_images_directory is None:
                raise ValueError('For Training Image Directory is Must, or put training mode to False and use test_image_directory')
            # Train Data Generator for Training
            
            if self.user_model is not None:
                self.target_image_size = self.user_model.input_shape[1:3]
            train_data_generator = data_generator.flow_from_directory(directory = training_images_directory,
                                                                      target_size = (self.target_image_size[0],self.target_image_size[1]),
                                                                      batch_size = self.batch_size,
                                                                      class_mode = self.class_mode)
            #Checking Validation Data Directory , If exists return train and validation data generator
            if validation_images_directory is not None:
                validation_data_generator = data_generator.flow_from_directory(directory = validation_images_directory,
                                                                               target_size = (target_image_size[0],target_image_size[1]),
                                                                               class_mode = self.class_mode)
                return train_data_generator,validation_data_generator
            
            else:
                return train_data_generator,None
        else:
            if test_image_directory is None:
                raise ValueError('Test Image Directory Required for Prediction')
            # Data Generator Function for preprocessing Function
            target_size = self.pred.target_shape_generator()
            data_generator = ImageDataGenerator(
                                                rescale = 1./255)
            test_data_generator = data_generator.flow_from_directory(directory = test_image_directory,
                                                                     target_size = target_size,
                                                                     shuffle=False,
                                                                     batch_size = 1
                                                                     )
            return test_data_generator
            
    def Get_Images_from_DataFrame(self,dataframe,x_column_name,split=0.1,y_column_name=None,image_directory=None):
        """
        This Function will take Filename from Dataframe and Preprocess it from that loacation 
        
        and store corresponding Class value
        
        Arguments:
            
            Dataframe --> This DataFrame Contains Filename of Images and Class Variables
            
            x_column_name --> This Will contain the column name of Filename containg Column
            
            y_column_name --> This will contain the column name of Classes Containing column
                                   
        Outputs:
            
            Training data Generator, Validation Data Generator
            
        """
        
        #Defining Preprocess Function
        preprocessing_function = self.preprocess_architecture_function()
        
        # Data Generator Function for preprocessing Function
        data_generator = ImageDataGenerator(preprocessing_function = preprocessing_function,
                                            rescale = 1./255)
        
        #Splitting Functions for training and Predictions :
        if self.training:
            dataframe_length = dataframe.shape[0]
            
            #Splitting dataframe for Validation and Training Data 
            training_length = round((1-split)*dataframe_length)
            training_dataframe = dataframe[:training_length]
            validation_dataframe = dataframe[training_length:]
            validation_dataframe.columns = dataframe.columns.values
            
            if y_column_name is None:
                raise ValueError(' For Training Y Columns is required')
            # Train Data Generator for Training
            if self.user_model is not None:
                self.target_image_size = self.user_model.input_shape[1:3]
            train_data_generator = data_generator.flow_from_dataframe(dataframe=training_dataframe,
                                                                      directory = image_directory,
                                                                      x_col = x_column_name,
                                                                      y_col = y_column_name,
                                                                      target_size = (self.target_image_size[0],self.target_image_size[1]),
                                                                      class_mode = self.class_mode,
                                                                      
                                                                      batch_size = self.batch_size)
            try:
                validation_data_generator = data_generator.flow_from_dataframe(dataframe=validation_dataframe,
                                                                          directory = image_directory,
                                                                          x_col = x_column_name,
                                                                          y_col = y_column_name,
                                                                          target_size = (self.target_image_size[0],self.target_image_size[1]),
                                                                          class_mode = self.class_mode,
                                                                
                                                                          batch_size = self.batch_size)
                print('-----Validation and Train Data Generated----------')
                return train_data_generator,validation_data_generator
            except:
                return train_data_generator,None
        
        else:
            # Test Data Generator 
            target_size = self.pred.target_shape_generator()
            test_data_generator = data_generator.flow_from_dataframe(dataframe=dataframe,
                                                                      directory = image_directory,
                                                                      x_col = x_column_name,
                                                                      y_col = y_column_name,
                                                                      target_size = target_size,
                                                                      shuffle=False,
                                                                      batch_size = 1
                                                                     )
            return test_data_generator
        
    def Get_Data(self,x_train=None,y_train=None,x_test=None,y_test=None):
        """
        This Function will take value from arrays containing Data and Labels
        
        Arguments:
            x_training --> Containg Image Data in form of arrays
            y --> Containg Labels for Image Data
            
        Output:
            Train Data generator , Validation Data Generator
            
        """
        #Defining Preprocess Function
        preprocessing_function = self.preprocess_architecture_function()
        
        # Data Generator Function for preprocessing Function
        data_generator = ImageDataGenerator(preprocessing_function,
                                            rescale = 1./255)
        if len(x_train.shape)==3:
            x_train = np.expand_dims(x_train,axis=-1)
            x_test = np.expand_dims(x_test,axis=-1)
        
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        if self.training:
             # Train Data Generator for x_train,y_train
            if y_train is None:
                raise ValueError('For Training, Classes are required')
            train_data_generator = data_generator.flow(x = x_train,
                                                       y = y_train,
                                                
                                                        batch_size = self.batch_size)
            validation_data_generator = data_generator.flow(x = x_test,
                                                       y = y_test,
                                                    
                                                       batch_size = self.batch_size)
            return train_data_generator,validation_data_generator
        
        else:
            test_data_generator = data_generator.flow(x = x_test,
                                                       y = y_test
                                                       )
            
            return test_data_generator
        
    
    def Get_Image(self,image_path,model_name = None,user_model= None,grayscale = False):
        """
        This Function will be for Image Processing for Single Images not on any Directory
        
        Arguments:
            model_name --> Model Name for PreDefined Architecture: None(Default)
            user_model --> Path to user own Model if they have Predefined Models : None(Default)
            
            target_image_size --> This will be the target image size user model or Predefined Architecture Model
            image_path --> Image Path which is to be Predicted
    
        Returns:
            
            This Function will Return the Preprocessed Image
            
        """
        target_shape = self.pred.target_shape_generator()
        
        if user_model is not None:
            
            img = image.load_img(image_path,grayscale=grayscale,target_size=target_shape)
            img = image.img_to_array(img)
            img = np.expand_dims(img,axis=0)
            
            return img
        
        else:
             # Recalling Function from Above for different models
            
            img = image.load_img(image_path,grayscale=grayscale,target_size=target_shape)
            img = image.img_to_array(img)
            img = np.expand_dims(img,axis=0)
            
            return img

            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
