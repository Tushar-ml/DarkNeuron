"""
This Script contains Preprocessing Functions according to different Pre-Trained

architecture.

Different Architectures : Xception, MobileNetV2, InceptionV3, Resnet50

Author: Tushar Goel

"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Preprocess_Image:
    """
    This Class will be Preprocessing Feature based on differnt Different Models and Differnt Methods for custom image 
    
    training
    
    Attributes:
    -----------
            model_name --> Differnt Architecture Models based On ImageNet 
            num_classes --> 2(Default) Number of Classes or Number of Objects to  
             
    Methods:
    --------
           Flow_from_Directory --> Taking Images from Data Directory
           Flow_from_DataFrame --> Taking Filename and Classes from DataFrame **Filename = Image_location
           Flow                --> Taking values x_train,y_train
           
    """
    
    def __init__(self,model_name=None,num_classes=2,batch_size=32,target_image_size=(224,224,3),training=False):
        if model_name==None:
            raise ValueError('Define The Predefine Architecture Name. It is a required Variable for Training')
        
        self.model_name=model_name.lower()
        
        self.training = training
        if num_classes==2:
            self.class_mode='binary'                    # Creating Class Mode for Image Data Processing
        else:
            self.class_mode = 'categorical'
        self.batch_size = batch_size
        self.target_image_size = target_image_size
            
        print('\t\t----------------------------\n')
        print('\t\t  Image Preprocessing Phase\n')
        print('\t\t----------------------------')
        
        
        
    def preprocess_architecture_function(self):
        """
        This Function will define preprocess function depends on the basis of the Model Architecture
        
        Arguments:
            None
        
        Output:
            Preprocess_function
        
        """
        #Based on Different Architecture of Models:
        if self.model_name in ['mobilenetv2','mobilenet_v2']:
            preprocess_function = tf.keras.applications.mobilenet_v2.preprocess_input
            return preprocess_function
        
        if self.model_name in ['resnet50']:
            preprocess_function = tf.keras.applications.resnet50.preprocess_input
            return preprocess_function
        
        if self.model_name in ['inceptionv3','inception_v3']:
            preprocess_function = tf.keras.applications.inception_v3.preprocess_input
            return preprocess_function
        
        if self.model_name in ['vgg16']:
            preprocess_function = tf.keras.applications.vgg16.preprocess_input
            return preprocess_function
        
        if self.model_name in ['xception']:
            preprocess_function = tf.keras.applications.xception.preprocess_input
            return preprocess_function
        
        if self.model_name in ['vgg19']:
            preprocess_function = tf.keras.applications.vgg19.preprocess_input
            return preprocess_function
        
    def Get_Images_from_Directory(self,training_images_directory,validation_images_directory=None):
        
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
        preprocessing_function = self.preprocess_architecture_function()
        
        # Data Generator Function for preprocessing Function
        data_generator = ImageDataGenerator(preprocessing_function = preprocessing_function,
                                                reshape = 1./255)
        
        if self.training:
            
            # Train Data Generator for 
            train_data_generator = data_generator.flow_from_directory(directory = training_images_directory,
                                                                      target_size = (target_image_size[0],target_image_size[1]),
                                                                      batch_size = self.batch_size,
                                                                      class_mode = self.class_mode)
            #Checking Validation Data Directory , If exists return train and validation data generator
            if not validation_images_directory:
                validation_data_generator = data_generator.flow_from_directory(directory = validation_images_directory,
                                                                               target_size = (target_image_size[0],target_image_size[1]))
                return train_data_generator,validation_data_generator
            
            return train_data_generator,None
        else:
            # Data Generator Function for preprocessing Function
            data_generator = ImageDataGenerator(preprocessing_function = preprocessing_function,
                                                reshape = 1./255)
            test_data_generator = data_generator.flow_from_directory(directory = training_images_directory,
                                                                     target_size = (target_image_size[0],target_image_size[1])
                                                                     )
            return test_data_generator
            
    def Get_Images_from_Dataframe(self,dataframe,x_column_name,split=0.2,y_column_name=None,image_directory=None):
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
        data_generator = ImageDataGenerator(preprocessing_function = preprocessing_function)
        
        #Splitting Functions for training and Predictions :
        if self.training:
            dataframe_length = dataframe.shape[0]
            
            #Splitting dataframe for Validation and Training Data 
            training_length = round((1-split)*dataframe_length)
            training_dataframe = dataframe[:training_length+1]
            validation_dataframe = dataframe[training_length+1:]
            
            # Train Data Generator for 
            train_data_generator = data_generator.flow_from_dataframe(dataframe=training_dataframe,
                                                                      directory = image_directory,
                                                                      x_col = x_column_name,
                                                                      y_col = y_column_name,
                                                                      target_size = (self.target_size[0],self.target_size[1]),
                                                                      class_mode = self.class_mode,
                                                                      batch_size = self.batch_size)
            validation_data_generator = data_generator.flow_from_dataframe(dataframe=validation_dataframe,
                                                                      directory = image_directory,
                                                                      x_col = x_column_name,
                                                                      y_col = y_column_name,
                                                                      target_size = (self.target_size[0],self.target_size[1]),
                                                                      class_mode = self.class_mode,
                                                                      batch_size = self.batch_size)
            return train_data_generator,validation_data_generator
        
        else:
            # Test Data Generator 
            test_data_generator = data_generator.flow_from_dataframe(dataframe=dataframe,
                                                                      directory = image_directory,
                                                                      x_col = x_column_name,
                                                                      target_size = (self.target_size[0],self.target_size[1]),
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
        data_generator = ImageDataGenerator(preprocessing_function = preprocessing_function)
        
        if self.training:
             # Train Data Generator for x_train,y_train
            train_data_generator = data_generator.flow(x = x_train,
                                                       y = y_train,
                                                       target_size = (self.target_size[0],self.target_size[1]),
                                                       class_mode = self.class_mode,
                                                       batch_size = self.batch_size)
            validation_data_generator = data_generator.flow(x = x_test,
                                                       y = y_test,
                                                       target_size = (self.target_size[0],self.target_size[1]),
                                                       class_mode = self.class_mode,
                                                       batch_size = self.batch_size)
            return train_data_generator,validation_data_generator
        
        else:
            test_data_generator = data_generator.flow(x = x_test,
                                                       y = None,
                                                       target_size = (self.target_size[0],self.target_size[1]),
                                                       )
            
            return test_data_generator
        
    
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
