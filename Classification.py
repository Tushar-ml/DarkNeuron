""" Using this File, we will be classifying Images using Predefined Models

and Scratch Models .

Author: Tushar Goel


Different Architectures:
    --> InceptionV3
    -->Xception
    --> VGG16
    --> VGG19
    --> Resnet50
    --> MobileNetV2

"""
from Deep_Stack import Deep_Stack
import  tensorflow as tf            # Powerful Framework for Deep Learning
import keras                        # A Deep Learning API 
import os                           # For Searching Folder within the system
from models import Create_Model, Train_Model           # Script containing Different Models
from Preprocessing_Image import Preprocess_Image      #Preprocessing Image Script
 
class CNN(Deep_Stack):
    """
    This Class will have Following Properties:
        
    Attributes:
        --Working Directory
        -- Output Directory
        -- train --> Whether to train to predict
    Methods:
        --Preprocess_the_Images
        --Create_the_Model
        --Train_the_Model
        --Predict_from_the_Model
        --Visualize_the_Model
        --Deploy_the_Model
    """
    def __init__(self,working_directory,output_directory,target_image_size=(224,224,3),train=False):
        """
        In this function we will call Parent Function containing other Function
        
        and Define other variables.
        
        Arguments:
        ---------    
            working_directory --> Directory Containing Raw Data
            
            output_directory --> Output Sirectory to which Results will be posted
            
            Image_Folder_name --> Different Model Name for Different Models
            
            Train --> False Or True (For Prediction)
            
        Output:
        ------    
            None
        
        """
        Deep_Stack.__init__(self,working_directory,output_directory)
        self.epochs = 10                    #Initializing Epochs
        self.loss = 'binary_crossentropy' 
        self.optimizer = 'adam'
        self.train = train
        self.target_image_size = target_image_size
<<<<<<< HEAD
        
<<<<<<< HEAD

            
=======
        print('\t\t--------------------------------------')
        print('\t\t| Step:1 Call Preprocess_the_Image() |')
        print('\t\t--------------------------------------')
=======
        self.working_directory = working_directory
        self.output_directory = output_directory

>>>>>>> model_class_creation
    """
    Defining Preprocess Function to Preprocess the Images with Different Flow Method
    
    """
    def Preprocess_the_Image(self,model_name,num_classes,method,batch_size=32,training_image_directory=None,validation_image_directory=None,dataframe=None,
                            test_image_directory=None,x_train=None,x_test=None,y_train=None,y_test=None,x_col=None,y_col = None,split=0.1,image_directory=None):
        """
        This function Will do image processing and return training Data Generator, Validation Data Generator
        
        and Test Data Generator on the Basis of Training Argument whether it is True Or False.
        
        Arguments:
            model_name --> Name for The Predefined Architecture
            num_classes --> Number of Classes
            batch_size --> Batch Size
            method --> Method by which Images will flow in the Function --> directory,dataframe,point
            training_image_directory --> Directory for method Directory
            validation_image_directory --> (Optional) For method Directory
            
            
        Outputs:
            It will Return the Data Generator for Train and Test
        
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.method = method
        self.training_directory = training_image_directory
        self.validation_directory = validation_image_directory
        self.test_directory = test_image_directory
        self.dataframe = dataframe
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_col_name = x_col
        self.y_col_name = y_col
        self.split = split
        self.image_directory = image_directory
        #Defining Variables for Preprocessing
        preprocessing = Preprocess_Image(self.model_name,self.num_classes,self.batch_size,self.target_image_size,self.train)
        
        #Getting results based on Different flow methods
        if self.method == 'directory':
            print('\n\t\t-----Getting Images From Directory------\n')
            if self.train:# From Preprocessing_Image.py File
                train_data,validation_data = preprocessing.Get_Images_from_Directory(self.training_directory,self.validation_directory,self.test_directory)
                print('\n\t\t-------Training Data Generated--------\n')
                return train_data,validation_data
            else:
                test_data = preprocessing.Get_Images_from_Directory(self.training_directory,self.validation_directory,
                                                                    self.test_directory)
                print('\n\t\t-------Test Data Generated--------\n')
                return test_data
            
        elif self.method=='dataframe':
            print('\n\t\t-----Getting Images From DataFrame------\n')
            if self.train:
                train_data,validation_data = preprocessing.Get_Images_from_DataFrame(self.dataframe,self.x_col_name,self.split,self.y_col_name,self.image_directory)
                print('\n\t\t-----Training Data Generated------\n')
                
                return train_data,validation_data
            
            else:
                test_data = preprocessing.Get_Images_from_DataFrame(self.dataframe,self.x_col_name,self.split,self.y_col_name,self.image_directory)
                print('\n\t\t-----Test Data Generated------\n')
                
                return test_data
        
        elif self.method=='point':
            print('\n\t\t-----Getting Images From Points------\n')
            if self.train:
                train_data,validation_data = preprocessing.Get_Data(self.x_train,self.y_train,self.x_test,self.y_test)
                print('\n\t\t------Training Data Generated-------\n')
                return train_data,validation_data
            else:
                test_data = preprocessing.Get_Data(self.x_train,self.y_train,self.x_test,self.y_test)
                print('\n\t\t---------Test Data Generated--------\n')
                return test_data
            
        else:
            
            raise ValueError('Invalid Method Input --Must be from "directory","dataframe","point"')
            
<<<<<<< HEAD


=======
    def Create_the_Model(self):
            
        """
        This Function will be used for Initialisation of Model according to Model name Given
        
        Arguments:
            None
>>>>>>> model_class_creation
            
        Returns:
            It will return the model for Training the model
            
        """
        print('\n\t\t--------------Model Creation Phase-----------\n')
        
        model_init = Create_Model(self.working_directory,self.target_image_size,self.train)
        
        # Defining Model based on Model name:
        if self.model_name in ['mobilenetv2','MobileNetV2','mobilenet_v2','MobileNet_V2']:
            
            # Checking whether Target Image size is within bounds for Predefined Architecture
            if self.target_image_size[0] <32 or self.target_image_size[1]<32:
                    Model_Target_Value_Checker()  #Check the Function Below which Raise the Value Error
                                 
            print('\n\t\t-------MobileNetV2 Model Initiated Successfully----------\n')
            return model_init.MobileNetV2()
            
        if self.model_name in ['inceptionv3','InceptionV3','inception_v3','Inception_V3']:
            
            # Checking whether Target Image size is within bounds for Predefined Architecture
            if self.target_image_size[0] <75 or self.target_image_size[1]<75:
                    Model_Target_Value_Checker()  #Check the Function Below which Raise the Value Error
            print('\n\t\t-------InceptiontV3 Model Initiated Successfully----------\n')                     
            return model_init.InceptionV3()
            
        if self.model_name in ['resnet50','ResNet50','Resnet50']:
            
            # Checking whether Target Image size is within bounds for Predefined Architecture
            if self.target_image_size[0] <32 or self.target_image_size[1]<32:
                    Model_Target_Value_Checker()  #Check the Function Below which Raise the Value Error
            print('\n\t\t-------Resnet50 Model Initiated Successfully----------\n')                     
            return model_init.ResNet50()
        
        if self.model_name in ['Xception','xception']:
            
            # Checking whether Target Image size is within bounds for Predefined Architecture
            if self.target_image_size[0] <71 or self.target_image_size[1]<71:
                    Model_Target_Value_Checker()  #Check the Function Below which Raise the Value Error
            print('\n\t\t-------Xception Model Initiated Successfully----------\n')                     
            return model_init.Xception()
        
        if self.model_name in ['VGG16','Vgg16','vgg16']:
            
            # Checking whether Target Image size is within bounds for Predefined Architecture
            if self.target_image_size[0] <32 or self.target_image_size[1]<32:
                    Model_Target_Value_Checker()  #Check the Function Below which Raise the Value Error
            print('\n\t\t-------VGG16 Model Initiated Successfully----------\n')                     
            return model_init.VGG16()
    
        if self.model_name in ['VGG19','Vgg19','vgg19']:
                    
                    # Checking whether Target Image size is within bounds for Predefined Architecture
                    if self.target_image_size[0] <32 or self.target_image_size[1]<32:
                            Model_Target_Value_Checker()  #Check the Function Below which Raise the Value Error
                    print('\n\t\t-------VGG19 Model Initiated Successfully----------\n')                     
                    return model_init.VGG19()            
            
            
            
            
        
    def Train_the_Model(self,model,train_data_object=None,validation_data_object=None,test_data_object=None,epochs = 10,optimizer='adam',loss = 'binary_crossentropy',fine_tuning = False,layers = 20,metrics='accuracy',save_model = True):
        """
        This function will call up the Initialised Model 
        
        """
        
        print('\n\t\t------------Model Training To be Start---------------')
        history,model = Train_Model(model=model,num_classes = self.num_classes,train_data_object=train_data_object,model_name = self.model_name,
                                    working_directory = self.working_directory,output_directory = self.output_directory,loss = loss,epochs=epochs,
                                    optimizer = optimizer,metrics = metrics,validation_data_object = validation_data_object,fine_tuning = fine_tuning,
                                    layers = layers,save_model=save_model)
        self.model_history = history
        return history,model
        
        
def Model_Target_Value_Checker():
        raise ValueError('Try with Different Model.Get '
             'information on Keras Documentation\n'
             'The Lowest Dimensions allowed for Different Model are : \n'
             'Try to change in Preprocess Images Process \n'
             'MobileNetV2 --> (32,32) \n'
             'InceptionV3 --> (75,75)\n'
             'Resnet50 --> (32,32) \n'
             'Xception --> (71,71) \n'
             'VGG16 --> (32,32) \n'
             'VGG19 --> (32,32) \n'
             )

import pandas as pd
df = pd.read_csv('C:/Users/Tushar Goel/Desktop/animals.csv')
directory = 'C:/Users/Tushar Goel/Desktop'
image_directory = 'C:/Users/Tushar Goel/Desktop/Animals'
cnn = CNN(directory,directory,(224,224,3),True)
train,val = cnn.Preprocess_the_Image('Xception',2,'dataframe',32,dataframe=df,x_col='Filename',y_col='classes',image_directory=image_directory)
model = cnn.Create_the_Model()
#print(model.summary())
history,model = cnn.Train_the_Model(model=model,train_data_object = train,validation_data_object = val,fine_tuning =False ,layers='all',epochs=5)
