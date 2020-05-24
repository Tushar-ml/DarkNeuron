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
import cv2                          # Computer Vision Library
import  tensorflow as tf            # Powerful Framework for Deep Learning
import keras                        # A Deep Learning API 
import os                           # For Searching Folder within the system
from models import Models           # Script containing Different Models
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
        --Predict_the_Model
        --Generate_the_Model
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
        self.loss = 'sparse_categorical_crossentropy' 
        self.optimizer = 'adam'
        self.train = train
        self.target_image_size = target_image_size
        
<<<<<<< HEAD

            
=======
        print('\t\t--------------------------------------')
        print('\t\t| Step:1 Call Preprocess_the_Image() |')
        print('\t\t--------------------------------------')
    """
    Defining Preprocess Function to Preprocess the Images with Different Flow Method
    
    """
    def Preprocess_the_Image(self,model_name,num_classes,batch_size,method,training_image_directory=None,validation_image_directory=None,dataframe=None,
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
                print('\n-----Training Data Generated------\n')
                
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
            


            
    
            
            
            
            
            
        
        
        
    
        
        

