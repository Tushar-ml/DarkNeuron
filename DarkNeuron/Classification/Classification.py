""" Using this File, we will be classifying Images using Predefined Models

and Scratch Models .

Author: Tushar Goel


Different Architectures:
    --> InceptionV3
    -->Xception
    --> VGG16
    --> VGG19
    --> Resnet50

"""
from .Dark_Neuron_CNN import Dark_Neuron
import  tensorflow as tf            # Powerful Framework for Deep Learning
import os                           # For Searching Folder within the system
from .models import Create_Model, Train_Model           # Script containing Different Models
from .Preprocessing_Image import Preprocess_Image      #Preprocessing Image Script
from .Prediction import Prediction
import matplotlib.pyplot as plt

 
class Classify_Images(Dark_Neuron):
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
        --Visualize_the_Metric
        
    """
    def __init__(self,working_directory,output_directory):
        """
        In this function we will call Parent Function containing other Function
        
        and Define other variables.
        
        Arguments:
        ---------    
            working_directory --> Directory Containing Raw Data
            
            output_directory --> Output Sirectory to which Results will be posted
            
        Output:
        ------    
            None
        
        """
        Dark_Neuron.__init__(self,working_directory,output_directory)
        self.working_directory = working_directory
        self.output_directory = output_directory
        
    def load_model(self,user_model_name):
        user_model_path = os.path.join(self.working_directory,user_model_name)
        return tf.keras.models.load_model(user_model_path)

    """
    Defining Preprocess Function to Preprocess the Images with Different Flow Method
    
    """
    def Preprocess_the_Image(self,method,train,num_classes=2,batch_size=32,target_image_size=(224,224,3),model_name = None,user_model = None,image_path=None,grayscale=None,training_image_directory=None,validation_image_directory=None,dataframe=None,
                            test_image_directory=None,x_train=None,x_test=None,y_train=None,y_test=None,x_col=None,y_col = None,split=0.1,image_directory=None,input_tensor=None):
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
        self.train = train
        self.target_image_size = target_image_size
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
        self.input_tensor = input_tensor
        self.image_path = image_path
        if user_model is not None:
            self.user_model = user_model
        else:
            self.user_model = None
        #Defining Variables for Preprocessing
        preprocessing = Preprocess_Image(model_name=self.model_name,user_model=self.user_model,target_image_size = self.target_image_size,num_classes = self.num_classes,batch_size=self.batch_size,training=self.train,method=self.method,working_directory = self.working_directory)
        
        #Getting results based on Different flow methods
        if self.method == 'directory':
            print('\n\t\t-----Getting Images From Directory------\n')
            if self.train:# From Preprocessing_Image.py File
                train_data,validation_data = preprocessing.Get_Images_from_Directory(self.training_directory,self.validation_directory,self.test_directory)
                print('\n\t\t-------Training Data Generated--------\n')
                return train_data,validation_data,(train_data.class_indices)
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
                
                return train_data,validation_data,(train_data.class_indices)
            
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
            
        elif self.method == 'image':
            if image_path is None:
                raise ValueError('Provide Image Path for Image Prediction or If it is containing in a directory having multiple ',
                                 'images, then set method = "directory"')
            print('\n\t\t----------Getting Image -------------\n')
            image = preprocessing.Get_Image(image_path = image_path,model_name = self.model_name,user_model=user_model,
                                            grayscale=grayscale)
            return image
       
        else:
            
            raise ValueError('Invalid Method Input --Must be from "directory","dataframe","point","image"')
            
            



    def Create_the_Model(self):
            
        """
        This Function will be used for Initialisation of Model according to Model name Given
        
        Arguments:
            None

            
        Returns:
            It will return the model for Training the model
            
        """
        print('\n\t\t--------------Model Creation Phase-----------\n')
    
        
        model_init = Create_Model(working_directory=self.working_directory,image_shape = self.target_image_size,train = self.train,input_tensor=self.input_tensor)
        
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
            
            
            
            
        
    def Train_the_Model(self,model,rebuild=False,train_data_object=None,validation_data_object=None,test_data_object=None,epochs = 10,optimizer='adam',loss = 'binary_crossentropy',fine_tuning = False,layers = 20,metrics='accuracy',validation_steps=80,save_model = True,steps_per_epoch = 50,callbacks=None):
        """
        This function will call up the Initialised Model 
        
        """
        
        print('\n\t\t------------Model Training To be Start---------------')
        history,model = Train_Model(model=model,rebuild=rebuild,num_classes = self.num_classes,train_data_object=train_data_object,
                                    working_directory = self.working_directory,output_directory = self.output_directory,loss = loss,epochs=epochs,
                                    optimizer = optimizer,metrics = metrics,validation_data_object = validation_data_object,fine_tuning = fine_tuning,
                                    layers = layers,validation_steps=validation_steps,save_model=save_model,steps_per_epoch = steps_per_epoch,callbacks=callbacks)
        self.model_history = history
        return model
    
    def Visualize_the_Metrics(self):
        import matplotlib.pyplot as plt
        # Plot for Training Loss and Training Accuracy
        plt.plot(self.model_history.history['loss'],label='Training Loss')
        plt.plot(self.model_history.history['acc'],label = 'Training Accuracy')
        plt.title('Training Loss vs Training Accuracy')
        plt.legend()
        plt.show()
        #excetuted when validation set will be there
        try:
            # Training Loss vs Vaidation Loss 
            plt.plot(self.model_history.history['val_loss'],label='Test Loss')
            plt.plot(self.model_history.history['loss'],label = 'Training Loss')
            plt.title('Training Loss vs Validation Loss')
            plt.legend()
            plt.show()
            
            plt.plot(self.model_history.history['acc'],label='Training Accuracy')
            plt.plot(self.model_history.history['val_acc'],label = 'Validation Accuracy')
            plt.title('Training Accuracy vs Validation Accuracy')
            plt.legend()
            plt.show()
            
            plt.plot(self.model_history.history['val_loss'],label='Validation_Loss')
            plt.plot(self.model_history.history['val_acc'],label = 'Validation_Accuracy')
            plt.title('Validation Loss vs Validation Accuracy')
            plt.legend()
            plt.show()
        
        except:
            pass
    
    def Predict_from_the_Model(self,labels=None,generator=None,img = None,top = 5,model=None):
        """
        This Function will be used to predict the classes from Model
        
        Arguments:
            preprocessed_image --> preprocessed_image suitable for model
            model --> model get from trained part
            top --> Number of High Probabilities
            
        Return:
            Classes
        
        """
        self.generator = generator
        self.img = img
        prediction = Prediction(working_directory = self.working_directory,labels = labels,method = self.method,
                                model_name = self.model_name,user_model=self.user_model,
                                img = img,top=top,image_directory=self.image_path)
        
        if self.user_model is not None:
            model = self.user_model
        else:
            if model is None:
                raise ValueError('Provide Model, model argument should not be empty')
            model = model
        predicted_indices,predictions = prediction.prediction(method=self.method,model=model,img=img,data_generator=generator)
        
        label_score = prediction.label_class_provider(label=labels,predictions=predictions,predicted_indices=predicted_indices,
                                                      generator=generator,img=img)
        print('\n\t\t--------------Generating Predictions with Score----------------')
        
        self.label_score = label_score
        if len(label_score) == 0:
            print('\n\t\t----------No Predictions-----------')
            return label_score
        
        else:
            print(f'\n\t\t------------Found {len(label_score)} Predicitons-------' )
            return label_score
        
    def Visualize_the_Predictions(self,number=6):
        if number > len(self.label_score):
            number = len(self.label_score)
            
        if number ==0:
            print('No predictions to Show')
       
        
        else:
            if self.generator is not None:
                for label_score in self.label_score[:number]:
                    filepath = os.path.join(self.test_directory,label_score[0])
                    img = plt.imread(filepath)
                    plt.imshow(img)
                    plt.title(f'Predicted:{label_score[1].title()} ---- Score: {label_score[2]*100}')
                    plt.show()
            elif self.img is not None:
                for label_score in self.label_score[:number]:
                    filepath = label_score[0]
                    img = plt.imread(filepath)
                    plt.imshow(img)
                    plt.title(f'Predicted:{label_score[1].title()} ---- Score: {label_score[2]*100}')
                    plt.show()
            
            
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


