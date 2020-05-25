""" This Python Script Contains all Predefined ImageNet Models with Their Implementations

which will be directed towards Classification File

Author : Tushar Goel

"""
import tensorflow as tf
import os
import wget
import sys


class Create_Model:
    """
    This Class contains functions of Different models which will return model
    
    Arguments:
        working directory --> Working Directory where Material is present
        image_shape --> Default Shape: (224,224,3)
        train --> False(default), whether to predict or train the model
        
    Methods:
        Different Architecture Model:
            InceptionV3
            Xception
            VGG16
            ResNet50
            VGG19
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


def Train_Model(model,num_classes,train_data_object,model_name,working_directory,output_directory,optimizer,loss,epochs,metrics,validation_data_object=None,fine_tuning=False,
                layers = 20 , save_model = True,steps_per_epoch = 50):
    """
    This Function will be used to train the Model and save the model to Output Directory
    
    Arguments:
        Model --> This will be the Initiated Model returned from Create Model Class
        fine_tuning --> whether to unfreeze the layers and tune them
        layers --> no of layers to unfreeze : 20(Default)
        save_model --> True(Default) whether to save Model or not
        train_data_object --> Training Data Generated from Preprocessing the Function
        
    Returns :
        Trained Model
    """
    if num_classes == 2:
        target = num_classes - 1
        activation = 'sigmoid'
    else:
        target = num_classes
    activation = 'softmax'
    model.trainable = False    
    if fine_tuning:
        #Fine tuning for Increasing the accuracy of the model
        
        model.trainable = True #Unfreeze all layer
        layers_length = len(model.layers)
        
        if layers == 'all':
            model.trainable = True
        # Lets Freeze the Bottom Layers:
        else:
            freeze_layer_length = layers_length - layers
            for layers in model.layers[:freeze_layer_length]:
                layers.trainable = False
            
            for layers in model.layers[freeze_layer_length:]:
                layers.trainable = True

    log_directory = os.path.join(working_directory,'\logs')
    model_checkpoint_directory = os.path.join(working_directory,'\model_checkpoint')            
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
    
    if not os.path.exists(model_checkpoint_directory):
        os.mkdir(model_checkpoint_directory)
    
    layer1 = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    output = tf.keras.layers.Dense(target,activation=activation)(layer1)
    
    New_Model = tf.keras.models.Model(inputs = model.input , outputs = output)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=2),
                    #tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_checkpoint_directory,'model.{epoch:02d}-{val_loss:.2f}.h5')),
                    tf.keras.callbacks.TensorBoard(log_dir=log_directory),
                    reduce_lr
                    ]
    
    New_Model.compile(loss = loss,optimizer = optimizer,metrics = [metrics])
    if validation_data_object is None:
        
        history = New_Model.fit_generator(train_data_object,steps_per_epoch = 50,epochs= epochs, callbacks = my_callbacks)
        
    else:
        
        history = New_Model.fit_generator(train_data_object,steps_per_epoch = 3,epochs=epochs,validation_data = validation_data_object,
                            callbacks = my_callbacks)
        

    return history,New_Model

 
def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()
        