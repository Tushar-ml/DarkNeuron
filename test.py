from Classification.Classification import Classify_Images
import tensorflow as tf
import os



directory = 'C:/Users/Tushar Goel/Desktop'
train_directory = r'C:\Users\Tushar Goel\Desktop\image_class\seg_test\seg_test'
validation_directory = r'C:\Users\Tushar Goel\Desktop\image_class\seg_test\seg_test'
test_directory = r'C:\Users\Tushar Goel\Desktop\image_class\seg_pred'
cnn = Classify_Images(directory,directory)
'''reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=2),
                        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(directory,'model.{epoch:02d}-{val_loss:.2f}.h5')),
                        reduce_lr
                        ]'''

''''train,val,labels = cnn.Preprocess_the_Image(model_name='VGG16',method='directory',train=True,num_classes=6,training_image_directory=train_directory)
model = cnn.Create_the_Model()

model = cnn.Train_the_Model(model=model,rebuild=True,train_data_object=train,validation_data_object=val,epochs=1,steps_per_epoch=5,fine_tuning=False,validation_steps=80,callbacks=False,loss='categorical_crossentropy')
cnn.Visualize_the_Metrics()
model.save('model.h5')'''

test = cnn.Preprocess_the_Image(model_name='InceptionV3',method='directory',num_classes=1000,train=False,test_image_directory=test_directory)
model = cnn.Create_the_Model()
labels = ['buildings','street','glacier','mountain','forest','sea']
print(cnn.Predict_from_the_Model(model=model,generator=test,top=5))
cnn.Visualize_the_Predictions(number=20)