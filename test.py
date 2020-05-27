from Classification.Classification import Classify_Images
import tensorflow as tf
import pandas as pd
directory = 'C:/Users/Tushar Goel/Desktop/'
cnn = Classify_Images(directory,directory)
# model = cnn.load_model('inceptionv3_model.h5')
df = pd.read_csv('C:/Users/Tushar Goel/Desktop/animals.csv')
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
train,val,labels = cnn.Preprocess_the_Image(train=True,num_classes =2,method='dataframe',batch_size=32,model_name='VGG16',dataframe=df,image_directory='C:/Users/Tushar Goel/Desktop/animals',x_col='file',y_col='classes')
model = cnn.Create_the_Model()
model = cnn.Train_the_Model(model=model,rebuild=True,train_data_object = train,validation_data_object = val,epochs=5,steps_per_epoch=15)
cnn.Visualize_the_Metric()
test = cnn.Preprocess_the_Image(user_model=model,method='directory',train=False,num_classes=2,test_image_directory=r'C:/Users/Tushar Goel/Desktop/cat-and-dog/test_set/test_set')
print(cnn.Predict_from_the_Model(labels=labels,generator=test,model=model))
cnn.Visualize_the_Predictions(number=9)