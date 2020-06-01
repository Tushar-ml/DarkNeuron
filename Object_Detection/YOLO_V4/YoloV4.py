"""

This File will contain the implementation of YOLOV3 Algorithm,

containing Image_Annotation, Training and Prediction

Author: Tushar Goel

"""
import os
from Yolo_Format import Image_Annotation
from Download_Convert_Yolo_Weights import Yolo4_weights,Download_weights
from Model_Training import Train_Yolo
from Detection import Detector

class YOLOv4:
    """
    This Class Will provide full implementation of YOLOv3 
    
    within simple steps.
    
    Attributes:
        working_directory --> The Directory where all Images and Labels have been provided
        
    Method:
        Prepare_the_Data --> This Method Will be used to Convert your raw into YOLO Format.
        
        Train_the_Yolo --> This Will Train the YOLOv3 For Users with no additional arguments
        
        Predict_from_Yolo --> It will help in Detection of objects whether it is Image or Video.
        
    """
    
    def __init__(self,working_directory):
        
        # Creating Attribute Working Directory
        self.working_directory = working_directory
        
    
    def Prepare_the_Data(self,file_type,dataframe_name=None,class_file_name=None):
        """
        This Function will Prepare the Data according to the format of the file
        
        Arguments:
            
            file_type --> This Function will take dataframe,text or xml format
            dataframe_name --> Name of the Dataframe name in working directory. Only Needed when method = dataframe.
            class_file_name --> class_file_name containing label file names in .txt format
            
        Returns:
            None
        
        Additional Files Formed:
            data_train.txt --> Yolo Format of the Data
            data_classes.txt --> Classes of the Data
            
        """
            
        self.file_type = file_type
        # Method Involves 'dataframe','xml','text'
        Image_Annot = Image_Annotation(working_directory = self.working_directory,dataframe_name=dataframe_name)
        if self.file_type not in ['dataframe','xml','text']:
            raise ValueError('Method either should be dataframe,xml,text')
            
        if self.file_type == 'dataframe':
            
            Image_Annot.Convert_to_Yolo_Format()
            
        elif self.file_type == 'xml':
            
            df = Image_Annot.csv_from_xml()
            Image_Annot.Convert_to_Yolo_Format(df=df)

        elif self.file_type == 'text':
            if class_file_name is None:
                raise ValueError('Provide Class File Name for method text')
            df = Image_Annot.csv_from_text(class_list_file_name = class_file_name)
            Image_Annot.Convert_to_Yolo_Format(df=df)
            
    def Train_the_Yolo(self,plot_model=False,save_weights=False,score = 0.5,iou = 0.5,
                       epochs1 = 51,epochs2 = 50, batch_size1 = 32,batch_size2 = 4,gpu_num = 1,
                       validation_split = 0.1):
        
        yolo_file_path = os.path.join(self.working_directory,'yolov4.h5')
        self.class_path = os.path.join(self.working_directory,'class_list.txt')
        self.anchors_path = 'yolo4_anchors.txt'
        self.weight_path = os.path.join(self.working_directory,'yolov4.weights')
        #Checking whether User have Yolo File or Not 
        #If no File, then it will be downloaded Automatically and Converted to Keras Model
        if not os.path.exists(yolo_file_path):
            
            Download_weights(working_directory = self.working_directory)
            yolov4 = Yolo4_weights(score=score,iou=iou,anchors_path = self.anchors_path,classes_path = self.class_path,
                                   model_path = yolo_file_path,weights_path = self.weight_path,gpu_num = gpu_num)
            yolov4.load_yolo()

        
        print('Model Training to be Start ....')
        history = Train_Yolo(working_directory = self.working_directory,val_split = validation_split,
                   epochs1=epochs1,epochs2 = epochs2,batch_size1 = batch_size1,
                   batch_size2 = batch_size2)
        
        
        self.history = history
        
    def Detect(self,test_folder_name,model_name = None,real_time = False,tracking = None,classes = None,score=0.5,gpu_num = 1):
        
        Detector(working_directory = self.working_directory,test_folder_name = test_folder_name,
                 classes = classes,score = score,model_name = model_name,gpu_num = gpu_num)
    
working_directory = r'C:\Users\TusharGoel\Desktop\OpenLabeling-master\images'       
yolo = YOLOv4(working_directory)
#yolo.Prepare_the_Data(file_type='text',class_file_name = 'class_list.txt')
#yolo.Train_the_Yolo(epochs1 = 3,epochs2 = 1,batch_size1 = 1,batch_size2 = 2)      
yolo.Detect(test_folder_name = 'test_folder',model_name = 'logstrained_weights_final.h5',score=0.1)       

