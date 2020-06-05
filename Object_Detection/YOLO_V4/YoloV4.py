"""

This File will contain the implementation of YOLOV3 Algorithm,

containing Image_Annotation, Training and Prediction

Author: Tushar Goel

"""
import os
from .realtime_detect import Real_Time_Tracking
from .Yolo_Format import Image_Annotation
from .Download_Convert_Yolo_Weights import Yolo4_weights,Download_weights
from .Model_Training import Train_Yolo
from .Detection import Detector
from .web_cam_detect import Web_Cam_Detection
from .deep_sort_tracking import DeepSort_Tracking
from .yolo_tracking import YOLO_Tracker
import matplotlib.pyplot as plt

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
        # Method Involves 'csv','xml','text'
        Image_Annot = Image_Annotation(working_directory = self.working_directory,dataframe_name=dataframe_name)
        if self.file_type not in ['csv','xml','text']:
            raise ValueError('Method either should be dataframe,xml,text')
            
        if self.file_type == 'csv':
            print("Generating Data......")
            Image_Annot.Convert_to_Yolo_Format()
            
        elif self.file_type == 'xml':
            print("Generating Data......")
            print('Generating txt files from xml')
            Image_Annot.csv_from_xml(class_list_file_name = class_file_name)
            print('Generated')
            df = Image_Annot.csv_from_text(class_list_file_name = class_file_name)
           
            Image_Annot.Convert_to_Yolo_Format(df=df)

        elif self.file_type == 'text':
            if class_file_name is None:
                raise ValueError('Provide Class File Name for method text')
            print("Generating Data......")
            df = Image_Annot.csv_from_text(class_list_file_name = class_file_name)
            Image_Annot.Convert_to_Yolo_Format(df=df)
            
    def Train_the_Yolo(self,model_name = 'yolov4.h5',input_shape = (608,608),plot_model=False,save_weights=False,score = 0.5,iou = 0.5,
                       epochs1 = 51,epochs2 = 50, batch_size1 = 32,batch_size2 = 4,gpu_num = 1,
                       validation_split = 0.1,process1 = True,process2 = True):
        
        yolo_file_path = os.path.join(self.working_directory,'yolov4.h5')
        self.class_path = os.path.join(self.working_directory,'class_list.txt')
        self.anchors_path = os.path.join(os.path.dirname(__file__),'model_data/yolo4_anchors.txt')
        self.weight_path = os.path.join(self.working_directory,'yolov4.weights')
        self.coco_class = os.path.join(os.path.dirname(__file__),'model_data/coco_classes.txt')
        #Checking whether User have Yolo File or Not 
        #If no File, then it will be downloaded Automatically and Converted to Keras Model
        if not os.path.exists(yolo_file_path):
            
            Download_weights(working_directory = self.working_directory)
            yolov4 = Yolo4_weights(score=score,iou=iou,anchors_path = self.anchors_path,classes_path = self.coco_class,
                                   model_path = yolo_file_path,weights_path = self.weight_path,gpu_num = gpu_num)
            yolov4.load_yolo()

        
        print('Model Training to be Start ....')
        history = Train_Yolo(working_directory = self.working_directory,model_name = model_name,val_split = validation_split,
                   epochs1=epochs1,epochs2 = epochs2,batch_size1 = batch_size1,
                   batch_size2 = batch_size2,process1 = process1, process2 = process2)
        
        
        self.history = history
        
        plt.plot(self.history.history['loss'],label = 'Training Loss')
        plt.plot(self.history.history['val_loss'],label = 'Validation Loss')
        plt.title('Training Loss vs Validation Loss')
        plt.legends()
        plt.show()
        
    def Detect(self,test_folder_name='test',model_name = None,cam = False,real_time = False,videopath = 0,classes = [],tracking =False,score=0.5,gpu_num = 1):
        
        if model_name is None:
            model_path = os.path.join(self.working_directory,'yolov4.h5')
            print('Yolo File Not Found...Downloading and Converting...')
            if not os.path.exists(model_path):
                Download_weights(working_directory = self.working_directory)
                yolov4 = Yolo4_weights(score=score,anchors_path = self.anchors_path,classes_path = self.coco_class,
                                   model_path = model_path,weights_path = self.weight_path,gpu_num = gpu_num)
                yolov4.load_yolo()
        
        yolo_tracker = YOLO_Tracker(working_directory = self.working_directory,model_name=model_name,
                                    score = score,classes = classes)
        if real_time:
            Real_Time_Tracking(working_directory = self.working_directory,classes = classes,score = score)
        
        elif cam:

            Web_Cam_Detection(working_directory = self.working_directory,videopath = videopath,model_name=model_name,score=score,
                              gpu_num = gpu_num,classes = classes)
            
        elif tracking :
                
            DeepSort_Tracking(working_directory = self.working_directory,yolo = yolo_tracker,file_path = videopath)
        
        else:
            Detector(working_directory = self.working_directory,test_folder_name = test_folder_name,
                     classes = classes,score = score,model_name = model_name,gpu_num = gpu_num)
            
    
       

