"""

This File will contain the implementation of YOLOV3 Algorithm,

containing Image_Annotation, Training and Prediction

Author: Tushar Goel

"""

from Yolo_Format import Image_Annotation
from Download_Convert_Yolo_Weights import Download_weights,Convert_weights

class YOLOv3:
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
            
    def Train_the_Yolo(self,yolo_file_name=None,plot_model=False,save_weights=False):
        
        self.yolo_file_name = yolo_file_name
        #Checking whether User have Yolo File or Not 
        #If no File, then it will be downloaded Automatically and Converted to Keras Model
        if yolo_file_name is None:
            Download_weights(self.working_directory)
            Convert_weights(self.working_directory,plot_model=plot_model,save_weights=save_weights)
            
        
    
    
working_directory = r'C:\Users\Tushar Goel\Desktop\OpenLabeling-master\images'
yolo = YOLOv3(working_directory=working_directory)
yolo.Prepare_the_Data(file_type='text',class_file_name='class_list.txt')
yolo.Train_the_Yolo()            
        
        
