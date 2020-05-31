"""
This Script will convert Images and Labels in the Yolo Format

Images and Labels Can be In csv Format, will be guided in Readme.md File

about Microsoft VOTT (Visual Object Tagging Tool) and How to export csv File 

from it. 

Author : Tushar Goel

"""
from PIL import Image
import os
import re
import pandas as pd
import numpy as np
from glob import glob

class Image_Annotation:
    """
    This Class will used for conversion of Image and labels into YOLO format
    
    Attributes:
        working_directory --> working_directory where files and Labels will be there.
        Dataframe Name --> DataFrame Name containing Label Files and Images
        
    Methods:
        Convert_to_Yolo_Format
        Convert_csv_to_Yolo
        csv_from_xml
        csv_from_text
        
    Additional Files Formed:
        data_train.txt --> Containing Images and Labels in YOLO Format
        data_classes.txt --> having Classes of Custom Object
        
    """
    
    def __init__(self,working_directory,dataframe_name=None):
        
        self.working_directory = working_directory
        
        #Initialing Variables Required in Image Annotation
        """
        self.data_train --> Containing File and labels in Yolo Format
        self.data_classes --> Containing Classes of the data
        
        """
        if dataframe_name is not None:
            self.dataframe_path = os.path.join(self.working_directory,dataframe_name)
        self.data_train = os.path.join(self.working_directory,'data_train.txt')
        self.data_classes = os.path.join(self.working_directory,'data_classes.txt')
        
    
    def Convert_to_Yolo_Format(self,df=None):
        
        if df is not None:
            multi_df = df
        else:
            multi_df = pd.read_csv(self.dataframe_path)
        labels = multi_df["label"].unique()
        labeldict = dict(zip(labels, range(len(labels))))
        multi_df.drop_duplicates(subset=None, keep="first", inplace=True)
        train_path = self.working_directory
        self.convert_csv_to_yolo(
            multi_df, labeldict, path=train_path, target_name=self.data_train
        )
    
        # Make classes file
        file = open(self.data_classes, "w")
    
        # Sort Dict by Values
        SortedLabelDict = sorted(labeldict.items(), key=lambda x: x[1])
        for elem in SortedLabelDict:
            file.write(elem[0] + "\n")
        file.close()
        
    def convert_csv_to_yolo(   self,
                                    vott_df,
                                    labeldict=dict(zip(["Yolo_Training"], [0,])),
                                    path="",
                                    target_name="data_train.txt",
                                    abs_path=False,
                                ):
        
    # Encode labels according to labeldict if code's don't exist
        if not "code" in vott_df.columns:
            vott_df["code"] = vott_df["label"].apply(lambda x: labeldict[x])
        # Round float to ints
        for col in vott_df[["xmin", "ymin", "xmax", "ymax"]]:
            vott_df[col] = (vott_df[col]).apply(lambda x: round(x))
    
        # Create Yolo Text file
        last_image = ""
        txt_file = ""
    
        for index, row in vott_df.iterrows():
            if not last_image == row["image"]:
                if abs_path:
                    txt_file += "\n" + row["image_path"] + " "
                else:
                    txt_file += "\n" + os.path.join(path, row["image"]) + " "
                txt_file += ",".join(
                    [
                        str(x)
                        for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                    ]
                )
            else:
                txt_file += " "
                txt_file += ",".join(
                    [
                        str(x)
                        for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                    ]
                )
            last_image = row["image"]
        file = open(target_name, "w")
        file.write(txt_file[1:])
        file.close()
        return True
    
    def csv_from_xml(self, path_name=""):
        # First get all images and xml files from path and its subfolders
        image_paths = self.GetFileList(self.working_directory, ".jpg")
        xml_paths = self.GetFileList(self.working_directory, ".xml")
        result_df = pd.DataFrame()
        if not len(image_paths) == len(xml_paths):
            print("number of annotations doesnt match number of images")
            return False
        for image in image_paths:
            target_filename = os.path.join(path_name, image) if path_name else image
            source_filename = os.path.join(self.working_directory, image)
            y_size, x_size, _ = np.array(Image.open(source_filename)).shape
            source_xml = image.replace(".jpg", ".xml")
            txt = open(source_xml, "r").read()
            y_vals = re.findall(r"(?:x>\n)(.*)(?:\n</)", txt)
            ymin_vals = y_vals[::2]
            ymax_vals = y_vals[1::2]
            x_vals = re.findall(r"(?:y>\n)(.*)(?:\n</)", txt)
            xmin_vals = x_vals[::2]
            xmax_vals = x_vals[1::2]
            label_vals = re.findall(r"(?:label>\n)(.*)(?:\n</)", txt)
            label_name_vals = re.findall(r"(?:labelname>\n)(.*)(?:\n</)", txt)
            df = pd.DataFrame()
            df["xmin"] = xmin_vals
            df["xmin"] = df["xmin"].astype(float) * x_size
            df["ymin"] = ymin_vals
            df["ymin"] = df["ymin"].astype(float) * y_size
            df["xmax"] = xmax_vals
            df["xmax"] = df["xmax"].astype(float) * x_size
            df["ymax"] = ymax_vals
            df["ymax"] = df["ymax"].astype(float) * y_size
            df["label"] = label_name_vals
            df["code"] = label_vals
            df["image_path"] = target_filename
            df["image"] = os.path.basename(target_filename)
            result_df = result_df.append(df)
        #     Bring image column first
        cols = list(df.columns)
        cols = [cols[-1]] + cols[:-1]
        result_df = result_df[cols]
        return result_df
    def csv_from_text(self,class_list_file_name):
        
        text_file_paths = glob(os.path.join(self.working_directory,'*.txt'))
        image_file_paths = self.GetFileList(self.working_directory,['.jpg','.jpeg','.png'])
        #Removing Class_text file from text path:
        
        class_file_text = os.path.join(self.working_directory,class_list_file_name)
        text_file_paths.remove(class_file_text) #This is not our Data 
        assert len(text_file_paths) == len(image_file_paths),"Length of image files and Their corressponding text files does not match"
   
        class_file = open(class_file_text,'r')
        classes = class_file.readlines()
        class_dic = dict()
        count = 0
        for cla in classes:
            cla = cla.split()[0]
            class_dic[count] = cla
            count += 1
        image_name = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        label = []
        df = pd.DataFrame()
        
        for text_file in text_file_paths:
            for i in image_file_paths:
                try:
                    if text_file.split('\\')[-1].split('.')[0] == i.split('\\')[-1].split('.')[0]:
                        image_path = i
                        break
                except:
                    if text_file.split('/')[-1].split('.')[0] == i.split('/')[-1].split('.')[0]:
                        image_path = i
                        break
            y_size, x_size, _ = np.array(Image.open(image_path)).shape
            file = open(text_file,'r')
            
            lines = file.readlines()
            for line in lines:
                line = line.split()
                image_name.append(image_path)
                xmin.append(float(line[1])*x_size)
                ymin.append(float(line[2])*y_size)
                xmax.append(float(line[3])*x_size)
                ymax.append(float(line[4])*y_size)
                
                label.append(class_dic[int(line[0])])
                
        df['image'] = image_name
        df['xmin'] = xmin
        df['ymin'] = ymin
        df['xmax'] = xmax
        df['ymax'] = ymax
        df['label'] = label

        return df
        
    def GetFileList(self,dirName, endings=[".jpg", ".jpeg", ".png", ".mp4"]):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Make sure all file endings start with a '.'
    
        for i, ending in enumerate(endings):
            if ending[0] != ".":
                endings[i] = "." + ending
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.GetFileList(fullPath, endings)
            else:
                for ending in endings:
                    if entry.endswith(ending):
                        allFiles.append(fullPath)
        return allFiles

