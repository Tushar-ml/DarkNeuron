# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:48:33 2020

@author: Tushar Goel
"""

import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
import cv2

def getImagesInDir(dirName, endings=[".jpg", ".jpeg", ".png", ".mp4"]):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Make sure all file endings start with a '.'
        endings_final = [0]*len(endings)
        for i, ending in enumerate(endings):
            if ending[0] != ".":
                endings_final[i] = "." + ending
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + getImagesInDir(fullPath, endings)
            else:
                for ending in endings:
                    if entry.endswith(ending):
                        allFiles.append(fullPath)
        return allFiles

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]

    xmin = box[0]*dw
    ymin = box[2]*dh
    xmax = box[1]*dw
    ymax = box[3]*dh
    
    return (xmin,ymin,xmax,ymax)

def convert_annotation(dir_path, output_path, image_path,classes):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]
    img = cv2.imread(image_path)
    
    in_file = open(dir_path + '\\' + basename_no_ext + '.xml')
    
    out_file = open(output_path +'\\'+ basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = img.shape[1]
    h = img.shape[0]
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        
        cls = obj.find('name').text.lower()
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        
        xmlbox = obj.find('bndbox')
        
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h),b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file.close()