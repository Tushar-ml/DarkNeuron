"""
This Script is used for Detection purposes whether it's video or images

and also be used for Real-Time Detection

Author: Tushar Goel

"""

import os
import sys
from .yolo import YOLO,detect_video
from PIL import Image
from timeit import default_timer as timer
from .utils import load_extractor_model, load_features, parse_input, detect_object
import pandas as pd
import numpy as np
from .Yolo_Format import GetFileList
import random


def Detector(working_directory,output_directory,test_folder_name,classes = [],model_name=None,score=0.5,gpu_num = 1):
    """
    This Function will be used for Detection of Objects in Video Or Images
    
    Arguments:
        working_directory --> Working Directory where weights and Test Images are Kept.
        Test_Folder_name -->  Name of the Test Folder name.
        model_name --> Name of the Model(None--> using yolov4.h5)
        
    Return:
        Detections
        
    """
    image_test_folder = os.path.join(working_directory,test_folder_name)

    if model_name == 'yolov4.h5':
        model_weights = os.path.join(output_directory,'yolov4.h5')
        model_classes = os.path.join(os.path.dirname(__file__),'coco_classes.txt')
    else:
        model_weights = os.path.join(output_directory,model_name)
        model_classes = os.path.join(working_directory,'data_classes.txt')
    
    anchors_path = os.path.join(os.path.dirname(__file__),'yolo4_anchors.txt')
    detection_results_file = os.path.join(output_directory,'Detections_results.csv')
    postfix = 'Detection'
    save_img = True
    
    input_paths = GetFileList(dirName = image_test_folder)
    img_endings = (".jpg", ".jpeg", ".png",'.bmp')
    vid_endings = (".mp4", ".mpeg", ".mpg", ".avi",".mkv")

    input_image_paths = []
    input_video_paths = []
    for item in input_paths:
        if item.endswith(img_endings):
            input_image_paths.append(item)
        elif item.endswith(vid_endings):
            input_video_paths.append(item)

    output_path = os.path.join(working_directory,'Test_results')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # define YOLO detector
    yolo = YOLO(
        **{
            "model_path": model_weights,
            "anchors_path": anchors_path,
            "classes_path": model_classes,
            "score": score,
            "gpu_num": gpu_num,
            "model_image_size": (608,608),
        }
    )
    
    # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",
        ]
    )
    
    # labels to draw on images
    class_file = open(model_classes, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

    if input_image_paths:
        print(
            "Found {} input images: {} ...".format(
                len(input_image_paths),
                [os.path.basename(f) for f in input_image_paths[:5]],
            )
        )
        start = timer()
        text_out = ""
        
        # This is for images
        for i, img_path in enumerate(input_image_paths):
            print(img_path)
            prediction, image = detect_object(
                yolo,
                img_path,
                save_img=save_img,
                save_img_path=output_path,
                postfix=postfix,
            )
            y_size, x_size, _ = np.array(image).shape
            for single_prediction in prediction:
                out_df = out_df.append(
                    pd.DataFrame(
                        [
                            [
                                os.path.basename(img_path.rstrip("\n")),
                                img_path.rstrip("\n"),
                            ]
                            + single_prediction
                            + [x_size, y_size]
                        ],
                        columns=[
                            "image",
                            "image_path",
                            "xmin",
                            "ymin",
                            "xmax",
                            "ymax",
                            "label",
                            "confidence",
                            "x_size",
                            "y_size",
                        ],
                    )
                )
        end = timer()
        print(
            "Processed {} images in {:.1f}sec - {:.1f}FPS".format(
                len(input_image_paths),
                end - start,
                len(input_image_paths) / (end - start),
            )
        )
        out_df.to_csv(detection_results_file, index=False)
        
    # This is for videos
    if input_video_paths:
        print(
            "Found {} input videos: {} ...".format(
                len(input_video_paths),
                [os.path.basename(f) for f in input_video_paths[:5]],
            )
        )
        start = timer()
        for i, vid_path in enumerate(input_video_paths):
            output_path = os.path.join(
                output_path,
                os.path.basename(vid_path).replace(".", postfix + "."),
            )
            detect_video(yolo, vid_path, output_path=output_path)

        end = timer()
        print(
            "Processed {} videos in {:.1f}sec".format(
                len(input_video_paths), end - start
            )
        )
    # Close the current yolo session
    yolo.close_session()


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        