from PIL import ImageGrab,Image
import numpy as np
import cv2
from .yolo import YOLO
import os 
import pyautogui

width,height = pyautogui.size()[0],pyautogui.size()[1]

def Real_Time_Tracking(working_directory,classes = [],model_name=None,score = 0.6,iou = 0.5,gpu_num = 1):
    
    anchors_path = os.path.join(os.path.dirname(__file__),'model_data/yolo4_anchors.txt')
    if model_name is None:
        model_path = os.path.join(working_directory,'yolov4.h5')
        classes_path = os.path.join(os.path.dirname(__file__),'model_data/coco_classes.txt')
    else:
        model_path = os.path.join(working_directory,model_name)
        classes_path = os.path.join(working_directory,'data_classes.txt')
    
    yolo = YOLO(
        **{
            "model_path": model_path,
            "anchors_path": anchors_path,
            "classes_path": classes_path,
            "score": score,
            "gpu_num": gpu_num,
            "model_image_size": (608,608),
        }
    )
    out = cv2.VideoWriter(os.path.join(working_directory,'output.avi'),cv2.VideoWriter_fourcc('X','V','I','D'), 10, (width,height))
    while True:
        # make a screenshot
        img = pyautogui.screenshot()
        # convert these pixels to a proper numpy array to work with OpenCV
        prediction,img = yolo.detect_image(img,classes = classes,score = score,show_stats=False)
        frame = np.array(img)
        # convert colors from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # write the frame
        out.write(frame)
        # show the frame
        cv2.imshow("screenshot", frame)
        # if the user clicks q, it exits
        if cv2.waitKey(1) == ord("q"):
            break
    out.release()
    cv2.destroyAllWindows()