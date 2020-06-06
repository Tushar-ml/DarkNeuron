from PIL import Image
import numpy as np
import cv2
from .yolo import YOLO
import os 


def Web_Cam_Detection(working_directory,videopath=0,model_name=None,classes = [],score = 0.6,iou = 0.5,gpu_num = 1):
    
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
    cap = cv2.VideoCapture(videopath)
    out = cv2.VideoWriter(os.path.join(working_directory,'output_web.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 5, (int(cap.get(3)), int(cap.get(4))))    
    
    while True:
        ret,frame = cap.read()
        
        img = Image.fromarray(frame)
        pred,image = yolo.detect_image(img,classes = classes,score=score)
        
        image = np.array(image)
        out.write(image)
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image',(600,600))
        cv2.imshow('image',image)
        
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
        