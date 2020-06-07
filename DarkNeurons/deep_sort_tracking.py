from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
import os
from .deepsort_preprocessing import *
from .nn_matching import *
from .deepsort_detection import Detection
from .tracker import Tracker
from  .generate_detections import *
import imutils.video


warnings.filterwarnings('ignore')

def DeepSort_Tracking(yolo,working_directory,file_path = 0):

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = os.path.join(os.path.dirname(__file__),'mars-small128.pb')
    print('Deep Sort Model Generated')
    encoder = create_box_encoder(model_filename,batch_size=1)
    
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    

    file_path = file_path
    video_capture = cv2.VideoCapture(file_path)



    if writeVideo_flag:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(working_directory,'output_Tracking.avi'), fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
             break

        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)[0]
        confidence = yolo.detect_image(image)[1]

        features = encoder(frame,boxs)

        detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(boxs, confidence, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2)
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0,255,0),2)
            
        cv2.imshow('', frame)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        fps = (fps + (1./(time.time()-t1))) / 2
        print("FPS = %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))
    video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()

