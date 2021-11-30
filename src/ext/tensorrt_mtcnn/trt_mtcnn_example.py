"""trt_mtcnn.py

This script demonstrates how to do real-time face detection with
Cython wrapped TensorRT optimized MTCNN engine.
"""

import time
import argparse

import cv2
from mtcnn import TrtMtcnn
BBOX_COLOR = (0, 255, 0)  # green


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img

def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        for j in range(5):
            cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, BBOX_COLOR, 2)
    return img

def main():
    img = cv2.imread('./doc/avengers.png')
    mtcnn = TrtMtcnn()
    start=time.time()
    dets, landmarks = mtcnn.detect(img, minsize=50)
    img = show_faces(img, dets, landmarks)
    img = show_fps(img, 1/(time.time()-start))
    cv2.imshow('trt', img)
    cv2.waitKey(0)
    


if __name__ == '__main__':
    main()
