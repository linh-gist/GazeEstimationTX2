#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import sys
import cv2

sys.path.append("ext/tensorrt_mtcnn/")
from mtcnn import TrtMtcnn

class face:

    def __init__(self):
        self.BBOX_COLOR = (0, 255, 0)  # green
        self.mtcnn = TrtMtcnn()

    def show_faces(self, img, boxes, landmarks):
        """Draw bounding boxes and face landmarks on image."""
        for bb, ll in zip(boxes, landmarks):
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), self.BBOX_COLOR, 2)
            #for j in range(5):
            #    cv2.circle(img, (int(ll[j]), int(ll[j + 5])), 2, self.BBOX_COLOR, 2)
        return img

    def detect(self, frame, scale = 1.0, use_max='SIZE'):

        # detect face
        frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        bounding_boxes, landmarks = self.mtcnn.detect(frame_rgb, minsize=40)

        dets = [x[:4] for x in bounding_boxes]
        scores = [x[4] for x in bounding_boxes]

        face_location = []
        if len(dets) > 0:
            max = 0
            max_id = -1
            for i, d in enumerate(dets):
                if use_max == 'SCORE':
                    property = scores[i]
                elif use_max == 'SIZE':
                    property = abs(dets[i][2] - dets[i][0]) * abs(dets[i][3] - dets[i][1])
                if max < property:
                    max = property
                    max_id = i
            if use_max == 'SCORE':
                if max > -0.5:
                    face_location = dets[max_id]
            else:
                face_location = dets[max_id]
            face_location = face_location * (1/scale)

        return face_location

