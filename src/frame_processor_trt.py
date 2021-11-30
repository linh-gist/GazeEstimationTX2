#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import cv2
import numpy as np
import pickle
import torch

from undistorter import Undistorter
from KalmanFilter1D import Kalman1D

from face_tensorrt import face
from landmarks import landmarks
from head import PnPHeadPoseEstimator
from normalization import normalize

class frame_processer:

    def __init__(self, cam_calib):

        self.cam_calib = cam_calib

        #######################################################
        #### prepare Kalman filters, R can change behaviour of Kalman filter
        #### play with it to get better smoothing, larger R - more smoothing and larger delay
        #######################################################
        self.kalman_filters = list()
        for point in range(2):
            # initialize kalman filters for different coordinates
            # will be used for face detection over a single object
            self.kalman_filters.append(Kalman1D(sz=100, R=0.01 ** 2))

        self.kalman_filters_landm = list()
        for point in range(68):
            # initialize Kalman filters for different coordinates
            # will be used to smooth landmarks over the face for a single face tracking
            self.kalman_filters_landm.append(Kalman1D(sz=100, R=0.005 ** 2))

        # initialize Kalman filter for the on-screen gaze point-of regard
        self.kalman_filter_gaze = list()
        self.kalman_filter_gaze.append(Kalman1D(sz=100, R=0.01 ** 2))

        self.undistorter = Undistorter(self.cam_calib['mtx'], self.cam_calib['dist'])
        self.landmarks_detector = landmarks()
        self.head_pose_estimator = PnPHeadPoseEstimator()
        self.face = face()

    def process(self, queue, read_frames, mon, device, gaze_network):

        while True:
            elements = list()
            while queue.qsize():
                elements.append(queue.get())
            if len(elements) > 0:
                img = elements[-1] # Get the latest frame, discard all past frames
            else:
               continue

            img = self.undistorter.apply(img)
            # detect face
            face_location = self.face.detect(img,  scale=0.25, use_max='SIZE')

            if len(face_location) > 0:
                # use kalman filter to smooth bounding box position
                # assume work with complex numbers:
                output_tracked = self.kalman_filters[0].update(face_location[0] + 1j * face_location[1])
                face_location[0], face_location[1] = np.real(output_tracked), np.imag(output_tracked)
                output_tracked = self.kalman_filters[1].update(face_location[2] + 1j * face_location[3])
                face_location[2], face_location[3] = np.real(output_tracked), np.imag(output_tracked)

                # detect facial points
                pts = self.landmarks_detector.detect_dlib(face_location, img)
                # run Kalman filter on landmarks to smooth them
                for i in range(68):
                    kalman_filters_landm_complex = self.kalman_filters_landm[i].update(pts[i, 0] + 1j * pts[i, 1])
                    pts[i, 0], pts[i, 1] = np.real(kalman_filters_landm_complex), np.imag(kalman_filters_landm_complex)

                # compute head pose
                fx, _, cx, _, fy, cy, _, _, _ = self.cam_calib['mtx'].flatten()
                camera_parameters = np.asarray([fx, fy, cx, cy])
                rvec, tvec = self.head_pose_estimator.fit_func(pts, camera_parameters)

                ######### GAZE PART #########

                # create normalized eye patch and gaze and head pose value,
                # if the ground truth point of regard is given
                head_pose = (rvec, tvec)
                por = None
                entry = {
                        'full_frame': img,
                        '3d_gaze_target': por,
                        'camera_parameters': camera_parameters,
                        'full_frame_size': (img.shape[0], img.shape[1]),
                        'face_bounding_box': (int(face_location[0]), int(face_location[1]),
                                              int(face_location[2] - face_location[0]),
                                              int(face_location[3] - face_location[1]))
                        }
                [patch, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = normalize(entry, head_pose)
                # cv2.imshow('raw patch', patch)

                def preprocess_image(image):
                    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
                    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
                    # cv2.imshow('processed patch', image)

                    image = np.transpose(image, [2, 0, 1])  # CxHxW
                    image = 2.0 * image / 255.0 - 1
                    return image

                # estimate the PoR using the gaze network
                processed_patch = preprocess_image(patch)
                processed_patch = processed_patch[np.newaxis, :, :, :]

                # Functions to calculate relative rotation matrices for gaze dir. and head pose
                def R_x(theta):
                    sin_ = np.sin(theta)
                    cos_ = np.cos(theta)
                    return np.array([
                        [1., 0., 0.],
                        [0., cos_, -sin_],
                        [0., sin_, cos_]
                    ]).astype(np.float32)

                def R_y(phi):
                    sin_ = np.sin(phi)
                    cos_ = np.cos(phi)
                    return np.array([
                        [cos_, 0., sin_],
                        [0., 1., 0.],
                        [-sin_, 0., cos_]
                    ]).astype(np.float32)

                def calculate_rotation_matrix(e):
                    return np.matmul(R_y(e[1]), R_x(e[0]))

                R_head_a = calculate_rotation_matrix(h_n)
                R_gaze_a = np.zeros((1, 3, 3))

                input_dict = {
                    'image_a': processed_patch,
                    'R_gaze_a': R_gaze_a,
                    'R_head_a': R_head_a,
                }

                # compute eye gaze and point of regard
                for k, v in input_dict.items():
                    input_dict[k] = torch.FloatTensor(v).to(device).detach()

                gaze_network.eval()
                output_dict = gaze_network(input_dict)
                output = output_dict['gaze_a_hat']
                g_cnn = output.data.cpu().numpy()
                g_cnn = g_cnn.reshape(3, 1)
                g_cnn /= np.linalg.norm(g_cnn)

                # compute the POR on z=0 plane
                g_n_forward = -g_cnn
                g_cam_forward = inverse_M * g_n_forward
                g_cam_forward = g_cam_forward / np.linalg.norm(g_cam_forward)

                d = -gaze_cam_origin[2] / g_cam_forward[2]
                por_cam_x = gaze_cam_origin[0] + d * g_cam_forward[0]
                por_cam_y = gaze_cam_origin[1] + d * g_cam_forward[1]
                por_cam_z = 0.0

                x_pixel_hat, y_pixel_hat = por_cam_x, por_cam_y

                output_tracked = self.kalman_filter_gaze[0].update(x_pixel_hat + 1j * y_pixel_hat)
                x_pixel_hat, y_pixel_hat = np.ceil(np.real(output_tracked)), np.ceil(np.imag(output_tracked))

                # show point of regard on screen
                por_img = mon.visual_por_xy(x_pixel_hat, y_pixel_hat)
                # also show the face:
                cv2.rectangle(img, (int(face_location[0]), int(face_location[1])),
                              (int(face_location[2]), int(face_location[3])), (255, 0, 0), 2)
                self.landmarks_detector.plot_markers_neat(img, pts)
                self.head_pose_estimator.drawPose(img, rvec, tvec, self.cam_calib['mtx'], np.zeros((1, 4)))
                # show eyes region
                h, w, c = patch.shape
                hi, wi, _ = img.shape
                img[hi - h:hi, int(wi / 2 - w / 2):int(wi / 2 + w / 2), :] = 1.0 * patch
                # combine, por & image & patch in one image
                h, w, c = por_img.shape
                rez_img = cv2.resize(img, (int(wi * h / hi), h))
                visl = np.ones((h, w + rez_img.shape[1], 3), np.uint8)
                visl[:h, :rez_img.shape[1], :] = rez_img
                visl[:h, rez_img.shape[1]:visl.shape[1], :] = por_img
                cv2.imshow('Visualization', visl)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        # END while loop
        cv2.destroyAllWindows()
        read_frames.value = False # Stop pipeline of camera process

