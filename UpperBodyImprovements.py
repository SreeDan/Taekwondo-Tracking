import mediapipe as mp
import cv2
import numpy
import PoseModulue
import os


class Improvements:
    def __init__(self, frame_side, frame_front, annotated_side, annotated_front, files, results_side, results_front,
                 mp_pose, lm_list):
        self.frame_side = frame_side
        self.frame_front = frame_front
        self.annotated_side = annotated_side
        self.annotated_front = annotated_front
        self.files = files
        self.results_side = results_side
        self.results_front = results_front
        self.mp_pose = mp_pose
        self.lm_list = lm_list

    def improve_punch(self, is_leading_right):
        if is_leading_right:
            hand_leading_x, hand_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                self.results_side,
                                                                                self.mp_pose.PoseLandmark.RIGHT_WRIST)
            shoulder_leading_x, shoulder_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                        self.results_side,
                                                                                        self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
            hip_leading_x, hip_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                              self.results_side,
                                                                              self.mp_pose.PoseLandmark.RIGHT_HIP)
        else:
            hand_leading_x, hand_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                self.results_side,
                                                                                self.mp_pose.PoseLandmark.LEFT_WRIST)
            shoulder_leading_x, shoulder_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                        self.results_side,
                                                                                        self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            hip_leading_x, hip_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                              self.results_side,
                                                                              self.mp_pose.PoseLandmark.LEFT_HIP)

        midpoint_body_x, midpoint_body_y = self.midpoint(shoulder_leading_x, shoulder_leading_y, hip_leading_x,
                                                         hip_leading_y)
        midpoint_chest_x, midpoint_chest_y = self.midpoint(shoulder_leading_x, shoulder_leading_y, midpoint_body_x,
                                                           midpoint_body_y)
        slope = self.slope(midpoint_chest_x, midpoint_chest_y, hand_leading_x, hand_leading_y)

        print(slope)

    def improve_block(self, is_leading_right):
        if is_leading_right:
            index_leading_x, index_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                              self.results_side,
                                                                              self.mp_pose.PoseLandmark.RIGHT_INDEX)
            mouth_x, mouth_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.MOUTH_RIGHT)
        else:
            index_leading_x, index_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.LEFT_INDEX)
            mouth_x, mouth_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                  self.results_side,
                                                                  self.mp_pose.PoseLandmark.MOUTH_LEFT)

        slope = self.slope(index_leading_x, index_leading_y, mouth_x, mouth_y)
        print(slope)

    def midpoint(self, x1, y1, x2, y2):
        return (x1 + x2) / 2, (y1 + y2) / 2

    def slope(self, x1, y1, x2, y2):
        return (y2 - y1) / (x2 - x1)
