import math

import cv2
import mediapipe as mp
import os
import numpy as np
import PoseModulue

path = os.path.abspath("Annotations")

class Improvements:
    def __init__(self, frame_side, frame_front, annotated_side, annotated_front, files, results_side, results_front,
                 mp_pose):
        self.frame_side = frame_side
        self.frame_front = frame_front
        self.annotated_side = annotated_side
        self.annotated_front = annotated_front
        self.files = files
        self.results_side = results_side
        self.results_front = results_front
        self.mp_pose = mp_pose

    def improve_front_stance_side(self, is_leading_right):

        file_side = self.files['side']

        if is_leading_right:  # Leading = foot in front

            knee_leading_x, knee_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                self.results_side,
                                                                                self.mp_pose.PoseLandmark.RIGHT_KNEE)
            index_leading_x, index_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
            heel_leading_x, heel_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                self.results_side,
                                                                                self.mp_pose.PoseLandmark.RIGHT_HEEL)
            ankle_following_x, ankle_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                      self.results_side,
                                                                                      self.mp_pose.PoseLandmark.LEFT_ANKLE)
            knee_following_x, knee_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                    self.results_side,
                                                                                    self.mp_pose.PoseLandmark.LEFT_KNEE)
            hip_following_x, hip_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.LEFT_HIP)

        else:

            knee_leading_x, knee_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                self.results_side,
                                                                                self.mp_pose.PoseLandmark.LEFT_KNEE)
            index_leading_x, index_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
            heel_leading_x, heel_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                self.results_side,
                                                                                self.mp_pose.PoseLandmark.LEFT_HEEL)
            ankle_following_x, ankle_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                      self.results_side,
                                                                                      self.mp_pose.PoseLandmark.RIGHT_ANKLE)
            knee_following_x, knee_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                    self.results_side,
                                                                                    self.mp_pose.PoseLandmark.RIGHT_KNEE)
            hip_following_x, hip_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.RIGHT_HIP)

        midpoint_x = int((heel_leading_x + index_leading_x) / 2)
        midpoint_x = int((midpoint_x + index_leading_x) / 2)
        midpoint_y = int((heel_leading_y + index_leading_y) / 2)
        midpoint_y = int((midpoint_y + index_leading_y) / 2)

        slope_leading = (midpoint_y - knee_leading_y) / (midpoint_x - knee_leading_x)
        if slope_leading > 1.1 or slope_leading < 0.9:
            print("Place your knee above your toes")
            cv2.line(self.annotated_side, (midpoint_x, midpoint_y), (midpoint_x, knee_leading_y), (0, 255, 0), 5)
            cv2.imwrite(os.path.join(path, "ANNOTATED-SIDE-" + file_side), self.annotated_side)

        slope_following_lower = (knee_following_y - ankle_following_y) / (knee_following_x - ankle_following_x)
        slope_following_upper = (hip_following_y - knee_following_y) / (hip_following_x - knee_following_x)
        difference = abs(slope_following_upper - slope_following_lower)

        if difference > 2.5:
            print("fix the slope")

    def improve_front_stance_front(self, is_leading_right):
        if is_leading_right:
            heel_following_x, heel_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_front,
                                                                                    self.results_front,
                                                                                    self.mp_pose.PoseLandmark.LEFT_HEEL)
            index_following_x, index_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_front,
                                                                                      self.results_front,
                                                                                      self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
            index_leading_x, index_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_front,
                                                                                  self.results_front,
                                                                                  self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
        else:
            heel_following_x, heel_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_front,
                                                                                    self.results_front,
                                                                                    self.mp_pose.PoseLandmark.RIGHT_HEEL)
            index_following_x, index_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_front,
                                                                                      self.results_front,
                                                                                      self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
            index_leading_x, index_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_front,
                                                                                  self.results_front,
                                                                                  self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX)

        annotated_image = self.annotated_front.copy()
        cv2.circle(annotated_image, (index_following_x, index_following_y), 10, (0, 255, 0), 2)
        cv2.circle(annotated_image, (heel_following_x, heel_following_y), 10, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(path, "ANNOTATED-FRONT-" + self.files['front']), annotated_image)

        self.find_angle([index_following_x, -index_following_y], [heel_following_x, -heel_following_y],
                        [heel_following_x, -index_following_y])

    def improve_walking_stance(self, is_leading_right):
        if is_leading_right:
            ankle_leading_x, ankle_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.RIGHT_ANKLE)
            knee_leading_x, knee_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                self.results_side,
                                                                                self.mp_pose.PoseLandmark.RIGHT_KNEE)
            hip_leading_x, hip_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                              self.results_side,
                                                                              self.mp_pose.PoseLandmark.RIGHT_HIP)
            shoulder_following_x, shoulder_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose,
                                                                                            self.frame_side,
                                                                                            self.results_side,
                                                                                            self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            hip_following_x, hip_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.LEFT_HIP)
            knee_following_x, knee_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                    self.results_side,
                                                                                    self.mp_pose.PoseLandmark.LEFT_KNEE)
            ankle_following_x, ankle_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                      self.results_side,
                                                                                      self.mp_pose.PoseLandmark.LEFT_ANKLE,
                                                                                      show=True)


        else:
            ankle_leading_x, ankle_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.LEFT_ANKLE,
                                                                                  show=True)
            knee_leading_x, knee_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                self.results_side,
                                                                                self.mp_pose.PoseLandmark.LEFT_KNEE)
            hip_leading_x, hip_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                              self.results_side,
                                                                              self.mp_pose.PoseLandmark.LEFT_HIP)
            shoulder_following_x, shoulder_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose,
                                                                                            self.frame_side,
                                                                                            self.results_side,
                                                                                            self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
            hip_following_x, hip_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                  self.results_side,
                                                                                  self.mp_pose.PoseLandmark.RIGHT_HIP)

            knee_following_x, knee_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                    self.results_side,
                                                                                    self.mp_pose.PoseLandmark.RIGHT_KNEE)

            ankle_following_x, ankle_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side,
                                                                                      self.results_side,
                                                                                      self.mp_pose.PoseLandmark.RIGHT_ANKLE)

        slope_leading_lower = self.slope(ankle_leading_x, ankle_leading_y, knee_leading_x, knee_leading_y)
        slope_leading_upper = self.slope(knee_leading_x, knee_leading_y, hip_leading_x, hip_leading_y)

        difference = abs(slope_leading_upper - slope_leading_lower)

        if difference > 7.5:
            print("straighten your front leg")

        print("hip following" + str(shoulder_following_x) + ", " + str(shoulder_following_y))

        slope_following_lower = self.slope(ankle_following_x, ankle_following_y, knee_following_x, knee_following_y)
        slope_following_mid = self.slope(knee_following_x, knee_following_y, hip_following_x, hip_following_y)
        slope_following_upper = self.slope(hip_following_x, hip_following_y, shoulder_following_x, shoulder_following_y)

        difference_lower_mid = abs(slope_following_mid - slope_following_lower)

        if abs(slope_following_upper) < 12:
            print("Keep your back up")
        if difference_lower_mid > 1.5:
            print("Keep your back leg straight")

        annotated_image = self.annotated_side.copy()
        cv2.line(annotated_image, (int(shoulder_following_x), int(shoulder_following_y)),
                 (int(hip_following_x), int(hip_following_y)), (255, 0, 0), 10)

        cv2.circle(annotated_image, (int(knee_leading_x), int(knee_leading_y)), 10, (255, 0, 0), 2)
        cv2.circle(annotated_image, (int(347), int(504)), 10, (255, 0, 0), 2)
        cv2.line(annotated_image, (int(ankle_following_x), int(ankle_following_y)),
                 (int(hip_following_x), int(hip_following_y)), (255, 0, 0), 10)

        cv2.imwrite(os.path.join(path, "ANNOTATED-SIDE-" + self.files['side']), annotated_image)

    def improve_back_stance(self, is_leading_right):
        if is_leading_right:
            knee_following_x, knee_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side, self.results_side, self.mp_pose.PoseLandmark.LEFT_KNEE)
            index_following_x, index_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side, self.results_side, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
            hip_following_x, hip_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side, self.results_side, self.mp_pose.PoseLandmark.LEFT_HIP)
            shoulder_following_x, shoulder_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side, self.results_side, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            pass
        else:
            knee_following_x, knee_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side, self.results_side, self.mp_pose.PoseLandmark.RIGHT_KNEE)
            index_following_x, index_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side, self.results_side, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
            hip_following_x, hip_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side, self.results_side, self.mp_pose.PoseLandmark.RIGHT_HIP)
            shoulder_following_x, shoulder_following_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame_side, self.results_side, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
            pass

        slope_following_lower = abs(self.slope(knee_following_x, knee_following_y, index_following_x, index_following_y))
        slope_following_body = abs(self.slope(shoulder_following_x, shoulder_following_y, hip_following_x, hip_following_y))
        slope_following_leg = abs(self.slope(hip_following_x, hip_following_y, index_following_x, index_following_y))

        if slope_following_lower > 2.5:
            print("Keep your knee above your foot")

        if slope_following_body > 2.5:
            print("Keep your back straight")

        if slope_following_leg > 2.5:
            print("Try to sit back and keep your back foot in line with your body")


    def slope(self, x1, y1, x2, y2):
        return (y2 - y1) / (x2 - x1)

    def find_angle(self, p1, p2, p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        print(p1, p2, p3)

        #  Try to find out the length and use law of sin/cos

        P2 = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
        P3 = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        my_angle = np.arcsin(P2 / P3)
        print(my_angle * 180 / np.pi)

        radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        angle = np.abs(radians * 180.0 / np.pi)
        print(angle)
        if angle > 180.0:
            angle = 360 - angle

        print(angle)
