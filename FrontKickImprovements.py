import mediapipe as mp
import cv2
import numpy
import PoseModulue
import os


class FrontKickImprovements:
    def __init__(self):
        pass

    def find_points(self, is_leading_right):
        closest_chamber_slope = None
        closest_chamber_foot_slope = None

        for idx, results in enumerate(self.lm_list):
            results = results['results']
            if is_leading_right:
                knee_leading_x, knee_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                    self.results,
                                                                                    self.mp_pose.PoseLandmark.RIGHT_KNEE)
                ankle_leading_x, ankle_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                      self.results,
                                                                                      self.mp_pose.PoseLandmark.RIGHT_ANKLE)
                heel_leading_x, heel_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                    self.results,
                                                                                    self.mp_pose.PoseLandmark.RIGHT_HEEL)
                hip_leading_x, hip_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                  self.results,
                                                                                  self.mp_pose.PoseLandmark.RIGHT_HIP)
                index_leading_x, index_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                      self.results,
                                                                                      self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
            else:
                knee_leading_x, knee_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                    self.results,
                                                                                    self.mp_pose.PoseLandmark.LEFT_KNEE)
                ankle_leading_x, ankle_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                      self.results,
                                                                                      self.mp_pose.PoseLandmark.LEFT_ANKLE)
                heel_leading_x, heel_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                    self.results,
                                                                                    self.mp_pose.PoseLandmark.LEFT_HEEL)
                hip_leading_x, hip_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                  self.results,
                                                                                  self.mp_pose.PoseLandmark.LEFT_HIP)
                index_leading_x, index_leading_y = PoseModulue.PoseDetector.findPoint(self.mp_pose, self.frame,
                                                                                      self.results,
                                                                                      self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX)

            shin_slope = self.slope(knee_leading_x, knee_leading_y, ankle_leading_x, ankle_leading_y)
            upper_leg_slope = self.slope(hip_leading_x, hip_leading_y, knee_leading_x, knee_leading_y)
            foot_slope = self.slope(heel_leading_x, heel_leading_y, index_leading_x, index_leading_y)

            if abs(shin_slope) < 50:
                print("Keep your shin down on the chamber")

            if closest_chamber_slope is not None:
                if abs(shin_slope) > abs(closest_chamber_slope['chamber_slope']):
                    closest_chamber_slope = {'id': idx, 'chamber_slope': shin_slope, 'knee_x': knee_leading_x,
                                             'knee_y': knee_leading_y,
                                             'ankle_x': ankle_leading_x, 'ankle_y': ankle_leading_y}
            else:
                closest_chamber_slope = {'id': idx, 'chamber_slope': shin_slope, 'knee_x': knee_leading_x,
                                         'knee_y': knee_leading_y,
                                         'ankle_x': ankle_leading_x, 'ankle_y': ankle_leading_y}


            #if abs(foot_slope) < 15:
            #    print("print your foot down")

            if closest_chamber_foot_slope is not None:
                if abs(shin_slope) > abs(closest_chamber_slope['foot_slope']):
                    closest_chamber_foot_slope = {'id': idx, 'foot_slope': foot_slope, 'heel_x': heel_leading_x,
                                          'heel_y': heel_leading_y,
                                          'index_x': index_leading_x, 'index_y': index_leading_y}
            else:
                closest_chamber_foot_slope = {'id': idx, 'foot_slope': foot_slope, 'heel_x': heel_leading_x,
                                         'heel_y': heel_leading_y,
                                         'index_x': index_leading_x, 'index_y': index_leading_y}
        pass

        annotated_closest_chamber = closest_chamber_slope['frame'].copy()
        cv2.circle(annotated_closest_chamber, (closest_chamber_slope['ankle_x'], closest_chamber_slope['ankle_y']), 10,
                   (255, 0, 0), 5)
        cv2.circle(annotated_closest_chamber, (closest_chamber_slope['knee_x'], closest_chamber_slope['knee_y']), 10,
                   (255, 0, 0), 5)
        cv2.imwrite("Annotations/ANNOTATED-Chamber.png", annotated_closest_chamber)

    def improve_extension(self, is_leading_right):
        if is_leading_right:
            pass
        else:
            pass

        pass

    def slope(self, x1, x2, y1, y2):
        return (y2 - y1) / (x2 - x1)
