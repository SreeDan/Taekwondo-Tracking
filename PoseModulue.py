import cv2
import mediapipe as mp
import math
import StanceImprovements
import os
from os import path

annotations_path = os.path.abspath("Annotations")

frame_list = []
pause_counter = 0
stopped = False
current_move = 1
temp = 0
stop = False
h, w, _ = None, None, None


# Segments the video into clips
def getInfo(frame, make_video=False):
    global pause_counter, stopped, current_move, frame_list, temp, stop, h, w, _
    # print("try")
    # try:
    if make_video:
        create_video(w, h)
        return

    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    print(results.pose_landmarks.landmark)
    if len(frame_list) > 0:
        last_frame = frame_list[-1]
        landmarks = get_results(mp_pose, last_frame, frame, results)
        # print(PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'],
        #                             mp_pose.PoseLandmark.RIGHT_ANKLE))
        # print(landmarks[0])
        if landmarks is None:
            return
        for lms in landmarks:
            difference_x, difference_y = int(lms[0][0] - lms[1][0]), int(lms[0][1] - lms[1][1])
            if stopped == False and (difference_x > 5 or difference_y > 5):
                pause_counter = 0
                frame_list.append(
                    {
                        'frame': frame,
                        'results': results
                    }
                )

                if temp > 10:
                    stopped = False
                    # print(frame_list)
                temp += 1
                # print(temp)
                stopped = False

            else:
                frame_list.append(
                    {
                        'frame': frame,
                        'results': results
                    }
                )
                pause_counter += 1
                print(77)
                if pause_counter >= 15:
                    h, w, _ = frame_list[0]['frame'].shape
                    if stop == False:
                        print(81)
                        create_video(w, h)
                        print(83)
                    stop = True
                    stopped = True


    else:
        frame_list.append(
            {
                'frame': frame,
                'results': results
            }
        )
    '''except Exception as e:
        print("e")
        pass'''


def create_video(w, h):
    global current_move, pause_counter
    out = cv2.VideoWriter('Clips/clip--' + str(current_move) + '.mp4', 0x7634706d, 40.0, (w, h))
    current_move += 1
    for i in range(len(frame_list)):
        out.write(frame_list[i]['frame'])
    out.release()
    frame_list.clear()
    pause_counter = 0
    print("clip " + str(current_move - 1) + " is " + str(path.exists('Clips/clip--' + str(current_move - 1) + '.mp4')))


class PoseDetector():
    def __init__(self, mode=True, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    # def findPose(self, frame_side, frame_front, draw=True):
    def findPose(self, frame_side, draw=True):
        image_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
        # image_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
        self.results_side = self.pose.process(image_side)
        # self.results_front = self.pose.process(image_front)

        if self.results_side.pose_landmarks:
            for id, lm in enumerate(self.results_side.pose_landmarks.landmark):
                # print(self.results_side.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])

                h, w, _ = frame_side.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(cx, cy)
                # print(id, lm)

        # if self.results.pose_landmarks:
        # if draw:
        # self.mp_draw.draw_landmarks(frame, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        # return frame_side, frame_front
        return frame_side

    '''def findPosition(self, frame, draw=True):

        h, w, _ = frame.shape
        landmarks = self.results.pose_landmarks.landmark
        cx, cy = (landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * w), (
                landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * h)

        print(str(cx) + ", " + str(w))
        print(str(cy) + ", " + str(h))
        annotated_image = frame.copy()
        cv2.circle(annotated_image, (int(cx), int(cy)), 25, (255, 0, 255), cv2.FILLED)
        cv2.imwrite('annotated_image' + '.png', annotated_image)

        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = frame.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

    def findPosition(self, frame, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                print(w)
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])'''

    @staticmethod
    def findPoint(mp_pose, frame, results, point_on_body, stance=False, show=False):
        h, w, _ = frame.shape
        # print(_)

        able = results.pose_landmarks

        if stance:
            if able:
                landmark = results.pose_landmarks.landmark
                # try:
                landmarks = [
                    landmark[mp_pose.PoseLandmark.RIGHT_HEEL],
                    landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
                    landmark[mp_pose.PoseLandmark.LEFT_HEEL],
                    landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                ]
                '''right_heel = landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL]
                right_foot_index = landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                left_heel = landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]
                left_foot_index = landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]'''

                for lm in landmarks:
                    lm.x, lm.y = int(lm.x * w), int(lm.y * h)

                right_heel = landmarks[0]
                right_foot_index = landmarks[1]
                left_heel = landmarks[2]
                left_foot_index = landmarks[3]

                return right_heel, right_foot_index, left_heel, left_foot_index
                # except:
                #    return None, None, None, None
            else:
                return None
        if able:
            '''lm = self.results.pose.landmarks.landmark
            cx, cy = it(nlm.x * w), int(lm.y * h)'''
            lm = results.pose_landmarks.landmark

            # print(lm)

            cx, cy = int(lm[point_on_body].x * w), int(lm[point_on_body].y * h)

            return cx, cy
        return None, None

    def findStance(self, frame_side, frame_front, files):
        file = files['side']
        h, w, _ = frame_side.shape
        right_heel, right_foot_index, left_heel, left_foot_index = self.findPoint(self.mp_pose, frame_side,
                                                                                  self.results_side, None, stance=True)
        if right_heel is None:
            return None
        slope_right = abs(right_heel.y - right_foot_index.y
                          / right_heel.x - right_foot_index.x)
        slope_left = abs(left_heel.y - left_foot_index.y
                         / left_heel.x - left_foot_index.x)

        if slope_right <= slope_left:
            foot_length = math.hypot(right_heel.x - right_foot_index.x, right_heel.y - right_foot_index.y)
            total_distance = math.hypot(right_foot_index.x - left_heel.x, right_foot_index.y - left_heel.y)
            is_leading_right = True
        else:
            foot_length = math.hypot(left_heel.x - left_foot_index.x, left_heel.y - left_foot_index.y)
            total_distance = math.hypot(left_foot_index.x - right_heel.x, left_foot_index.y - right_heel.y)
            is_leading_right = False

        feet_distance = total_distance / foot_length

        annotated_image = frame_side.copy()
        cv2.line(annotated_image, (int(right_heel.x), int(right_heel.y)),
                 (int(left_foot_index.x), int(left_foot_index.y)), (255, 0, 0), 10)
        cv2.circle(annotated_image, (20, 20), 10, (255, 0, 0), 2)
        cv2.circle(annotated_image, (20, 100), 10, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(annotations_path, "ANNOTATED-SIDE-" + file), annotated_image)

        landmarks = [
            right_heel,
            right_foot_index,
            left_heel,
            left_foot_index
        ]

        for lm in landmarks:
            lm.x, lm.y = lm.x / w, lm.y / h

        improvements = StanceImprovements.Improvements(frame_side, frame_front, annotated_image, frame_front, files,
                                                       self.results_side, self.results_front, self.mp_pose)
        if 3.75 < feet_distance < 5:
            print("front stance in file " + file + " - " + str(feet_distance) + " feet")

            improvements.improve_front_stance_side(is_leading_right)
            improvements.improve_walking_stance(is_leading_right)

            # self.improveFrontStance(frame_side, frame_front, annotated_image, files, is_leading_right)

        elif feet_distance < 2.75:
            print("walking stance in file " + file + " - " + str(feet_distance) + " feet")
            improvements.improve_walking_stance(is_leading_right)

        elif 2.75 <= feet_distance <= 3.75:
            print("Probably back stance")
            improvements.improve_back_stance(is_leading_right)

        else:
            print("not anything yet in  " + file + " - " + str(feet_distance) + " feet")
            improvements.improve_walking_stance(is_leading_right)

        # improvements.improve_front_stance_front(is_leading_right)

    def improveFrontStance(self, frame_side, frame_front, annotated_image_side, files, is_leading_right):

        file_side = files['side']

        if is_leading_right:  # Leading = foot in front
            knee_leading_x, knee_leading_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                            self.mp_pose.PoseLandmark.RIGHT_KNEE)
            index_leading_x, index_leading_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                              self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
            heel_leading_x, heel_leading_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                            self.mp_pose.PoseLandmark.RIGHT_HEEL)
            ankle_following_x, ankle_following_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                                  self.mp_pose.PoseLandmark.LEFT_ANKLE, show=True)
            knee_following_x, knee_following_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                                self.mp_pose.PoseLandmark.LEFT_KNEE)
            hip_following_x, hip_following_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                              self.mp_pose.PoseLandmark.LEFT_HIP)

        else:
            knee_leading_x, knee_leading_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                            self.mp_pose.PoseLandmark.LEFT_KNEE)
            index_leading_x, index_leading_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                              self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
            heel_leading_x, heel_leading_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                            self.mp_pose.PoseLandmark.LEFT_HEEL)
            ankle_following_x, ankle_following_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                                  self.mp_pose.PoseLandmark.RIGHT_ANKLE)
            knee_following_x, knee_following_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                                self.mp_pose.PoseLandmark.RIGHT_KNEE)
            hip_following_x, hip_following_y = self.findPoint(self.mp_pose, frame_side, self.results_side,
                                                              self.mp_pose.PoseLandmark.RIGHT_HIP)

        midpoint_x = int((heel_leading_x + index_leading_x) / 2)
        midpoint_x = int((midpoint_x + index_leading_x) / 2)
        midpoint_y = int((heel_leading_y + index_leading_y) / 2)
        midpoint_y = int((midpoint_y + index_leading_y) / 2)

        slope_leading = (midpoint_y - knee_leading_y) / (midpoint_x - knee_leading_x)
        if slope_leading > 1.1 or slope_leading < 0.9:
            print("Place your knee above your toes")
            cv2.line(annotated_image_side, (midpoint_x, midpoint_y), (midpoint_x, knee_leading_y), (0, 255, 0), 5)
            cv2.imwrite(os.path.join(annotations_path, "ANNOTATED-SIDE-" + file_side), annotated_image_side)

        slope_following_lower = (knee_following_y - ankle_following_y) / (knee_following_x - ankle_following_x)
        slope_following_upper = (hip_following_y - knee_following_y) / (hip_following_x - knee_following_x)
        difference = abs(slope_following_upper - slope_following_lower)

        if difference > 2.5:
            print("fix the slope")


def main():
    '''IMAGE_FILES = [
        {
            'side': 'Images/short.png',
            'front': 'Images/90-angle.jpg'
        }
    ]'''

    cap = cv2.VideoCapture('Images/cut-video.mp4')
    frametime = 25

    detector = PoseDetector()

    '''for idx, file in enumerate(IMAGE_FILES):
        frame_side = cv2.imread(file['side'])
        frame_front = cv2.imread(file['front'])
        image_side, image_front = detector.findPose(frame_side, frame_front)
        # img = detector.findPosition(frame)
        detector.findStance(frame_side, frame_front, file)'''

    while cap.isOpened():
        ret, frame = cap.read()
        # image = detector.findPose(frame)

        if frame is None:
            print("no frame")
            # getInfo(cv2.imread("Images/90-angle.jpg"), make_video=True)
            create_video(w, h)
            cap.release()
            cv2.destroyAllWindows()
            print(path.exists("clip--1.mp4"))
            break

        # framey = cv2.imread('Images/short.png')
        # print("attmept")
        # print(frame)
        getInfo(frame)
        # print("done")

        # frame1 = cv2.imread("Images/90-angle.jpg")
        # crop_frame = frame1[0:984, 0:555]
        # frame2 = cv2.imread("Images/after.jpg")
        # difference = cv2.subtract(crop_frame, frame2)
        # difference = cv2.subtract(frame1, frame2)
        # b, g, r = cv2.split(difference)
        # print(b, g, r)
        # print(cv2.countNonZero(r))
        # lm_list = detector.findPosition(frame)

        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print(fps)

        if ret:
            cv2.imshow("Pose Detection", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
        else:
            cap.release()
            cv2.destroyAllWindows()


def get_results(mp_pose, last_frame, frame, results):
    ''''landmarks = [
                {
                    "last_ankle_right": PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'], mp_pose.PoseLandmark.RIGHT_ANKLE),
                    "last_ankle_left": PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'], mp_pose.PoseLandmark.LEFT_ANKLE),
                    "last_knee_right": PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'], mp_pose.PoseLandmark.RIGHT_KNEE),
                    "last_knee_left": PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'], mp_pose.PoseLandmark.LEFT_KNEE),
                    "last_wrist_right": PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'], mp_pose.PoseLandmark.RIGHT_WRIST),
                    "last_wrist_left": PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'], mp_pose.PoseLandmark.LEFT_WRIST)
                },
                {
                    "current_ankle_right": PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    "current_ankle_left": PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.LEFT_ANKLE),
                    "current_knee_right": PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.RIGHT_KNEE),
                    "current_knee_left": PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.LEFT_KNEE),
                    "current_wrist_right": PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.RIGHT_WRIST),
                    "current_wrist_left": PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.LEFT_WRIST)
                }
            ]'''
    landmarks = None
    try:
        landmarks = [
            [
                PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'],
                                       mp_pose.PoseLandmark.RIGHT_ANKLE),
                PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.RIGHT_ANKLE)
            ],
            [
                PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'],
                                       mp_pose.PoseLandmark.LEFT_ANKLE),
                PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.LEFT_ANKLE)
            ],
            [
                PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'],
                                       mp_pose.PoseLandmark.RIGHT_KNEE),
                PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.RIGHT_KNEE)
            ],
            [
                PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'],
                                       mp_pose.PoseLandmark.LEFT_KNEE),
                PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.LEFT_KNEE)
            ],
            [
                PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'],
                                       mp_pose.PoseLandmark.RIGHT_WRIST),
                PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.RIGHT_WRIST)
            ],
            [
                PoseDetector.findPoint(mp_pose, last_frame['frame'], last_frame['results'],
                                       mp_pose.PoseLandmark.LEFT_WRIST),
                PoseDetector.findPoint(mp_pose, frame, results, mp_pose.PoseLandmark.LEFT_WRIST)
            ]
        ]
    except Exception as e:
        print(e)
    return landmarks


if __name__ == "__main__":
    main()
