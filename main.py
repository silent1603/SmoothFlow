from ursina import *
import cv2, time
from PIL import Image as im
import mediapipe as mp
import numpy as np
from direct.showbase.Loader import Loader
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from dtaidistance import dtw

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 modelComplexity=1,detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplexity = modelComplexity

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStype = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.modelComplexity, self.upBody, self.smooth,self.detectionCon, self.trackCon)

    def findPose(self, imgInput, draw=True):
        self.results = self.pose.process(imgInput)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(imgInput, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return imgInput

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy, cz = float(lm.x*10), float(lm.y*10), float(lm.z*10)
                self.lmList.append([id, cx, cy, cz])
        return self.lmList

def calculate_cosine_to_angle(cosine_angle):
    # Calculate the angle in radians and convert it to degrees
    angle_radians = math.acos(cosine_angle)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def compare_poses_using_Euclidean_distance(landmarks1, landmarks2) -> float:
    if not landmarks1 or not landmarks2:
        # Return a default value or handle the case where landmarks are not detected
        return 0.0

    # Extract landmark positions
    points1 = np.array([(lm[1], lm[2]) for lm in landmarks1])
    points2 = np.array([(lm[1], lm[2]) for lm in landmarks2])

    for i in range(len(points1)-1):
        points1[i] /= np.linalg.norm(points1[i])
        points2[i] /= np.linalg.norm(points2[i])

    # Calculate Euclidean distance between corresponding landmarks
    distances = np.linalg.norm(points1 - points2, axis=1)

    # Calculate a similarity score (you can customize this based on your requirements)
    similarity_score = 1.0 / (1.0 + np.mean(distances))

    return similarity_score

# Here Choose video (0 for webcam or choose Your own file):
#videoSample = "./Data/data1.mp4"
#videoCheck = "./Data/data1.mp4"
#capSample = cv2.VideoCapture(videoSample)
#capCheck = cv2.VideoCapture(videoCheck)
#mp_image = mp.Image.create_from_file("./Data/images.jpg")
cap = cv2.VideoCapture(0)
# Create an Audio object and load the MP4 file

def calculate_total_Cosine_Angle(landmarks1, landmarks2) -> float:
    if not landmarks1 or not landmarks2:
        # Return a default value or handle the case where landmarks are not detected
        return 0.0

    # Extract landmark positions
    points1 = np.array([(lm[1], lm[2]) for lm in landmarks1])
    points2 = np.array([(lm[1], lm[2]) for lm in landmarks2])

    total_cosine_angle_delta = 0.0
    total_cosine_angle_number = 0.0


    # Iterate through joint positions
    for i in range(len(points1) - 2):
        joint1_1 = points1[i]
        joint2_1 = points1[i + 1]

        joint1_2 = points2[i]
        joint2_2 = points2[i + 1]

        vA = joint1_2 - joint1_1
        vB = joint2_2 - joint2_1

        # Normalize vectors
        norm_vector_1 = vA / np.linalg.norm(vA)
        norm_vector_2 = vB / np.linalg.norm(vB)

        dot_product = np.dot(vA,vB)

        total_cosine_angle_delta += dot_product
        total_cosine_angle_number += 1.0

    score = (total_cosine_angle_delta / total_cosine_angle_number)

    return score


spd = 10
app = Ursina()
#loader = Loader( app )
#video_sound = loader.loadSfx("Data/data1.mp4")

#video_sound.play()

nose = Entity(model='sphere', color=color.blue, scale_x=0.5, scale_y=0.5, scale_z=0.5)
left_shoulder = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)
right_shoulder = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)

left_elbow = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)
right_elbow = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)

left_wrist = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)
right_wrist = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)

left_hip = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)
right_hip = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)

left_knee = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)
right_knee = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)

left_ankle = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)
right_ankle = Entity(model='sphere', color=color.red, scale_x=0.5, scale_y=0.5, scale_z=0.5)


detector = poseDetector()

window.size = (800, 800)

#sampleQuad = Entity(model="quad", position=(0, 0, 10), scale=15, color=color.white)
# #checkQuad =  Entity(model="quad", position=(20, 0, 10), scale=15, color=color.white)
frame = cv2.imread("./Data/pose-4.jpg", cv2.IMREAD_COLOR)
imgStatic = detector.findPose(frame,True)
lm_list_sample = detector.findPosition(imgStatic)
print("The variable, name is of type:", type(lm_list_sample))
lm_list_sample_Op =[lm_list_sample[11],lm_list_sample[12],lm_list_sample[13],lm_list_sample[14],lm_list_sample[15],lm_list_sample[16],lm_list_sample[23],lm_list_sample[24],lm_list_sample[25],lm_list_sample[26],lm_list_sample[27],lm_list_sample[28] ]
imgStatic.flags.writeable = True
frame = cv2.flip(frame, 1)
data = im.fromarray(frame)
data = data.convert("RGBA")
av = Texture(data)
#sampleQuad.texture = av

detector2 = poseDetector()

def update():

    success, imgCamera = cap.read()
    #imgCamera = cv2.resize(imgCamera, (frame.shape[0],frame.shape[1] ), interpolation=cv2.INTER_AREA)
    imgCamera = detector2.findPose(imgCamera)

    lmList = detector2.findPosition(imgCamera, draw=False)
    #shows all body from opencv window (optional):
    cv2.imshow('image', imgCamera)
    if success and len(lmList) != 0: #len(lmList) detect if there is an object
        lmListOp = [lmList[11], lmList[12], lmList[13], lmList[14], lmList[15], lmList[16], lmList[23], lmList[24], lmList[25],
             lmList[26], lmList[27], lmList[28]]
        #data of body!!!
        nose.setX(-lmList[0][1])
        nose.setY(-lmList[0][2])
        nose.setZ(-lmList[0][3])

        left_shoulder.setX(-lmList[11][1])
        left_shoulder.setY(-lmList[11][2])
        left_shoulder.setZ(-lmList[11][3])

        right_shoulder.setX(-lmList[12][1])
        right_shoulder.setY(-lmList[12][2])
        right_shoulder.setZ(-lmList[12][3])

        left_elbow.setX(-lmList[13][1])
        left_elbow.setY(-lmList[13][2])
        left_elbow.setZ(-lmList[13][3])

        right_elbow.setX(-lmList[14][1])
        right_elbow.setY(-lmList[14][2])
        right_elbow.setZ(-lmList[14][3])

        left_wrist.setX(-lmList[15][1])
        left_wrist.setY(-lmList[15][2])
        left_wrist.setZ(-lmList[15][3])

        right_wrist.setX(-lmList[16][1])
        right_wrist.setY(-lmList[16][2])
        right_wrist.setZ(-lmList[16][3])

        left_hip.setX(-lmList[23][1])
        left_hip.setY(-lmList[23][2])
        left_hip.setZ(-lmList[23][3])

        right_hip.setX(-lmList[24][1])
        right_hip.setY(-lmList[24][2])
        right_hip.setZ(-lmList[24][3])

        left_knee.setX(-lmList[25][1])
        left_knee.setY(-lmList[25][2])
        left_knee.setZ(-lmList[25][3])

        right_knee.setX(-lmList[26][1])
        right_knee.setY(-lmList[26][2])
        right_knee.setZ(-lmList[26][3])

        left_ankle.setX(-lmList[27][1])
        left_ankle.setY(-lmList[27][2])
        left_ankle.setZ(-lmList[27][3])

        right_ankle.setX(-lmList[28][1])
        right_ankle.setY(-lmList[28][2])
        right_ankle.setZ(-lmList[28][3])

        value = calculate_total_Cosine_Angle(lmListOp, lm_list_sample_Op)

        print(value)

    camera_control()

def camera_control():
    camera.z += held_keys["w"] * spd * time.dt
    camera.z -= held_keys["s"] * spd * time.dt
    camera.x += held_keys["d"] * spd * time.dt
    camera.x -= held_keys["a"] * spd * time.dt
    camera.y += held_keys["e"] * spd * time.dt
    camera.y -= held_keys["q"] * spd * time.dt


'''
def update():
 
    retS, frameS = capSample.read()
    retC ,frameC = capCheck.read()
    if retS and retC:

        frame = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        img = detector.findPose(frame,False)
        lm_list_sample = detector.findPosition(frame)
        img.flags.writeable = True
        frame = cv2.flip(frame, 1)
        data = im.fromarray(frame)
        data = data.convert("RGBA")
        av = Texture(data)
        sampleQuad.texture = av


        frame = cv2.cvtColor(frameC, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        img = detector.findPose(frame,False)
        lm_list_check = detector.findPosition(frame)
        img.flags.writeable = True
        frame = cv2.flip(frame, 1)
        data = im.fromarray(frame)
        data = data.convert("RGBA")
        av = Texture(data)
        checkQuad.texture = av

        print(compare_poses(lm_list_check,lm_list_sample))
'''





def input(key):
    if key == 'escape':
        application.quit()

window.fullscreen_resolution = (640, 480)

window.screen_resolution = (300, 300)
window.center_on_screen = True

window.fps_counter.enabled = True

EditorCamera()
app.run()