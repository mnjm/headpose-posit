'''
    Author: Manjunath M
    Email: mnj.m0@yahoo.com
    Landmark based headpose demo
'''
import cv2
import dlib
import math
import numpy as np
import argparse

landmarkModelFile = '/home/mm/Workspace/Trained_Models/shape_predictor_68_face_landmarks.dat'

argParser = argparse.ArgumentParser()
argParser.add_argument('--landmark_model', default=landmarkModelFile,
                    help='Location of the Landmark File.')
argParser.add_argument('--video_file', default='/dev/video0',
                    help='Video Source File. Default Web Cam')
args = argParser.parse_args()
landmarkModelFile = args.landmark_model
videoSource = args.video_file

faceBBoxDetector = dlib.get_frontal_face_detector()
landmarksPredictor = dlib.shape_predictor(landmarkModelFile)

# 3D landmarks (From LearnOpenCV.com)
landmarks3d = np.array([
                        (0.0, 0.0, 0.0),             # Nose tip
                        (0.0, -330.0, -65.0),        # Chin
                        (-225.0, 170.0, -135.0),     # Left eye left corner
                        (225.0, 170.0, -135.0),      # Right eye right corner
                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                        (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
# Corresponding 2D landmarks
landmarks2dIdx = np.array((33, 8, 36, 45, 48, 54), dtype=np.uint32)

def getCameraMatrix(imgSize):
    focalLength = imgSize[1]
    center = (imgSize[1]/2, imgSize[0]/2)
    cameraMat = np.array([
                        (focalLength, 0, center[0]),
                        (0, focalLength, center[1]),
                        (0, 0, 1)
                        ], dtype=np.float64)
    return cameraMat

def rotationMatrixToEulerAngles(R):
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    x *= 180.0/np.pi
    y *= 180.0/np.pi
    z *= 180.0/np.pi
    return x, y, z

def getLandmarks(img):
    global faceBBoxDetector
    global landmarksPredictor

    def shapeToLandmarks(shape):
        landmarks = np.zeros((2,68), dtype=np.uint32)
        for i in  range(68):
            landmarks[0,i], landmarks[1,i] = shape.part(i).x, shape.part(i).y
        return landmarks

    # def getMaxAreaRect(rects):
    #     areas = [(rect.bottom()-rect.top())*(rect.right()-rect.left()) for rect in rects]    
    #     areas = np.array(areas)
    #     idx = np.where(areas == max(areas))
    #     return rects[idx]

    rects = faceBBoxDetector(img, 0)
    if len(rects) == 0:
        success = False
        return success, None
    else: success = True
    # rect = getMaxAreaRect(rects)
    rect = rects[0]
    shape = landmarksPredictor(img, rect)
    return success, shapeToLandmarks(shape)

def plotLandmarks(img, landmarks):
    for (x,y) in landmarks.transpose():
        cv2.circle(img, (x,y), 3, (0,0,255), -1)

def plotTextonImg(img, text):
    cv2.putText(img, text, (15,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

def main():
    videoCapture = cv2.VideoCapture(videoSource)
    success, frame = videoCapture.read()
    if not success:
        print('Error: Could not read the frame')
        return
    cameraMat = getCameraMatrix(frame.shape)
    distCoeffs = np.zeros((4,1)) # Assuming no distortions.

    while True:
        success, frame = videoCapture.read()
        if not success:
            print('Error: Could not read the frame')
            break
        success, landmarks = getLandmarks(frame)
        if not success:
            plotTextonImg(frame, 'Face not detected.')
            cv2.imshow('Video Feed', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue
        landmarks2d = landmarks[:, landmarks2dIdx]
        plotLandmarks(frame, landmarks2d)
        landmarks2d = landmarks2d.astype(np.float64)
        success, rotationVec, translationVec = cv2.solvePnP(landmarks3d, landmarks2d.T, cameraMat,
                                                distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rotationMat = cv2.Rodrigues(rotationVec)[0]
        pitch, yaw, roll = rotationMatrixToEulerAngles(rotationMat)
        plotTextonImg(frame, "Pitch:{:3.2}, Yaw:{:3.2}, Roll:{:3.2} ".format(pitch, yaw, roll))
        noseEndPoint = cv2.projectPoints(np.array([(0,0,1000.0)]), rotationVec, translationVec,
                                                cameraMat, distCoeffs)[0]
        
        point1 = (int(landmarks2d[0,0]), int(landmarks2d[1,0]))
        point2 = (int(noseEndPoint[0][0][0]), int(noseEndPoint[0][0][1]))
        cv2.line(frame, point1, point2, (255,0,0), 2)

        cv2.imshow('Video Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    videoCapture.release()

if __name__ == '__main__':
    main()
