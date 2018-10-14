'''
    Author: Manjunath M
    Email: mnj.m0@yahoo.com
    Dlib Landmarks Demo
'''
import numpy as np
import dlib
import cv2
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--use_video",
                default='/dev/video0',
                help='Video File to Use. Default webcam')
argParser.add_argument("--landmark_model", 
                default='/home/mm/Workspace/Trained_Models/shape_predictor_68_face_landmarks.dat',
                help='Dlib Landmarks Predictor Model File')
args = argParser.parse_args()
predictorFile = args.landmark_model
videoSource = args.use_video

def shapeToLandmarks(shape):
    landmarks = np.zeros((2,68), dtype=np.uint32)
    for i in  range(68):
        landmarks[0,i], landmarks[1,i] = shape.part(i).x, shape.part(i).y
    return landmarks

def rectToBBox(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x,y,w,h

def plotLandmarks(img, landmarks):
    for (x,y) in landmarks.transpose():
        cv2.circle(img, (x,y), 3, (0,0,255), -1)

def plotBBox(img, x, y, w, h):
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

def main():
    bBoxDetector = dlib.get_frontal_face_detector()
    landmarksPredictor = dlib.shape_predictor(predictorFile)

    videoCapture = cv2.VideoCapture(videoSource)

    while True:
        success, frame = videoCapture.read()
        if not success:
            print('Frame Error: Exiting..')
            break
        rects = bBoxDetector(frame, 0)
        for rect in rects:
            x,y,w,h = rectToBBox(rect)
            plotBBox(frame, x,y,w,h)
            shape = landmarksPredictor(frame, rect)
            landmarks = shapeToLandmarks(shape)
            plotLandmarks(frame, landmarks)

        cv2.imshow('Video Feed', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
