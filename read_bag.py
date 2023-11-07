import rosbag 
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#defining path 
path = '/home/frc-ag-1/Desktop/datasets/11-9-2022/linear_scan/2022-11-09-17-48-16.bag'

#reading bag file
bag = rosbag.Bag(path)

#reading topics
topics = bag.get_type_and_topic_info()[1].keys()
print(topics)
bridge = CvBridge()
#reading messages


if not os.path.exists('./images/' + path.split('/')[-1]):
        os.mkdir('./images/' + path.split('/')[-1])
        os.mkdir('./images/' + path.split('/')[-1] + '/cam0/')
        os.mkdir('./images/' + path.split('/')[-1] + '/cam1')
        os.mkdir('./images/' + path.split('/')[-1] + '/cam2')
        os.mkdir('./images/' + path.split('/')[-1] + '/cam3')

for j in range(4):
    i=0
    for topic, msg, t in bag.read_messages(topics=['/theia/cam'+ str(j) +'/image_raw']):
        #save sensor image in folder path.split('/)[-1] + images
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        #saving image 
        cv2.imwrite('./images/' + path.split('/')[-1] + '/cam' + str(j) + '/' + str(i) + '.png', cv_img)   
        i+=1 

#print camera info
for j in range(4):
    for topic, msg, t in bag.read_messages(topics=['/theia/cam' + str(j) + '/camera_info']):

        # save intrinsic matrix as numpy array
        K = np.array(msg.K).reshape(3,3)
        np.save('./images/' + path.split('/')[-1] + '/cam' + str(j) + '/K.npy', K)
        

#save extrinsic matrix as numpy array
for j in range(4):
    i = 0 
    for topic, msg, t in bag.read_messages(topics=['/theia/cam' + str(j) + '/camera_info']):
        P = np.array(msg.P).reshape(3,4)
        np.save('./images/' + path.split('/')[-1] + '/cam' + str(j) + '/P' + str(i) + '.npy', P)
        R = np.array(msg.R).reshape(3,3)
        np.save('./images/' + path.split('/')[-1] + '/cam' + str(j) + '/R' + str(i) + '.npy', R)
        D = np.array(msg.D).reshape(1,5)
        np.save('./images/' + path.split('/')[-1] + '/cam' + str(j) + '/D' + str(i) + '.npy', D)
        K = np.array(msg.K).reshape(3,3)
        np.save('./images/' + path.split('/')[-1] + '/cam' + str(j) + '/K' + str(i) + '.npy', K)

        i+=1

#now save rectified images
for j in range(7):
    for i in range(2):
        if i == 0:
            pair = [0, 1]
        else :
            pair = [2, 3]

        imgL = cv2.imread('./images/' + path.split('/')[-1] + '/cam' + str(pair[0]) + '/' + str(j) + '.png')
        imgR = cv2.imread('./images/' + path.split('/')[-1] + '/cam' + str(pair[1]) + '/' + str(j) + '.png')

        heightL, widthL, channelsL = imgL.shape
        heightR, widthR, channelsR = imgR.shape

        cameraMatrixL = np.load('./images/' + path.split('/')[-1] + '/cam' + str(pair[0]) + '/K' + str(j) + '.npy')
        distL = np.load('./images/' + path.split('/')[-1] + '/cam' + str(pair[0]) + '/D' + str(j) + '.npy')
        RotL = np.load('./images/' + path.split('/')[-1] + '/cam' + str(pair[0]) + '/R' + str(j) + '.npy')

        cameraMatrixR = np.load('./images/' + path.split('/')[-1] + '/cam' + str(pair[1]) + '/K' + str(j) + '.npy')
        distR = np.load('./images/' + path.split('/')[-1] + '/cam' + str(pair[1]) + '/D' + str(j) + '.npy')
        RotR = np.load('./images/' + path.split('/')[-1] + '/cam' + str(pair[1]) + '/R0.npy')

        projMatrixL = np.load('./images/' + path.split('/')[-1] + '/cam' + str(pair[0]) + '/P' + str(j) + '.npy')
        projMatrixR = np.load('./images/' + path.split('/')[-1] + '/cam' + str(pair[1]) + '/P' + str(j) + '.npy')

        rectL = projMatrixL[:,:3]
        rectR = projMatrixR[:,:3]

        map_x_left, map_y_left = cv2.initUndistortRectifyMap(cameraMatrixL, distL, RotL, projMatrixL, (widthL, heightL), cv2.CV_32FC1)
        map_x_right, map_y_right = cv2.initUndistortRectifyMap(cameraMatrixR, distR, RotR, projMatrixR, (widthR, heightR), cv2.CV_32FC1)

        rectified_left = cv2.remap(imgL, map_x_left, map_y_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(imgR, map_x_right, map_y_right, cv2.INTER_LINEAR)

        cv2.imwrite('./images/' + path.split('/')[-1] + '/cam' + str(pair[0]) + '/' + str(j) + '_rectified.png', rectified_left)
        cv2.imwrite('./images/' + path.split('/')[-1] + '/cam' + str(pair[1]) + '/' + str(j) + '_rectified.png', rectified_right)


