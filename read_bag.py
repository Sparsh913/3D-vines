import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt
import os

#defining path 
path = '/home/uas-laptop/Kantor_Lab/3D-vines/2023-12-04-15-20-37.bag'

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

for j in range(14):
    i=0
    for topic, msg, t in bag.read_messages(topics=['/theia/cam'+ str(j) +'/image_raw']):
        #save sensor image in folder path.split('/)[-1] + images
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        #saving image 
        cv2.imwrite('./images/' + path.split('/')[-1] + '/cam' + str(j) + '/' + str(i) + '.png', cv_img)   
        i+=1 

#print camera info
for j in range(14):
    for topic, msg, t in bag.read_messages(topics=['/theia/cam' + str(j) + '/camera_info']):

        # save intrinsic matrix as numpy array
        K = np.array(msg.K).reshape(3,3)
        np.save('./images/' + path.split('/')[-1] + '/cam' + str(j) + '/K.npy', K)
        

#save extrinsic matrix as numpy array
for j in range(14):
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
        
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,10)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

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

        #finding corresponding points
        # Initiate SIFT detector
        # sift = cv2.SIFT_create()
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(rectified_left,None)
        kp2, des2 = sift.detectAndCompute(rectified_right,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        pts1 = []
        pts2 = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        
        # now calculating epipolar lines
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        print(F)
        
        img1 = rectified_left
        img2 = rectified_right
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        
        
        # We select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        random_indices = np.random.choice(pts1.shape[0], 10)
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = drawlines(img1,img2,lines1[random_indices],pts1[random_indices],pts2[random_indices])
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        #defining drawlines function
        
        img3,img4 = drawlines(img2,img1,lines2[random_indices],pts2[random_indices],pts1[random_indices])
        plt.subplot(121),plt.imshow(img5)
        plt.subplot(122),plt.imshow(img3)
        plt.show()
        
        cv2.imwrite('./images/' + path.split('/')[-1] + '/cam' + str(pair[0]) + '/' + str(j) + '_rectified.png', rectified_left)
        cv2.imwrite('./images/' + path.split('/')[-1] + '/cam' + str(pair[1]) + '/' + str(j) + '_rectified.png', rectified_right)


