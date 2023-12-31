#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
import cv2
import numpy as np

bridge = CvBridge()
rectified_bot = []
rectified_top = []
countBot = 0
countTop = 0
# cv_image0 = []
# cv_image1 = []
# cv_image2 = []
# cv_image3 = []

# def callback0(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def callback0(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    # global cv_image0
    cv_image0 = bridge.imgmsg_to_cv2(data, "bgr8")
    # cv2.imshow("Image window", cv_image0)
    # cv2.waitKey(0)
    print("resolution img 0:", cv_image0.shape)
    return cv_image0

def callback1(data):
    global rectified_bot
    global countBot
    cv_image1 = bridge.imgmsg_to_cv2(data, "bgr8")

    imgL = callback0(rospy.wait_for_message("/theia/cam0/image_raw", Image))
    imgR = cv_image1

    print("Rectifying Bottom images")
    
    rectified_left, rectified_right = rectify_bot(imgL, imgR)
    cv2.imwrite(f"Images/RectBot/Left_Rect_Bot_{countBot}.png", rectified_left)
    cv2.imwrite(f"Images/RectBot/Right_Rect_Bot_{countBot}.png", rectified_right)
    countBot += 1
    # rectified_bot.append((rectified_left, rectified_right))
    # cv2.imshow("Left Rect Bot", rectified_left)
    # cv2.imshow("Right Rect Bot", rectified_right)
    # cv2.waitKey(0)
 

def callback2(data):

    cv_image2 = bridge.imgmsg_to_cv2(data, "bgr8")

    return cv_image2

def callback3(data):
    global rectified_top
    global countTop
    cv_image3 = bridge.imgmsg_to_cv2(data, "bgr8")

    imgL = callback2(rospy.wait_for_message("/theia/cam2/image_raw", Image))
    imgR = cv_image3

    print("Rectifying Top images")

    rectified_left, rectified_right = rectify_top(imgL, imgR)
    cv2.imwrite(f"Images/RectTop/Left_Rect_Top_{countTop}.png", rectified_left)
    cv2.imwrite(f"Images/RectTop/Right_Rect_Top_{countTop}.png", rectified_right)
    # rectified_top.append((rectified_left, rectified_right))
    # cv2.imshow("Left Rect Top", rectified_left)
    # cv2.imshow("Right Rect Top", rectified_right)
    # cv2.waitKey(0)
    countTop += 1


def rectify_bot(imgL, imgR):

    heightL, widthL, channelsL = imgL.shape
    heightR, widthR, channelsR = imgR.shape
    cameraMatrixL = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam0/K0.npy')
    distL = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam0/D0.npy')
    RotL = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam0/R0.npy')

    cameraMatrixR = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam1/K0.npy')
    distR = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam1/D0.npy')
    RotR = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam1/R0.npy')                                       

    projMatrixL = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam0/P0.npy')
    projMatrixR = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam1/P0.npy')

    map_x_left, map_y_left = cv2.initUndistortRectifyMap(cameraMatrixL, distL, RotL, projMatrixL, (widthL, heightL), cv2.CV_32FC1)
    map_x_right, map_y_right = cv2.initUndistortRectifyMap(cameraMatrixR, distR, RotR, projMatrixR, (widthR, heightR), cv2.CV_32FC1)

    rectified_left = cv2.remap(imgL, map_x_left, map_y_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(imgR, map_x_right, map_y_right, cv2.INTER_LINEAR)

    return rectified_left, rectified_right

def rectify_top(imgL, imgR):

    heightL, widthL, channelsL = imgL.shape
    heightR, widthR, channelsR = imgR.shape
    cameraMatrixL = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam2/K0.npy')
    distL = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam2/D0.npy')
    RotL = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam2/R0.npy')

    cameraMatrixR = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam3/K0.npy')
    distR = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam3/D0.npy')
    RotR = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam3/R0.npy')                                       

    projMatrixL = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam2/P0.npy')
    projMatrixR = np.load('/home/sparsh/Kantor_Lab/3D-vines/images/2022-11-09-13-52-53.bag/cam3/P0.npy')

    map_x_left, map_y_left = cv2.initUndistortRectifyMap(cameraMatrixL, distL, RotL, projMatrixL, (widthL, heightL), cv2.CV_32FC1)
    map_x_right, map_y_right = cv2.initUndistortRectifyMap(cameraMatrixR, distR, RotR, projMatrixR, (widthR, heightR), cv2.CV_32FC1)

    rectified_left = cv2.remap(imgL, map_x_left, map_y_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(imgR, map_x_right, map_y_right, cv2.INTER_LINEAR)

    return rectified_left, rectified_right



def listener():
    rospy.init_node('listener', anonymous=True)
    # rospy.Subscriber("chatter", String, callback)
    rospy.Subscriber("/theia/cam0/image_raw", Image, callback0)
    rospy.Subscriber("/theia/cam1/image_raw", Image, callback1)
    rospy.Subscriber("/theia/cam2/image_raw", Image, callback2)
    rospy.Subscriber("/theia/cam3/image_raw", Image, callback3)
    
    # img0 = callback0(rospy.wait_for_message("/theia/cam0/image_raw", Image))
    # img1 = callback1(rospy.wait_for_message("/theia/cam1/image_raw", Image))
    # img2 = callback2(rospy.wait_for_message("/theia/cam2/image_raw", Image))
    # img3 = callback3(rospy.wait_for_message("/theia/cam3/image_raw", Image))
    # rectified_bot.append(rectify(rospy.Subscriber("/theia/cam0/image_raw", Image, callback0), rospy.Subscriber("/theia/cam1/image_raw", Image, callback1)))
    # rectified_bot.append(rectify_bot(img0, img1))
    # rectified_top.append(rectify_top(img2, img3))

    # print("Number of images in rectified_bot:", len(rectified_bot))
    # print("Number of images in rectified_top:", len(rectified_top))

    # if (len(rectified_bot) < 7 and len(rectified_top) < 7):
    rospy.spin() # keeps python from exiting until this node is stopped

if __name__ == '__main__':
    listener()
    # print("No. of images in rectified_bot:", len(rectified_bot))
    # print("No. of images in rectified_top:", len(rectified_top))