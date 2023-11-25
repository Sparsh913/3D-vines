#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
import cv2
import numpy as np
import threading

bridge = CvBridge()
rectified_bot = []
rectified_top = []
countBot = 0
countTop = 0
path = '/home/uas-laptop/Kantor_Lab/3D-vines/images/2022-11-09-16-57-53.bag/'
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
    # print("resolution img 0:", cv_image0.shape)
    return cv_image0

def callback1(data):
    global rectified_bot
    global countBot
    cv_image1 = bridge.imgmsg_to_cv2(data, "bgr8")

    imgL = callback0(rospy.wait_for_message("/theia/cam0/image_raw", Image))
    imgR = cv_image1

    # print("Rectifying Bottom images")
    
    rectified_left, rectified_right = rectify_bot(imgL, imgR)
    # cv2.imwrite(f"Images/RectBot/Left_Rect_Bot_{countBot}.png", rectified_left)
    # cv2.imwrite(f"Images/RectBot/Right_Rect_Bot_{countBot}.png", rectified_right)
    countBot += 1
    # rectified_bot.append((rectified_left, rectified_right))
    # cv2.imshow("Left Rect Bot", rectified_left)
    # cv2.imshow("Right Rect Bot", rectified_right)
    # cv2.waitKey(0)
    
    # Publish the rectified images
    rect_l_bot_publisher.publish(bridge.cv2_to_imgmsg(rectified_left, "bgr8"))
    rect_r_bot_publisher.publish(bridge.cv2_to_imgmsg(rectified_right, "bgr8"))
    rate.sleep()
 

def callback2(data):

    cv_image2 = bridge.imgmsg_to_cv2(data, "bgr8")

    return cv_image2

def callback3(data):
    global rectified_top
    global countTop
    cv_image3 = bridge.imgmsg_to_cv2(data, "bgr8")

    imgL = callback2(rospy.wait_for_message("/theia/cam2/image_raw", Image))
    imgR = cv_image3

    # print("Rectifying Top images")

    rectified_left, rectified_right = rectify_top(imgL, imgR)
    # cv2.imwrite(f"Images/RectTop/Left_Rect_Top_{countTop}.png", rectified_left)
    # cv2.imwrite(f"Images/RectTop/Right_Rect_Top_{countTop}.png", rectified_right)
    # rectified_top.append((rectified_left, rectified_right))
    # cv2.imshow("Left Rect Top", rectified_left)
    # cv2.imshow("Right Rect Top", rectified_right)
    # cv2.waitKey(0)
    countTop += 1
    
    # Publish the rectified images
    rect_l_top_publisher.publish(bridge.cv2_to_imgmsg(rectified_left, "bgr8"))
    rect_r_top_publisher.publish(bridge.cv2_to_imgmsg(rectified_right, "bgr8"))
    rate.sleep()
    


def rectify_bot(imgL, imgR):

    heightL, widthL, channelsL = imgL.shape
    heightR, widthR, channelsR = imgR.shape
    cameraMatrixL = np.load(path + 'cam0/K0.npy')
    distL = np.load(path + 'cam0/D0.npy')
    RotL = np.load(path + 'cam0/R0.npy')

    cameraMatrixR = np.load(path + 'cam1/K0.npy')
    distR = np.load(path + 'cam1/D0.npy')
    RotR = np.load(path + 'cam1/R0.npy')                                       

    projMatrixL = np.load(path + 'cam0/P0.npy')
    projMatrixR = np.load(path + 'cam1/P0.npy')

    map_x_left, map_y_left = cv2.initUndistortRectifyMap(cameraMatrixL, distL, RotL, projMatrixL, (widthL, heightL), cv2.CV_32FC1)
    map_x_right, map_y_right = cv2.initUndistortRectifyMap(cameraMatrixR, distR, RotR, projMatrixR, (widthR, heightR), cv2.CV_32FC1)

    rectified_left = cv2.remap(imgL, map_x_left, map_y_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(imgR, map_x_right, map_y_right, cv2.INTER_LINEAR)

    return rectified_left, rectified_right

def rectify_top(imgL, imgR):

    heightL, widthL, channelsL = imgL.shape
    heightR, widthR, channelsR = imgR.shape
    cameraMatrixL = np.load(path + 'cam2/K0.npy')
    distL = np.load(path + 'cam2/D0.npy')
    RotL = np.load(path + 'cam2/R0.npy')

    cameraMatrixR = np.load(path + 'cam3/K0.npy')
    distR = np.load(path + 'cam3/D0.npy')
    RotR = np.load(path + 'cam3/R0.npy')                                       

    projMatrixL = np.load(path + 'cam2/P0.npy')
    projMatrixR = np.load(path + 'cam3/P0.npy')

    map_x_left, map_y_left = cv2.initUndistortRectifyMap(cameraMatrixL, distL, RotL, projMatrixL, (widthL, heightL), cv2.CV_32FC1)
    map_x_right, map_y_right = cv2.initUndistortRectifyMap(cameraMatrixR, distR, RotR, projMatrixR, (widthR, heightR), cv2.CV_32FC1)

    rectified_left = cv2.remap(imgL, map_x_left, map_y_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(imgR, map_x_right, map_y_right, cv2.INTER_LINEAR)

    return rectified_left, rectified_right



def listener():
    # rospy.Subscriber("chatter", String, callback)
    rospy.Subscriber("/theia/cam0/image_raw", Image, callback0, queue_size=qs)
    rospy.Subscriber("/theia/cam1/image_raw", Image, callback1, queue_size=qs)
    rospy.Subscriber("/theia/cam2/image_raw", Image, callback2, queue_size=qs)
    rospy.Subscriber("/theia/cam3/image_raw", Image, callback3, queue_size=qs)

    rospy.spin() # keeps python from exiting until this node is stopped

if __name__ == '__main__':
    rospy.init_node('raw2rect')
    qs = 200000
    rate = rospy.Rate(0.07)
    rect_l_bot_publisher = rospy.Publisher('left_bot_rect', Image, queue_size=qs)
    rect_r_bot_publisher = rospy.Publisher('right_bot_rect', Image, queue_size=qs)
    rect_l_top_publisher = rospy.Publisher('left_top_rect', Image, queue_size=qs)
    rect_r_top_publisher = rospy.Publisher('right_top_rect', Image, queue_size=qs)
    
    
    listener()
    # print("No. of images in rectified_bot:", len(rectified_bot))
    # print("No. of images in rectified_top:", len(rectified_top))