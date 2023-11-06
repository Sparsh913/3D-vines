import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
from PIL import Image

imgL = cv.imread(r'Cam2/Cam 2 viewpoint 0.png')
imgR = cv.imread(r'Cam3/Cam 3 viewpoint 0.png')
# grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
# grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

def downsample_image(image, reduce_factor):
    for i in range(0,reduce_factor):
        #Check if image is color or grayscale
        if len(image.shape) > 2:
            row,col = image.shape[:2]
        else:
            row,col = image.shape

        image = cv.pyrDown(image, dstsize= (col//2, row // 2))
    return image

heightL, widthL, channelsL = imgL.shape
heightR, widthR, channelsR = imgR.shape

cameraMatrixL = np.array([[1458.487107372897, 0.0, 1240.342813955822], [0.0, 1459.235418013435, 1033.06916164627], [0.0, 0.0, 1.0]])
distL = np.array([0.01403166071700596, -0.009877888639621944, -0.0002631009507331, 0.002564286563340517, 0.0])
RotL = np.array([[0.9998118475361226, 0.002095547219921453, 0.01928414395958794], [-0.002008828845999117, 0.9999877889510544, -0.004515147334100195], [-0.01929337018440567, 0.004475559253348096, 0.9998038483804197]])

cameraMatrixR = np.array([[1461.01459830991, 0.0, 1232.763797888794], [0.0, 1461.618927107633, 1049.796124062629], [0.0, 0.0, 1.0]])
distR = np.array([0.01704500052636147, -0.01327787262787529, 0.001815013722158938, -0.001461201997283755, 0.0])
RotR = np.array([[0.9998288384834567, 0.003461272523853886, 0.01817452418462287], [-0.003542945791233838, 0.9999837619575513, 0.004463558708409469], [-0.01815877947281163, -0.004527186072899344, 0.9998248663212569]])

projMatrixL = np.array([[1493.933966282113, 0.0, 1197.586318969727, 0.0], [0.0, 1493.933966282113, 1041.123760223389, 0.0], [0.0, 0.0, 1.0, 0.0]])
projMatrixR = np.array([[1493.933966282113, 0.0, 1197.586318969727, -163.6515578877036], [0.0, 1493.933966282113, 1041.123760223389, 0.0], [0.0, 0.0, 1.0, 0.0]])
rectL = projMatrixL[:,:3]
rectR = projMatrixR[:,:3]

## =======================Rectification by hand================================
# Step 1: Calculate R2 (rotation of right camera)
# R = RotR @ np.linalg.inv(RotL)
# print("R", R)
# print("determinant of R", np.linalg.det(R))

# # Step 2: Calculate R_rect
# r1 = np.array([-1.00, 0.00, 0.00])
# r2 = np.array([0.00, -1.00, 0.00])
# r3 = np.cross(r1, r2)
# print("Shape r3", r3.shape)
# R_rect = np.vstack((r1, r2, r3))
# print("Shape R_rect", R_rect.shape)
# print("R_rect", R_rect)
# print("Determinant of R_rect", np.linalg.det(R_rect))

# # Rotate left image by R_rect
# imgL = cv.warpAffine(imgL, R_rect.T[:2, :], (imgL.shape[1], imgL.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)

# # Rotate right image by R*R_rect
# imgR = cv.warpAffine(imgR, (R.T*R_rect.T)[:2, :], (imgR.shape[1], imgR.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
# print("R*R_R_rect", R*R_rect)
# imgL = downsample_image(imgL, 2)
# imgR = downsample_image(imgR, 2)

# cv.imshow('grayL_downsampled', imgL)
# cv.imshow('grayR_downsampled', imgR)
# cv.waitKey(0)

##===============================OpenCV Rectification=================================
map_x_left, map_y_left = cv.initUndistortRectifyMap(cameraMatrixL, distL, RotL, projMatrixL, (widthL, heightL), cv.CV_32FC1)
map_x_right, map_y_right = cv.initUndistortRectifyMap(cameraMatrixR, distR, RotR, projMatrixR, (widthR, heightR), cv.CV_32FC1)

rectified_left = cv.remap(imgL, map_x_left, map_y_left, cv.INTER_LINEAR)
rectified_right = cv.remap(imgR, map_x_right, map_y_right, cv.INTER_LINEAR)

rectified_left = downsample_image(rectified_left, 2)
rectified_right = downsample_image(rectified_right, 2)

cv.imshow('rectified_left', rectified_left)
cv.imshow('rectified_right', rectified_right)
cv.waitKey(0)

cv.imwrite('rectified_left__bot_0.jpg', rectified_left)
cv.imwrite('rectified_right_bot_0.jpg', rectified_right)