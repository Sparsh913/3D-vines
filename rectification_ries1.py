import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
from PIL import Image

imgL = cv.imread(r'Cam0/Cam 0 viewpoint 1.png')
imgR = cv.imread(r'Cam1/Cam 1 viewpoint 1.png')
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

cameraMatrixL = np.array([[1459.495636413435, 0.0, 1229.615959155466], [0.0, 1459.298418580223, 1026.665150026048], [0.0, 0.0, 1.0]])
distL = np.array([0.01428397629677654, -0.005994895706715472, 0.001693485306337808, 0.0003817540609308071, 0.0])
RotL = np.array([[0.9999912367490629, -0.002755207575390391, 0.003152024158568795], [0.00276134020913437, 0.9999943000090797, -0.001942922952641501], [-0.003146653036022185, 0.00195160973736934, 0.9999931448735556]])

cameraMatrixR = np.array([[1458.408487528577, 0.0, 1267.960936647924], [0.0, 1457.749794198929, 1029.575322761057], [0.0, 0.0, 1.0]])
distR = np.array([0.014728117256964, -0.01149371223159343, -0.0007742652135051956, 0.001557580643531326, 0.0])
RotR = np.array([[0.9999913361575908, 0.004139402821482137, 0.000439265338648475], [-0.004140250343753975, 0.9999895349394752, 0.001946365490666672], [-0.0004312039509064107, -0.001948167296132124, 0.9999980093516881]])

projMatrixL = np.array([[1521.708745579035, 0.0, 1246.444488525391, 0.0], [0.0, 1521.708745579035, 1027.341396331787, 0.0], [0.0, 0.0, 1.0, 0.0]])
projMatrixR = np.array([[1521.708745579035, 0.0, 1246.444488525391, -166.8304096393867], [0.0, 1521.708745579035, 1027.341396331787, 0.0], [0.0, 0.0, 1.0, 0.0]])
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

cv.imwrite('rectified_left_1.jpg', rectified_left)
cv.imwrite('rectified_right_1.jpg', rectified_right)