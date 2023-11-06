import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

imgL = cv.imread(r'Cam2/Cam 2 viewpoint 0.png')
imgR = cv.imread(r'Cam3/Cam 3 viewpoint 0.png')
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

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

newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

print("Camera Matrix L", cameraMatrixL)
print("New Camera Matrix L", newCameraMatrixL)
print("Distortion coeff L", distL)
print("Rotation Matrix L", RotL)


# Stereo Vision Calibration
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
# retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate()

# R is basically a relative rotation matrix to go from Left Cam coordinate frame to right cam cooridinate frame
R = RotR @ np.linalg.inv(RotL)
print("Rotation Matrix Relative", R)
T = np.array([-165.1763711199545, 0.0, 0.0])
print("Translation vector Relative", T)

# Essential Matrix
E = np.cross(T, R)
print("Essential Matrix", E)

# Fundamental Matrix
F = np.linalg.inv(cameraMatrixR.T) @ E @ np.linalg.inv(cameraMatrixL)
print("Fundamental Matrix", F)

# Stereo rectification
rectifyScale = 1

rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], R, T, rectifyScale, (0,0))
print("Perspective Transformation Matrix", Q)
print("rectL", rectL)
print("rectR", rectR)
print("Projection Matrix Left", projMatrixL)
print("Projection Matrix Right", projMatrixR)
# projMatrixL = np.array([[1502.928698323168, 0.0, 1248.91780090332, 0.0], [0.0, 1502.928698323168, 1025.463020324707, 0.0], [0.0, 0.0, 1.0, 0.0]])
# projMatrixR = np.array([[1502.928698323168, 0.0, 1248.91780090332, -165.1763711199545], [0.0, 1502.928698323168, 1025.463020324707, 0.0], [0.0, 0.0, 1.0, 0.0]])
# rectL = projMatrixL[:,:3]
# rectR = projMatrixR[:,:3]
print("Shape projMatrixL", projMatrixL.shape)
print("Shape projMatrixR", projMatrixR.shape)
print("Shape rectL", rectL.shape)
print("Shape rectR", rectR.shape)

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")

cv_file = cv.FileStorage('Top_stereoMap_bot.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x_bot',stereoMapL[0])
cv_file.write('stereoMapL_y_bot',stereoMapL[1])
cv_file.write('stereoMapR_x_bot',stereoMapR[0])
cv_file.write('stereoMapR_y_bot',stereoMapR[1])
cv_file.write('q_bot', Q)

cv_file.release()