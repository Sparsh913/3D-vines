import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

imgL = cv.imread(r'Cam0/Cam 0 viewpoint 1.png')
imgR = cv.imread(r'Cam1/Cam 1 viewpoint 1.png')
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

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

cv_file = cv.FileStorage('Top_stereoMap_1.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x_1',stereoMapL[0])
cv_file.write('stereoMapL_y_1',stereoMapL[1])
cv_file.write('stereoMapR_x_1',stereoMapR[0])
cv_file.write('stereoMapR_y_1',stereoMapR[1])
cv_file.write('q_1', Q)

cv_file.release()