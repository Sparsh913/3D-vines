import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt 


# Downsamples image x number (reduce_factor) of times. 
def downsample_image(image, reduce_factor):
    for i in range(0,reduce_factor):
        #Check if image is color or grayscale
        if len(image.shape) > 2:
            row,col = image.shape[:2]
        else:
            row,col = image.shape

        image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
    return image

# Stereo Calibration and rectification

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('Top_stereoMap_bot.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x_bot').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y_bot').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x_bot').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y_bot').mat()

Q = cv_file.getNode('q_bot').mat()

imgL = cv2.imread(r'Cam2/Cam 2 viewpoint 0.png')
imgR = cv2.imread(r'Cam3/Cam 3 viewpoint 0.png')

# Show the frames
cv2.imshow("frame left", imgL)
cv2.imshow("frame right", imgR) 

cv2.waitKey(0)

# Undistort and rectify images
imgR = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
imgL = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

########=======================================================###################
##============= Taking Rectified scaled down images from my PC I created from other code
imgL = cv2.imread(r'rectified_left__bot_0.jpg')
imgR = cv2.imread(r'rectified_right_bot_0.jpg')

# Show the frames
cv2.imshow("right undistorted rectified", imgR) 
cv2.imshow("left undistorted rectified", imgL)
cv2.waitKey(0)

# Downsample each image 1 times (because they're too big)
imgL = downsample_image(imgL,0)
imgR = downsample_image(imgR,0)
# cv2.imwrite('imgL_Rectified.jpg', imgL)
# cv2.imwrite('imgR_Rectified.jpg', imgR)

imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
print("Shape imgLgray", imgLgray.shape)

# Show the frames
cv2.imshow("frame right downscaled", imgR) 
cv2.imshow("frame left downscaled", imgL)
cv2.waitKey(0)


## Create Disparity Map from Stereo Vision

# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map

# Set disparity parameters
# Note: disparity range is tuned according to specific parameters obtained through trial and error. 
block_size = 4
min_disp = 0
max_disp = 96
num_disp = max_disp - min_disp # Needs to be divisible by 16

# Create Block matching object. 
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
    numDisparities = num_disp,
    blockSize = block_size,
    uniquenessRatio = 5,
    speckleWindowSize = 50,
    speckleRange = 2,
    disp12MaxDiff = 2,
    P1 = 0 * 3 * block_size**2,#8*img_channels*block_size**2,
    P2 = 2 * 3 * block_size**2,
    mode = cv2.STEREO_SGBM_MODE_HH) #32*img_channels*block_size**2)

# stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize = block_size)
# stereo = cv2.StereoBM()

# Compute disparity map
disparity_map = stereo.compute(imgLgray, imgRgray)
disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
print("disparity_map", disparity_map.shape)
np.save('disparity_map_Bot_Viewpoint_0.npy', disparity_map)

# Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
# plt.imshow(disparity_map,'gray')
# plt.show()
cv2.imshow('Disparity Map', disparity_map)
cv2.waitKey(0)
cv2.imwrite('Disparity_Map.jpg', disparity_map)
# cv2.imwrite()
## Generate Point Cloud from Disparity Map

# Get new downsampled width and height 
h,w = imgR.shape[:2]

# Convert disparity map to float32 and divide by 16 as shown in the documentation -> https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
print(disparity_map.dtype)
disparity_map = np.float32(np.divide(disparity_map, 16.0))
print(disparity_map.dtype)

# Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
# Get color of the reprojected points
colors = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

# Get rid of points with value 0 (no depth)
mask_map = disparity_map > (disparity_map.min())
# print("mask_map")

# Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

# Function to create point cloud file
def create_point_cloud_file(vertices, colors, filename):
    colors = colors.reshape(-1,3)
    # colors = np.array([255,255,255]).reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')
    
    pass


output_file = 'pointCloud_Bot_Viewpoint_0.ply'

# Generate point cloud file
create_point_cloud_file(output_points, output_colors, output_file)