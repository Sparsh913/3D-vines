#!/usr/bin/env python3

###################################### Node to register point clouds from bottom stereo (Cam 0 | Cam 1) images ######################################

import cv2 
import numpy as np
import matplotlib.pyplot as plt 
# from disparity_from_raft_bot import get_edge_map
import open3d as o3d
import copy
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

def rgb2gray(rgb):
        # Converts rgb to gray
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    
def get_edge_map(img):
        # Generates edge map from the image
        speed_scale = 32
        image_dim = int(min(img.shape[0:2]))

        gray = rgb2gray(img)
        grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        # grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

        # thresholding the gradient map to generate the edge-map as a proxy of the contextual cues
        m = grad.min()
        M = grad.max()
        middle = m + (0.1 * (M - m))
        grad[grad < middle] = 0
        grad[grad >= middle] = 1

        # simple dilation to increase the edge-map thickness
        grad = cv2.dilate(grad, np.ones((3, 3), np.uint8), iterations=10)

        #simple erosion to decrease the edge-map thickness
        # grad = cv2.erode(grad, np.ones((5, 5), np.uint8), iterations=2)
        
        #writing edge map in outputs folder
        cv2.imwrite('outputs/edge_map.png', grad * 255)
        return grad

def callback0(data):
    imgL = bridge.imgmsg_to_cv2(data, "bgr8")
    return imgL

def callback1(data):
    imgR = bridge.imgmsg_to_cv2(data, "bgr8")
    return imgR

def callback2(data):
    global img_list_left
    global img_list_right
    global disp_list
    
    imgL = callback0(rospy.wait_for_message("/left_bot_rect", Image))
    img_list_left.append(imgL)
    imgR = callback1(rospy.wait_for_message("/right_bot_rect", Image))
    img_list_right.append(imgR)
    disp = np.array(data.data).reshape((data.layout.dim[0].size, data.layout.dim[1].size))
    disp_list.append(disp)
    
    if (len(img_list_left)) == 7 and (len(img_list_right)) == 7 and (len(disp_list)) == 7:
        print("Got all the images")
        heightL, widthL, channelsL = img_list_left[0].shape
        heightR, widthR, channelsR = img_list_right[0].shape
        cameraMatrixL = np.load(path + 'cam0/K0.npy')
        distL = np.load(path + 'cam0/D0.npy')
        RotL = np.load(path + 'cam0/R0.npy')

        cameraMatrixR = np.load(path + 'cam1/K0.npy')
        distR = np.load(path + 'cam1/D0.npy')
        RotR = np.load(path + 'cam1/R0.npy')                                       

        projMatrixL = np.load(path + 'cam0/P0.npy')
        projMatrixR = np.load(path + 'cam1/P0.npy')
        
        baseline = np.abs(projMatrixR[0][3] / projMatrixR[0][0])
        f_norm = np.abs(cameraMatrixL[0][0])
        cx = cameraMatrixL[0][2]
        cy = cameraMatrixL[1][2]
        
        points_clouds = []
        color_clouds = []
        
        for i in range(7):
            IMSHAPE = (1536,2048)
            ALL_AX0, ALL_AX1 = np.where(np.ones(IMSHAPE))
            ALL_AX0 = ALL_AX0.reshape(IMSHAPE)
            ALL_AX1 = ALL_AX1.reshape(IMSHAPE)

            stub = -baseline / disp_list[i]
            xyz = np.dstack((
                stub * (ALL_AX1 - cx),  # columns
                stub * (ALL_AX0 - cy),  # rows
                stub * f_norm,
            ))

            # get edge map 
            edge_map = get_edge_map(img_list_left[i])

            xyz = np.nan_to_num(xyz, copy=False, nan=1e3)
            xyz = xyz.reshape(-1, 3)
            img_list_left[i] = cv2.cvtColor(img_list_left[i], cv2.COLOR_BGR2RGB)
            colors = img_list_left[i].reshape(-1, 3)/255


            # remove points with zero in edge map
            xyz = xyz[edge_map.reshape(-1) != 0]
            colors = colors[edge_map.reshape(-1) != 0]

            # remove points with DEPTH_LIMIT > 2
            xyz1 = xyz
            DEPTH_LIMIT = 1.3
            xyz1 = xyz[xyz[:, 2] < DEPTH_LIMIT]
            colors = colors[xyz[:, 2] < DEPTH_LIMIT]

            #remove points with DEPTH_LIMIT < 0.0
            colors = colors[xyz1[:, 2] > -6]
            xyz1 = xyz1[xyz1[:, 2] > -6]

            #remove points with below some brightness threshold
            #brightness is calculated as the sum of rgb values
            brightness = np.sum(colors, axis=1)
            colors = colors[brightness > 0.3]
            xyz1 = xyz1[brightness > 0.3]

            #append 
            points_clouds.append(xyz1)
            color_clouds.append(colors)
            
        initial_transformation_matrix = np.identity(4)
        initial_transformation_matrix[0][3] = 0.22493
        initial_transformation_matrix[1][3] = 0.0
        transformation_matrix_list = [initial_transformation_matrix]


        outlier_nb = 80
        outlier_std = 0.5
        max = 6
        for i in range(1, max+1):
            pcd_1 = o3d.geometry.PointCloud()
            pcd_1.points = o3d.utility.Vector3dVector(points_clouds[i-1])
            pcd_1.colors = o3d.utility.Vector3dVector(color_clouds[i-1])
            #remove outliers
            pcd_1, ind = pcd_1.remove_statistical_outlier(nb_neighbors=outlier_nb,
                                                            std_ratio=outlier_std)  

            pcd_2 = o3d.geometry.PointCloud()
            pcd_2.points = o3d.utility.Vector3dVector(points_clouds[i])
            pcd_2.colors = o3d.utility.Vector3dVector(color_clouds[i])

            #remove outliers
            pcd_2, ind = pcd_2.remove_statistical_outlier(nb_neighbors=outlier_nb,
                                                            std_ratio=outlier_std)

            target_cloud = copy.deepcopy(pcd_1)
            source_cloud = copy.deepcopy(pcd_2)

            #downsample point cloud
            # target_cloud = target_cloud.voxel_down_sample(voxel_size=0.005)
            # source_cloud = source_cloud.voxel_down_sample(voxel_size=0.005)

            for cloud in [target_cloud, source_cloud]:
                    cloud.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30),
                    )
                    cloud.orient_normals_towards_camera_location()

            #now performing icp registration
            # icp = o3d.pipelines.registration.registration_icp(
            #     source_cloud, target_cloud, max_correspondence_distance=0.003,
            #     init=transformation_matrix_list[-1],
            #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
            #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            # )

            # #performing colored icp
            icp = o3d.pipelines.registration.registration_colored_icp(
                source_cloud, target_cloud, max_correspondence_distance=0.003,
                init=transformation_matrix_list[-1],
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=100),
                estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP())

            transformation_matrix = icp.transformation
            transformation_matrix_list.append(transformation_matrix)
            print('transformation matrix: ', transformation_matrix)

            # source_cloud.transform(transformation_matrix)

            # stitched_cloud = stitched_cloud + source_cloud

            # pcd_1 = copy.deepcopy(source_cloud)

        #now transform the point clouds to stitch them together
        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(points_clouds[max])
        pcd_1.colors = o3d.utility.Vector3dVector(color_clouds[max])
        pcd_1, ind = pcd_1.remove_statistical_outlier(nb_neighbors=outlier_nb,
                                                            std_ratio=outlier_std)  

        stitched_cloud = copy.deepcopy(pcd_1)

        for i in range(max, 0, -1):

            transformation_matrix = transformation_matrix_list[i]
            stitched_cloud =  stitched_cloud.transform(transformation_matrix)

            pcd_1 = o3d.geometry.PointCloud()
            pcd_1.points = o3d.utility.Vector3dVector(points_clouds[i-1])
            pcd_1.colors = o3d.utility.Vector3dVector(color_clouds[i-1])
            pcd_1, ind = pcd_1.remove_statistical_outlier(nb_neighbors=outlier_nb,
                                                            std_ratio=outlier_std)

            source_cloud = copy.deepcopy(pcd_1)
            stitched_cloud = stitched_cloud + source_cloud

        # exit()


        # visualize point cloud
        pcd = stitched_cloud
        #visualize point cloud
        o3d.visualization.draw_geometries([pcd])
        
        # Convert open3d point cloud to ros point cloud
        points = np.asarray(pcd.points)
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        # header.frame_id = frame_id----------------------------------------------------------- Rectify -> the point cloud might not be published correctly

        pc2_msg = point_cloud2.create_cloud_xyz32(header, points)
        
        # Publish the point cloud
        pcd_bot_publisher.publish(pc2_msg)
        
        # Clear the lists
        img_list_left = []
        img_list_right = []
        disp_list = []
    

def listener():
    rospy.Subscriber("/left_bot_rect", Image, callback0)
    rospy.Subscriber("/right_bot_rect", Image, callback1)
    rospy.Subscriber("/disp_bot", Float64MultiArray, callback2)
    rospy.spin()
    
if __name__ == '__main__':
    img_list_left = []
    img_list_right = []
    disp_list = []
    bridge = CvBridge()
    path = '/home/uas-laptop/Kantor_Lab/3D-vines/images/2022-11-09-16-57-53.bag/'
    rospy.init_node('listener', anonymous=True)
    pcd_bot_publisher = rospy.Publisher("/pcd_bot", PointCloud2, queue_size=10)
    listener()