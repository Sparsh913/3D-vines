#!/usr/bin/env python3

############################ Node to find disparity from top stereo (Cam 2 | Cam 3) images using RAFT-Stereo ############################

import sys
sys.path.append('/home/uas-laptop/Kantor_Lab/RAFT-Stereo')
sys.path.append('/home/uas-laptop/Kantor_Lab/RAFT-Stereo/core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import rospy
from sensor_msgs.msg import Image as Image_msg
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray

def callback0(data):
    print("callback0")
    imgL = bridge.imgmsg_to_cv2(data, "bgr8")
    # cv2.imshow("Image window", cv_image0)
    # cv2.waitKey(0)
    # print("resolution img 0:", cv_image0.shape)
    return imgL

def callback1(data):
    print("callback1")
    torch.cuda.empty_cache()
    
    imgL = callback0(rospy.wait_for_message("/left_bot_rect", Image_msg))
    imgR = bridge.imgmsg_to_cv2(data, "bgr8")

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


    DEVICE = 'cuda'

    def load_image(imfile):
        # img = np.array(Image.open(imfile)).astype(np.uint8)
        img = imfile.astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def demo(args):
        model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        model.load_state_dict(torch.load(args.restore_ckpt))

        model = model.module
        model.to(DEVICE)
        model.eval()

        output_directory = Path(args.output_directory)
        output_directory.mkdir(exist_ok=True)

        with torch.no_grad():
            # left_images = sorted(glob.glob(args.left_imgs, recursive=True))
            # right_images = sorted(glob.glob(args.right_imgs, recursive=True))
            left_images = [imgL]
            right_images = [imgR]
            
            print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

            for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
                image1 = load_image(imfile1)    

                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)

                # edge_map = get_edge_map(np.array(Image.open(imfile1)))
                edge_map = get_edge_map(np.array(imfile1))

                _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
                flow_up = padder.unpad(flow_up).squeeze()

                file_stem = imfile1.split('/')[-1].split('.')[0]

                if args.save_numpy:
                    flow_up_npy = flow_up.cpu().numpy() * edge_map
                    np.save(output_directory / f"{file_stem}_masked.npy", flow_up_npy.squeeze())

                if args.save_numpy:
                    np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
                    
                plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='inferno')

                #save edge map 
                plt.imsave(output_directory / f"{file_stem}_edge_map.png", edge_map, cmap='gray')

                #multiply edge map with flow map
                flow_up = flow_up.cpu().numpy().squeeze()
                flow_up = flow_up * edge_map
                

                #save flow map with edge map
                plt.imsave(output_directory / f"{file_stem}_edge_flow_map.png", -flow_up, cmap='inferno')
                
        return flow_up.cpu().numpy().squeeze()
    disparity = demo(args)
    print("disparity shape: ", disparity.shape)
    np_msg = Float64MultiArray()
    np_msg.data = disparity
    disp_top_publisher.publish(np_msg)
    
                
def listener():
    rospy.Subscriber("/left_top_rect", Image_msg, callback0)
    rospy.Subscriber("/right_top_rect", Image_msg, callback1)
    
    rospy.spin()

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    bridge = CvBridge()
    
    # Initiate node
    rospy.init_node('disparity_from_raft_top')
    
    # Publisher
    disp_top_publisher = rospy.Publisher('disp_top', Float64MultiArray, queue_size=10)
    
    #Following two lines are required so that argparse does not parse rosparams
    sys.argv = list(filter(lambda arg: not arg.startswith('__'), sys.argv))
    sys.argc = len(sys.argv)

    args = parse_args()
    
    
    # args = parser.parse_args()

    # demo(args)
    listener()