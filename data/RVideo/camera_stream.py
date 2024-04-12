import pyrealsense2 as rs
import numpy as np
import cv2
import os
import pdb
import datetime
import threading
import argparse
import rospy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PIL_Image
from std_msgs.msg import Bool
from multiprocessing import Process, Manager


CFG_camera_rgb = '/camera/color/image_raw'
CFG_camera_depth = '/camera/aligned_depth_to_color/image_raw'
CFG_camera_info = '/camera/aligned_depth_to_color/camera_info'
HEIGHT = 720
WIDTH = 1280
FPS = 15


class ROSCameraStream(object):

    def __init__(self):
        self.loop_rate = rospy.Rate(FPS)   # HZ
        self.cv_bridge = CvBridge() 
        self.tf2_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.mp = Manager()

        self.curr_data = self.mp.dict({
            'camera_rgb': None,
            'camera_depth': None,
            'camera_info': None,
        })

        # topics
        self.camera_rgb_sub = rospy.Subscriber(CFG_camera_rgb, Image, self.camera_rgb_cb)
        self.camera_depth_sub = rospy.Subscriber(CFG_camera_depth, Image, self.camera_depth_cb)
        self.camera_info_sub = rospy.Subscriber(CFG_camera_info, CameraInfo, self.camera_info_cb)

        # 设置帧率控制器
        # self.rate = rospy.Rate(FPS)

    def camera_rgb_cb(self, msg):
        self.curr_data['camera_rgb'] = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")

    def camera_depth_cb(self, msg):
        self.curr_data['camera_depth'] = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
    
    def camera_info_cb(self, msg):
        self.curr_data['camera_info'] = msg

    def get_camera_in(self):
        camera_in = np.array(self.curr_data['camera_info'].K).reshape(3, 3)
        return camera_in

    def get_perception_obs(self):
        rgb_image = self.curr_data['camera_rgb']
        depth_image = self.curr_data['camera_depth']
        return rgb_image, depth_image


def save_frame(bgr_image, depth_frame, depth_color_frame, output_dir, frame_idx):
    
    rgb_filename = os.path.join(output_dir, f"rgb_{frame_idx}.png")
    depth_filename = os.path.join(output_dir, f"depth_{frame_idx}.png")
    depth_color_filename = os.path.join(output_dir, f"depth_color_{frame_idx}.png")

    cv2.imwrite(rgb_filename, bgr_image)
    cv2.imwrite(depth_filename, depth_frame)
    cv2.imwrite(depth_color_filename, depth_color_frame)


def display_frames(pipeline, args):

    recording = False
    os.makedirs(args.save_root, exist_ok=True)
    base_fp = os.path.join(args.save_root, args.task)
    os.makedirs(base_fp, exist_ok=True)
    output_dir = os.path.join(base_fp, 'cache')
    frame_idx = 0 

    while True:
        rgb_image, depth_image = pipeline.get_perception_obs()
        if (rgb_image is None) or (depth_image is None):
            continue
        # pdb.set_trace()
        # print(f"color_image: {color_image}")
        # print("Finish Get.")

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((bgr_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        key = cv2.waitKey(1)
        # print(key)

        if key == ord('r'):
            recording = not recording  # 切换录制状态
            if recording is True: print("[STATE]: Recording.")
            else: print("[STATE] Waiting.")
            if recording is True:
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                output_dir = os.path.join(base_fp, timestamp)
                frame_idx = 0
                os.makedirs(output_dir)
                camera_in = pipeline.get_camera_in()
                np.save(os.path.join(output_dir, 'camera_in.npy'), camera_in)

        if recording:
            save_frame(bgr_image, depth_image, depth_colormap, output_dir, frame_idx)
            frame_idx = frame_idx + 1

        # Press 'q' to close the image window
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

        pipeline.loop_rate.sleep()
    


if __name__ == "__main__":

    # roslaunch realsense2_camera  rs_camera.launch filters:=pointcloud align_depth:=true
    
    parser = argparse.ArgumentParser('RealSense RGBD Video Record')
    parser.add_argument('--save_root', type=str, default='recorded_rgbd', help='the file-path to save the inference results')
    parser.add_argument('--task', type=str, default='fold_Clothes')
    args, opts = parser.parse_known_args()

    rospy.init_node('ros_camera_stream', anonymous=True)
    ros_aligned_stream = ROSCameraStream()
    # display_frames(ros_aligned_stream, args)

    print("Press [r] to start & end record. Press [q] to leave.")
    # # 创建一个线程来处理图形界面
    display_thread = threading.Thread(target=display_frames, args=(ros_aligned_stream, args))
    display_thread.start()
