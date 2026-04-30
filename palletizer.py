from backend.kinova import BaseApp
from enum import Enum
import numpy as np
import time
from src.planning.kinematics import calc_inverse_kinematics, calc_forward_kinematics
from src.planning.environment import Pallet
from src.perception.box_pose_estimator import *
from src.planning.utils import EndEffector
import pyrealsense2 as rs
import cv2
import math

def camera_to_robot_base(point_camera):
    rvec = np.array([[-2.336613872251611], [-2.038652935760678], [-0.0073696870535258]], dtype=np.float64)
    tvec = np.array([[-0.012022628847092487], [-0.38039863812963065], [1.3811058819390545]], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    P_camera = np.array(point_camera, dtype=np.float64).reshape(3, 1)
    R_inv = R.T
    P_base = R_inv @ (P_camera - tvec)
    return P_base.flatten()

def tag_to_handle_offset(point_camera, rvec, offset_x=0.023, offset_y=0.115, offset_z=0.03):
    T_tag_to_handle = np.array([[offset_x], [offset_y], [offset_z]], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    P_handle = np.array(point_camera, dtype=np.float64).reshape(3, 1) + R @ T_tag_to_handle
    return P_handle.flatten()

def get_all_handle_base_positions():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Failed to start high res stream, falling back to 640x480: {e}")
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        
    align = rs.align(rs.stream.color)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    print("Scanning for all ArUco tags in scene...")
    handle_poses = []
    
    try:
        for _ in range(30):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame: continue
            
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if ids is not None and len(ids) > 0:
                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                camera_matrix = np.array([[color_intrin.fx, 0, color_intrin.ppx],
                                          [0, color_intrin.fy, color_intrin.ppy],
                                          [0, 0, 1]], dtype=np.float64)
                dist_coeffs = np.array(color_intrin.coeffs, dtype=np.float64)
                
                marker_length = 0.05
                obj_points = np.array([
                    [-marker_length/2, marker_length/2, 0],
                    [marker_length/2, marker_length/2, 0],
                    [marker_length/2, -marker_length/2, 0],
                    [-marker_length/2, -marker_length/2, 0]
                ], dtype=np.float32)

                for i in range(len(ids)):
                    c = corners[i][0]
                    center_x = int(np.mean(c[:, 0]))
                    center_y = int(np.mean(c[:, 1]))
                    
                    success, rvec, tvec = cv2.solvePnP(obj_points, c, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    
                    depth = depth_frame.get_distance(center_x, center_y)
                    if depth > 0 and success:
                        point_camera = rs.rs2_deproject_pixel_to_point(color_intrin, [center_x, center_y], depth)
                        point_handle_camera = tag_to_handle_offset(point_camera, rvec)
                        handle_base_pos = camera_to_robot_base(point_handle_camera)
                        
                        rvec_base_to_cam = np.array([[-2.336613872251611], [-2.038652935760678], [-0.0073696870535258]], dtype=np.float64)
                        R_base_to_cam, _ = cv2.Rodrigues(rvec_base_to_cam)
                        R_cam_to_base = R_base_to_cam.T
                        
                        R_tag_to_cam, _ = cv2.Rodrigues(rvec)
                        R_tag_to_base = R_cam_to_base @ R_tag_to_cam
                        
                        sy = math.sqrt(R_tag_to_base[0,0] * R_tag_to_base[0,0] + R_tag_to_base[1,0] * R_tag_to_base[1,0])
                        singular = sy < 1e-6
                        if not singular:
                            rotx = math.atan2(R_tag_to_base[2,1], R_tag_to_base[2,2])
                            roty = math.atan2(-R_tag_to_base[2,0], sy)
                            rotz = math.atan2(R_tag_to_base[1,0], R_tag_to_base[0,0])
                        else:
                            rotx = math.atan2(-R_tag_to_base[1,2], R_tag_to_base[1,1])
                            roty = math.atan2(-R_tag_to_base[2,0], sy)
                            rotz = 0
                        handle_base_rot = np.array([rotx, roty, rotz])
                        
                        handle_poses.append((handle_base_pos, handle_base_rot))
                
                if len(handle_poses) > 0:
                    break
    finally:
        pipeline.stop()
        
    return handle_poses

class State(Enum):
    HOME = 1
    SCAN = 2
    APPROACH = 3
    PICK = 4
    PALLETIZE = 5

class Palletizer(BaseApp):

    def start(self):
        self.HOME_POSITION = np.deg2rad(np.array([360, 340, 180, 214, 0 , 310, 90]))
        self.state = State.HOME
        self.pallet = Pallet(first_box_position=(0.0, 0.0, 0.0))
        self.box_poses = []
        self.ee_pose = None
        self.curr_joint_angles = None
    
    def loop(self):
        
        ## HOME STATE ##
        if self.state == State.HOME:
            self.kinova_robot.set_joint_angles(self.HOME_POSITION, gripper_percentage=0)

            # Save new ee pose and joint angles
            self.curr_ee = EndEffector(calc_forward_kinematics(self.HOME_POSITION)) # type: ignore
            self.curr_joint_angles = self.HOME_POSITION

            # transition state
            self.state = State.SCAN

        ## SCAN STATE ## 
        if self.state == State.SCAN:
            print("Scanning for boxes...")
            found_poses = get_all_handle_base_positions()
            
            if len(found_poses) == 0:
                print("No tags found, retrying scan in 1s...")
                time.sleep(1)
                return # stays in SCAN state
            
            print(f"Found {len(found_poses)} box(es).")
            self.box_poses = found_poses
            self.state = State.APPROACH
        
        ## APPROACH STATE ## 
        if self.state == State.APPROACH:
            pos, rot = self.box_poses[0]
            # Move to top of box with an approach clearance, applying handle orientation
            ee_goal = EndEffector(*pos, 0, math.pi, (rot[2] + math.pi/2) % (2*math.pi))
            ee_goal.z += 0.145 + 0.15  # 14.5cm base offset + 15cm approach clearance

            next_pose = calc_inverse_kinematics(ee_goal, self.curr_joint_angles)
            self.kinova_robot.set_joint_angles(next_pose)

            self.ee_pose = ee_goal
            self.curr_joint_angles = next_pose

            self.state = State.PICK
        
        ## PICK STATE ##
        if self.state == State.PICK:
            pos, rot = self.box_poses[0]
            # Move down and grasp the box
            ee_goal = EndEffector(*pos, 0, math.pi, (rot[2] + math.pi/2) % (2*math.pi))
            ee_goal.z = 0.23
            next_pose = calc_inverse_kinematics(ee_goal, self.curr_joint_angles)
            self.kinova_robot.set_joint_angles(next_pose)
            self.kinova_robot.close_gripper()

            self.ee_pose = ee_goal
            self.curr_joint_angles = next_pose

            self.state = State.PALLETIZE

        ## PALLETIZE STATE
        if self.state == State.PALLETIZE:
            self.state = State.HOME


if __name__ == "__main__":
    simulate = False
    
    if(simulate is None):
        raise ValueError("Pick simulate or real world robot")
    
    if simulate:
        final_project = Palletizer(simulate=True, urdf_path="visualizer/7dof/urdf/7dof.urdf")
        pass
    else:
        final_project = Palletizer(is_suction=False)
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        final_project.shutdown()

