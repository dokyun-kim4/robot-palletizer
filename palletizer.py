from backend.kinova import BaseApp
from enum import Enum
import numpy as np
import time
from src.planning.kinematics import calc_inverse_kinematics, calc_forward_kinematics
from src.planning.environment import Pallet
from src.perception.box_pose_estimator import *
from src.planning.utils import EndEffector

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
            self.kinova_robot.set_joint_angles(self.HOME_POSITION, gripper_percentage=100)

            # Save new ee pose and joint angles
            self.curr_ee = EndEffector(calc_forward_kinematics(self.HOME_POSITION)) # type: ignore
            self.curr_joint_angles = self.HOME_POSITION

            # transition state
            self.state = State.SCAN

        ## SCAN STATE ## 
        if self.state == State.SCAN:
            # Take a picture and update pose of target boxes
            self.box_poses = [(0.3, 0.3, 0.1)]
            self.state = State.APPROACH
        
        ## APPROACH STATE ## 
        if self.state == State.APPROACH:
            # Move to top of box with an offset, gripper orientation adjustment happens here as well
            ee_goal = EndEffector(*self.box_poses[0], 0, 3.14, 0)
            ee_goal.z += 0.2

            next_pose = calc_inverse_kinematics(ee_goal, self.curr_joint_angles)
            self.kinova_robot.set_joint_angles(next_pose)

            self.ee_pose = ee_goal
            self.curr_joint_angles = next_pose

            self.state = State.PICK
        
        ## PICK STATE ##
        if self.state == State.PICK:
            # Move down and grasp the box
            ee_goal = EndEffector(*self.box_poses[0], 0, 3.14, 0)
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
    simulate = True
    
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
        final_project.Palletizer()
