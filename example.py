from backend.kinova import BaseApp
import numpy as np
from src.planning.kinematics import calc_inverse_kinematics

class Main(BaseApp):
        
    def start(self):
        self.home = False     
        
    def loop(self):        

        HOME_POSITION = np.array([3.10, 5.99, 3.35, 2.54, 0.14, 4.92, 1.41])
        next_position = np.array([3.36416789, 6.00268663, 3.03072253, 2.54871456, 0.10663194, 4.93241474, 1.51049469])
            
        if(self.home):
            self.kinova_robot.set_joint_angles(next_position, gripper_percentage=100)
            print(self.kinova_robot.get_ee_pose())
            self.home = False

        else:
            self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)
            print(self.kinova_robot.get_ee_pose())
            self.home = True            

if __name__ == "__main__":
    simulate = True
    
    if(simulate is None):
        raise ValueError("Pick simulate or real world robot")
    
    if simulate:
        # final_project = Main(simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")
        final_project = Main(simulate=True, urdf_path="visualizer/7dof/urdf/7dof.urdf")
        pass
    else:
        final_project = Main(is_suction=False)
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        final_project.shutdown()
