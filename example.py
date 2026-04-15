from backend.kinova import Kinova
import sys, time
import numpy as np
import queue, threading

class Main:
        
    def __init__(self, loop_rate = 20) -> None:
        using_suction = False
        
        if(using_suction is None):
            raise ValueError("If you are using the suction cup set using_suction to true. If you are not using the suction cup set using_suction to false")
        else:
            self.kinova_robot = Kinova(is_suction=using_suction)
            
        self.LOOP_RATE = 1 / float(loop_rate)
        
        self.action_queue = queue.Queue()

        self.is_running = True
        
        self.start()
        
        self.background_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.background_thread.start()
        print("Loop Started")
        
    # DO NOT TOUCH
    def _start_loop(self):
        try:
            while self.is_running:
                if not self.action_queue.empty():
                    func, args = self.action_queue.get()
                    print(f'Executing: {func.__name__}')
                    func(*args)
                self.loop()
                time.sleep(self.LOOP_RATE)
        except Exception as e:
            print(f'ERROR Background loop crashed: {e}')
            
            
    # DO NOT TOUCH
    def shutdown(self):
        print("Shutting down gracefully")
        self.is_running = False
        self.kinova_robot.stop()
        sys.exit(0)
            
    def start(self):
        self.home = False
        self.kinova_robot.set_torque(False)
        pass        
        
    def loop(self):        
        is_7DOF = False
        
        if(is_7DOF is None):
            raise ValueError("If you are using the big robot set is_7DOF to true. If you are using the small robot set is_7DOF to false")
        
        if(is_7DOF):
            HOME_POSITION = np.array([3.10, 5.99, 3.35, 2.54, 0.14, 4.92, 1.41])
            next_position = np.array([2.67, 5.47, 2.85, 1.92, 0.14, 4.92, 1.41])
            
        else:
            HOME_POSITION = np.array([1.75, 5.76, 2.18, 2.44, 4.54, 0.0])
            next_position = np.array([0.79, 6.11, 1.48, 1.4, 6.11, 1.57])
            
        if(self.home):
            self.kinova_robot.set_joint_angles(next_position, gripper_percentage=100)
            self.home = False

        else:
            self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)
            self.home = True            

if __name__ == "__main__":
    final_project = Main()
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        final_project.shutdown()