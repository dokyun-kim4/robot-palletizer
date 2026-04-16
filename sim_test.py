import pybullet as p
import pybullet_data
import time

# 1. Connect to the PyBullet physics server in GUI mode
physicsClient = p.connect(p.GUI)

# 2. Set search path for default assets (like the ground plane)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Set gravity (x, y, z)
p.setGravity(0, 0, -9.81)

# 4. Load a flat ground plane
plane_id = p.loadURDF("plane.urdf")

# 5. Load your Kinova Gen3 7DOF URDF
# Path is relative to the root directory where this script is running
# urdf_path = "7dof/urdf/GEN3_URDF_V12.urdf"
# urdf_path = "visualizer/6dof/urdf/6dof.urdf"
urdf_path = "visualizer/7dof/urdf/7dof_epick.urdf"
start_pos = [0, 0, 0]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# useFixedBase=True anchors the base link to the world so the arm doesn't tip over
robot_id = p.loadURDF(urdf_path, start_pos, start_orientation, useFixedBase=True)

print("Successfully loaded URDF. Press Ctrl+C in the terminal to exit.")

# 6. Step the simulation forward continuously
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.) # PyBullet runs at 240 Hz by default
except KeyboardInterrupt:
    print("\nDisconnecting from PyBullet...")

# 7. Clean up
p.disconnect()