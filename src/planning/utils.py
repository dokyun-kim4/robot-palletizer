from dataclasses import dataclass
import numpy as np
from math import sqrt, sin, cos, atan2, pi, inf

@dataclass
class EndEffector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotx: float = 0.0
    roty: float = 0.0
    rotz: float = 0.0

JOINT_LIMS_7DOF = np.radians([
                [0, inf],
                [0, 128.9 * 2],
                [0, inf],
                [0, 147.8*2],
                [0, inf],
                [0, 120.3*2],
                [0, inf]
              ])

print(JOINT_LIMS_7DOF)

def check_joint_limits(theta: list[float]) -> bool:
    """Checks if the joint angles are within the specified limits.

    Args:
        theta (List[float]): Current joint angles in radians

    Returns:
        bool: True if all joint angles are within limits, False otherwise.
    """

    # # First ensure joint angles are within [0, 2*pi] by applying modulo operation
    # theta = [th % (2 * pi) for th in theta]

    for i, th in enumerate(theta):
        # skip joints with -inf, inf limits
        if not (JOINT_LIMS_7DOF[i][0] <= th <= JOINT_LIMS_7DOF[i][1]):
            return False
    return True

def dh_to_matrix(dh_params: list) -> np.ndarray:
    """
    Convert Denavit–Hartenberg (DH) parameters to a homogeneous transform using the classic
    DH convention.

    Reference: https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters

    Args:
        dh_params: DH parameters [alpha, a, d, theta], where:
            - alpha: link twist about current x (rad)
            - a: link length along current x (m)
            - d: link offset along previous z (m)
            - theta: joint angle (rad)

    Returns:
        4x4 homogeneous transformation matrix.

    Notes:
        This is the "standard" DH transform convention.
    """
    alpha, a, d, theta = dh_params
    return np.array([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])

def rotm_to_euler(R: np.ndarray) -> tuple:
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).

    This function assumes the rotation matrix uses the common Z-Y-X convention
    (yaw-pitch-roll composition). The implementation also includes handling for
    near-singular configurations (gimbal lock), where multiple Euler solutions exist.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        A tuple (roll, pitch, yaw) in radians.

    Notes:
        - If `r31` is close to ±1, pitch is near ±90° and the solution is not unique.
        - This function chooses a reasonable representative solution in those cases.
    """
    r11 = R[0,0] if abs(R[0,0]) > 1e-7 else 0.0
    r12 = R[0,1] if abs(R[0,1]) > 1e-7 else 0.0
    r21 = R[1,0] if abs(R[1,0]) > 1e-7 else 0.0
    r22 = R[1,1] if abs(R[1,1]) > 1e-7 else 0.0
    r32 = R[2,1] if abs(R[2,1]) > 1e-7 else 0.0
    r33 = R[2,2] if abs(R[2,2]) > 1e-7 else 0.0
    r31 = R[2,0] if abs(R[2,0]) > 1e-7 else 0.0

    if abs(r31) != 1:
        roll = atan2(r32, r33)        
        yaw = atan2(r21, r11)
        denom = sqrt(r11 ** 2 + r21 ** 2)
        pitch = atan2(-r31, denom)
    
    elif r31 == 1:
        # pitch is close to -90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we choose one possible solution where yaw = 0
        pitch, yaw = -pi/2, 0.0
        roll = -atan2(r12, r22)
    
    elif r31 == -1:
        # pitch is close to 90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we choose one possible solution where yaw = 0
        pitch, yaw = pi/2, 0.0
        roll = atan2(r12, r22)

    return roll, pitch, yaw

def euler_to_rotm(rpy: tuple) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.

    This uses Z-Y-X composition (yaw then pitch then roll):
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Args:
        rpy: (roll, pitch, yaw) in radians.

    Returns:
        3x3 rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, cos(rpy[0]), -sin(rpy[0])],
                    [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.array([[cos(rpy[1]), 0, sin(rpy[1])],
                    [0, 1, 0],
                    [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.array([[cos(rpy[2]), -sin(rpy[2]), 0],
                    [sin(rpy[2]), cos(rpy[2]), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x
