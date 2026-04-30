"""
Kinematics helpers for the 7 DoF Kinova arm.
Handles forward and inverse kinematics calculations, workspace bounds, and joint limits.
"""

import numpy as np
from math import pi
from src.planning import utils as ut

def calc_dh(joint_angles: list):
    """
    Takes current joint angles and returns DH table

    Args:
        joint_angles (list): List containing current joint angles of arm

    Returns:
        DH_TABLE (np.ndarray): NDarray representing DH table
    """
    q1,q2,q3,q4,q5,q6,q7 = joint_angles
    
    # DH parameters for the Kinova 7DOF arm, in the order of [alpha, a, d, theta]
    DH_TABLE = np.array([[pi, 0.0, 0.0, 0.0],
                         [pi/2, 0.0, -(0.1564 + 0.1284), q1],
                         [pi/2, 0.0, -(0.0054 + 0.0064), q2 + pi],
                         [pi/2, 0.0, -(0.2104 + 0.2104), q3 + pi],
                         [pi/2, 0.0, -(0.0064 + 0.0064), q4 + pi],
                         [pi/2, 0.0, -(0.2084 + 0.1059), q5 + pi],
                         [pi/2, 0.0, 0.0, q6 + pi],
                         [pi, 0.0, -(0.1059 + 0.0615), q7 + pi]])
    return DH_TABLE

def calc_jacobian(joint_values: list):
    """
    Calculate the Jacobian of linear velocity for the Kinova 7DOF arm.
    Note that the arm is overactuated, giving us a 6x7 Jacobian.

    Args:
        joint_values (list): Current joint angles in radians.
        radians (bool): Whether the input joint angles are in radians.

    Returns:
        np.ndarray: The Jacobian matrix.
    """
    DH = calc_dh(joint_values)
    
    H_LIST = [ut.dh_to_matrix(DH[i]) for i in range(len(joint_values) + 1)]
    H_B0, H_01, H_12, H_23, H_34, H_45, H_56, H_67 = H_LIST
    H_EE = H_B0@H_01@H_12@H_23@H_34@H_45@H_56@H_67  # Final transformation matrix for EE

    H_B1 = H_B0@H_01
    H_B2 = H_B1@H_12
    H_B3 = H_B2@H_23
    H_B4 = H_B3@H_34
    H_B5 = H_B4@H_45
    H_B6 = H_B5@H_56

    d_EE = H_EE[0:3, 3]
    k = np.array([0, 0, 1])
    jacobian = np.zeros(shape=(6, len(joint_values)))


    for i, H in enumerate([H_B0, H_B1, H_B2, H_B3, H_B4, H_B5, H_B6]):
        z = H[0:3, 0:3]@k
        r = d_EE - H[0:3, 3]
        Jv = np.cross(z, r)
        Jw = z
        jacobian[:, i] = np.hstack((Jv, Jw))

    return jacobian

def calc_inv_jacobian(joint_values: list, lambda_: float = 0.05):
        """
        Calculate the pseudo-inverse of the Jacobian with dampening.

        Using the formula: J^T * (J * J^T + lambda^2 * I)^(-1)
        where lambda is the dampening factor to avoid singularities.

        Args:
            joint_values (list): Current joint angles in radians.
            lambda_ (float): Dampening factor
        
        Returns:
            np.ndarray: The pseudo-inverse of the Jacobian matrix.

        """
        J = calc_jacobian(joint_values)
        return J.T@ np.linalg.inv(J@J.T + (lambda_**2)*np.eye(J.shape[0]))

def calc_forward_kinematics(joint_values: list, radians=True):
        """
        Calculate the forward kinematics for the Kinova 7DOF arm.

        Args:
            joint_values (list): List of joint angles in radians or degrees.
            radians (bool): Whether the input joint angles are in radians.

        Returns:
            tuple: (EndEffector, list of H matrices for each joint)
        """
        curr_joint_values = joint_values.copy()
        DH = calc_dh(joint_values)

        H_LIST = [ut.dh_to_matrix(DH[i]) for i in range(len(curr_joint_values)+1)]
        H_B0, H_01, H_12, H_23, H_34, H_45, H_56, H_67 = H_LIST
        H_EE = H_B0@H_01@H_12@H_23@H_34@H_45@H_56@H_67  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_EE @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_EE[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, H_EE

def calc_inverse_kinematics(ee: ut.EndEffector, joint_values: list[float] = [0.0]*7, tol: float = 0.01, ilimit: int = 5000):
        """
        Calculates the numerical inverse kinematics for the Kinova 7DOF arm using the Jacobian pseudo-inverse method.

        Args:
            ee (ut.EndEffector): Desired end effector pose (position and orientation).
            joint_values (list[float]): Initial guess for the joint angles in radians.
            tol (float): Tolerance for convergence. The algorithm stops when the norm of the pose difference is less than tol.
            ilimit (int): Maximum number of iterations to prevent infinite loops.
        
        Returns:
            guess (list[float]): The numerical IK solution
        """

        p_ee = np.array([ee.x, ee.y, ee.z])
        
        print(f"Initial guess provided {joint_values}")
        guess = joint_values.copy()

        icount = 0
        while icount < ilimit:
            fk_result, H_EE = calc_forward_kinematics(guess, True)
            pos_diff = p_ee - np.array([fk_result.x, fk_result.y, fk_result.z])
            # need to handle euler angle difference; can't just subtract.
            # http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
            # Compute PQ*
            R = ut.euler_to_rotm((ee.rotx, ee.roty, ee.rotz))@H_EE[:3, :3].T
            orient_diff = 0.5 * np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ])

            diff = np.hstack((pos_diff, orient_diff))
            if np.linalg.norm(diff) < tol:
                return guess
            
            guess += calc_inv_jacobian(guess)@diff
            icount += 1
