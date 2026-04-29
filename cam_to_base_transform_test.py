import pyrealsense2 as rs
import numpy as np
import cv2

def camera_to_robot_base(point_camera):
    """
    Converts a 3D point in camera coordinates to the robot base coordinate system.
    
    Args:
        point_camera: A list or numpy array [x, y, z] in camera coordinates (e.g. meters).
        
    Returns:
        numpy array [x, y, z] in robot base coordinates.
    """
    # Calibration vectors
    rvec = np.array([[-2.336613872251611], [-2.038652935760678], [-0.0073696870535258]], dtype=np.float64)
                     
    tvec = np.array([[-0.012022628847092487], [-0.38039863812963065], [1.3811058819390545]], dtype=np.float64)
    
    R, _ = cv2.Rodrigues(rvec)
    
    # Format the input point as a 3x1 column vector
    P_camera = np.array(point_camera, dtype=np.float64).reshape(3, 1)
    
    # Apply the inverse transformation
    R_inv = R.T
    P_base = R_inv @ (P_camera - tvec)
    
    return P_base.flatten()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            
            for i in range(len(ids)):
                # Calculate center of the tag
                c = corners[i][0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))
                
                depth = depth_frame.get_distance(center_x, center_y)
                
                if depth > 0:
                    # Deproject pixel to 3D point in camera coordinates
                    point_camera = rs.rs2_deproject_pixel_to_point(color_intrin, [center_x, center_y], depth)
                    
                    # Convert to robot base coordinates
                    point_base = camera_to_robot_base(point_camera)
                    
                    text_cam = f"Cam: {point_camera[0]:.3f}, {point_camera[1]:.3f}, {point_camera[2]:.3f}m"
                    text_base = f"Base: {point_base[0]:.3f}, {point_base[1]:.3f}, {point_base[2]:.3f}m"
                    
                    cv2.putText(color_image, text_cam, (center_x, center_y - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(color_image, text_base, (center_x, center_y - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    cv2.circle(color_image, (center_x, center_y), 4, (0, 0, 255), -1)


        cv2.imshow('RealSense', np.hstack((color_image, depth_colormap)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()