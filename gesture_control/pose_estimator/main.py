import ctypes
import multiprocessing
from typing import Sequence
import rclpy
from rclpy.duration import Duration
from  sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from  dataclasses import  dataclass
import cv2

import numpy as np
from scipy.spatial.transform import Rotation

from .pose_estimation import *
import tf2_ros

from ros2_numpy import numpify
from cv_bridge import CvBridge

from gesture_control_interfaces.msg import BodyLandmarksStamped

from functools import partialmethod

from rclpy.logging import LoggingSeverity

@dataclass
class CameraConfig:
    height: int = None
    width: int = None
    d: Sequence[float] = None
    k: Sequence[float] = None # Camera matrix, 3x3 row-major
    r: Sequence[float] = None # Rectification matrix, 3x3 row-major
    p: Sequence[float] = None # Projection  matrix, 3x4 row-major


class PoseEstimatorNode(rclpy.node.Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup attributes
        self.camera_info = {
            'left': CameraInfo(),
            'right': CameraInfo()
        }

        self._is_ready = False

        # setup params
        self.param_camera_left_frame = self.declare_parameter('camera_left_frame', value='camera_left_frame')
        self.param_camera_right_frame = self.declare_parameter('camera_right_frame', value='camera_right_frame')
        self.param_camera_colorspace = self.declare_parameter('camera_colorspace', value='GRAY')
        self.param_image_output_rate_hz = self.declare_parameter('image_output_rate_hz', value=10.0)
        self.param_landmark_output_rate_hz = self.declare_parameter('landmark_output_rate_hz', value=10.0)
        self.param_output_frame = self.declare_parameter('output_frame', value='stereo_camera_frame')

        # setup subscribers
        self.sub_image = {
            'left': self.create_subscription(
                        Image,
                        'left/image_raw',
                        lambda msg: self.image_callback('left', msg),
                        rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
                    ),
            'right': self.create_subscription(
                        Image,
                        'right/image_raw',
                        lambda msg: self.image_callback('right', msg),
                        rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
                    )
        }

        self.sub_cam_info = {
            'left': self.create_subscription(
                        CameraInfo,
                        'left/camera_info',
                        lambda msg: self.camera_info_callback('left', msg),
                        rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value
                    ),
            'right': self.create_subscription(
                        CameraInfo,
                        'right/camera_info',
                        lambda msg: self.camera_info_callback('right', msg),
                        rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value
                    )
        }
        
        # setup publishers
        self.pub_image = {
            'left': self.create_publisher(
                    Image,
                    'left/image_rect_color',
                    rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
                ),
            'right': self.create_publisher(
                    Image,
                    'right/image_rect_color',
                    rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
                ),
        }


        self.pub_body_landmarks = self.create_publisher(
            BodyLandmarksStamped,
            'body_landmarks',
            rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value
        )

        # setup tf listener
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # setup background process and shared memory
        self._stop_event = multiprocessing.Event()
        self._bg_proc = {
            'left': None,
            'right': None
        }

        # lock and flag for input shmem
        self._lock_in = {
            'left': multiprocessing.Lock(),
            'right': multiprocessing.Lock()
        }

        self._flag_in = {
            'left': multiprocessing.Value(ctypes.c_bool),
            'right': multiprocessing.Value(ctypes.c_bool)
        }

        # shared memory for input images
        self._shmem_img_in = {
            'left': None,
            'right': None
        }

        # lock and flag for output shmem
        self._lock_out = {
            'left': multiprocessing.Lock(),
            'right': multiprocessing.Lock()
        }

        self._flag_out = {
            'left': multiprocessing.Value(ctypes.c_bool),
            'right': multiprocessing.Value(ctypes.c_bool)
        }

        # shared memory for output images
        self._shmem_img_out = {
            'left': None,
            'right': None
        }

        # local cache for output images
        self._local_img_out = {
            'left': None,
            'right': None
        }

        # shared memory for output landmarks
        self._shmem_landmarks_out = {
            'left': {
                'body': None,
                'face': None,
                'hands': {
                    'left': None,
                    'right': None
                }
            },
            'right': {
                'body': None,
                'face': None,
                'hands': {
                    'left': None,
                    'right': None
                }
            },
        }

        # setup CV bridge
        self._cv_bridge = CvBridge()

    def camera_info_callback(self, cam, msg):
        '''
        Callback for camera_info messages.

        Updates camera_info cache and triggers teardown/setup of background
        processes if camera_info is changed.

        Parameters
        ==========
        cam : str
            'left' or 'right' camera
        msg : sensor_msgs.msg.CameraInfo
            CameraInfo ROS message.
        '''

        # ignore timestamp
        msg.header.stamp = rclpy.time.Time(seconds=0).to_msg()

        if msg != self.camera_info[cam]:
            # something changed, update
            self.camera_info[cam] = msg
            self.setup_bg_proc()

    def image_callback(self, cam, msg):
        '''
        Passes image via shmem to background process for processing.

        Parameters
        ==========
        cam : str
            'left' or 'right' camera
        msg : sensor_msgs.msg.Image
            Image ROS message.
        '''
        if self._is_ready:
            img = self._cv_bridge.imgmsg_to_cv2(msg) # passthrough encoding
            
            # img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding=self.param_camera_colorspace.value)

            with self._lock_in[cam]:
                np.copyto(self._shmem_img_in[cam].as_numpy(), img)
                self._flag_in[cam].value = True
    
    def setup_bg_proc(self):
        '''
        Setup background processes.

        Computes stereo rectification parameters and maps based on camera_info cache.
        Sets up shared memory for use with background processes.
        Initializes and starts background processes.
        '''
        
        self.get_logger().log('Initializing background processes...', LoggingSeverity.INFO)

        # unset is ready flag
        self._is_ready = False

        # set the stop event
        self._stop_event.set()

        # compute undistortion maps
        K_left  = np.array(self.camera_info['left'].k).reshape(3,3)
        D_left  = np.array(self.camera_info['left'].d[:4])

        K_right = np.array(self.camera_info['right'].k).reshape(3,3)
        D_right = np.array(self.camera_info['right'].d[:4])
        (width, height) = (self.camera_info['left'].width, self.camera_info['left'].height)

        # Get the relative extrinsics between the left and right camera
        camera_left_frame = self.param_camera_left_frame.value
        camera_right_frame  = self.param_camera_right_frame.value

        try:
            _tf = self.tf_buffer.lookup_transform(
                target_frame=camera_right_frame,
                source_frame=camera_left_frame,
                time=self.get_clock().now(),
                timeout=Duration(seconds=1)
            )
        
        except tf2_ros.TransformException as ex:
            self.get_logger().log('Could not get {} -> {} transform. {}'.format(camera_left_frame, camera_right_frame, ex), LoggingSeverity.WARN)

            # try again when ready
            self.get_logger().log('Waiting for transform...'.format(camera_left_frame, camera_right_frame, ex), LoggingSeverity.INFO)
            fut = self.tf_buffer.wait_for_transform_async(
                target_frame=camera_right_frame,
                source_frame=camera_left_frame,
                time=self.get_clock().now()
            )

            self.executor.spin_until_future_complete(fut)

            return


        T = np.array(numpify(_tf.transform.translation))
        R = Rotation.from_quat(numpify(_tf.transform.rotation)).as_matrix()


        stereo_fov_rad = 90 * (np.pi/180)  # 90 degree desired fov
        stereo_height_px = 600          # 300x300 pixel stereo output
        stereo_focal_px = stereo_height_px/2 / np.tan(stereo_fov_rad/2)

        # We set the left rotation to identity and the right rotation
        # the rotation between the cameras
        R_left = np.eye(3)
        R_right = R

        stereo_width_px = stereo_height_px
        self.stereo_size = (stereo_width_px, stereo_height_px)
        stereo_cx = (stereo_width_px - 1)/2
        stereo_cy = (stereo_height_px - 1)/2

        # Construct the left and right projection matrices, the only difference is
        # that the right projection matrix should have a shift along the x axis of
        # baseline*focal_length
        self.P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                        [0, stereo_focal_px, stereo_cy, 0],
                        [0,               0,         1, 0]])
        self.P_right = self.P_left.copy()
        self.P_right[0][3] = T[0]*stereo_focal_px

        # Create an undistortion map for the left and right camera which applies the
        # rectification and undoes the camera distortion. This only has to be done
        # once
        m1type = cv2.CV_32FC1
        distortion_maps = {
            'left': cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, self.P_left, self.stereo_size, m1type),
            'right': cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, self.P_right, self.stereo_size, m1type)
        }

        self.get_logger().log('Stopping background processes...', LoggingSeverity.INFO)
        
        # ensure all processes stopped and joined
        for proc in self._bg_proc.values():
            if proc is not None:
                proc.join()
        
        self.get_logger().log('Background process stopped.', LoggingSeverity.INFO)

        # clear stop event
        self._stop_event.clear()

        # get colorspace from param
        _cs = self.param_camera_colorspace.value
        if _cs == 'GRAY':
            _cs = ColorSpace.GRAY
        elif _cs == 'RGB':
            _cs = ColorSpace.RGB
        elif _cs == 'BGR':
            _cs = ColorSpace.BGR
        elif _cs == 'HSV':
            _cs = ColorSpace.HSV
        else:
            raise RuntimeError('Unknown camera colorspace.')

        self.get_logger().log('Setting up shared memory...', LoggingSeverity.INFO)

        for cam in ['left', 'right']:
            # setup local cache
            self._local_img_out[cam] = np.empty((stereo_height_px, stereo_width_px, 3), dtype=np.uint8)

            # setup shared memory
            self._shmem_img_in[cam] = SharedImage(width, height, _cs, changed_flag=self._flag_in[cam])
            self._shmem_img_out[cam] = SharedImage(stereo_width_px, stereo_height_px, ColorSpace.RGB, changed_flag=self._flag_out[cam])

            self._shmem_landmarks_out[cam]['body'] = SharedNumpyArray((PoseEstimator2DProcess.NUM_BODY_LANDMARKS,2), dtype=np.double, changed_flag=self._flag_out[cam])
            self._shmem_landmarks_out[cam]['face'] = SharedNumpyArray((PoseEstimator2DProcess.NUM_FACE_LANDMARKS,2), dtype=np.double, changed_flag=self._flag_out[cam])
            self._shmem_landmarks_out[cam]['hands']['left'] = SharedNumpyArray((PoseEstimator2DProcess.NUM_HAND_LANDMARKS,2), dtype=np.double, changed_flag=self._flag_out[cam])
            self._shmem_landmarks_out[cam]['hands']['right'] = SharedNumpyArray((PoseEstimator2DProcess.NUM_HAND_LANDMARKS,2), dtype=np.double, changed_flag=self._flag_out[cam])

            # setup bg process
            _config = PoseEstimator2DConfig(
                distortion_maps=distortion_maps[cam],
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self._bg_proc[cam] = PoseEstimator2DProcess(
                config=_config,
                in_lock=self._lock_in[cam],
                in_image=self._shmem_img_in[cam],
                out_lock=self._lock_out[cam],
                out_image=self._shmem_img_out[cam],
                out_body_landmarks=self._shmem_landmarks_out[cam]['body'],
                out_left_hand_landmarks=self._shmem_landmarks_out[cam]['hands']['left'],
                out_right_hand_landmarks=self._shmem_landmarks_out[cam]['hands']['right'],
                out_face_landmarks=self._shmem_landmarks_out[cam]['face'],
                stop_event=self._stop_event,
                out_annotate=PoseAnnotationType.BODY,
                name='{}BackgroundProcess'.format(cam.capitalize())
            )

        self.get_logger().log('Shared memory set up.', LoggingSeverity.INFO)

        # start bg processes
        for proc in self._bg_proc.values():
            proc.start()

        self.get_logger().log('Background processes started.', LoggingSeverity.INFO)

        # set is ready flag
        self._is_ready = True

        # setup timer to output rectified and body pose
        image_output_rate_hz = self.param_image_output_rate_hz.value
        self.image_timer = self.create_timer(1.0/image_output_rate_hz, self.image_timer_callback)

        landmark_output_rate_hz = self.param_landmark_output_rate_hz.value
        self.landmark_timer = self.create_timer(1.0/landmark_output_rate_hz, self.landmark_timer_callback)

        self.get_logger().log('Timers initialized.', LoggingSeverity.INFO)

    def image_timer_callback(self):
        '''
        Callback to publish rectified images.
        '''
        for cam in ['left', 'right']:
            with self._lock_out[cam]:
                np.copyto(self._local_img_out[cam], self._shmem_img_out[cam].as_numpy())
        
            msg = self._cv_bridge.cv2_to_imgmsg(self._local_img_out[cam], encoding='rgb8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.param_output_frame.value
            self.pub_image[cam].publish(msg)
              
    def landmark_timer_callback(self):
        '''
        Callback to publish 3D landmarks.
        '''
        # make a local copy of data
        body_landmarks = {
            'left': None,
            'right': None
        }

        for cam in ['left', 'right']:
            with self._lock_out[cam]:
                if self._flag_out[cam].value:
                    self._flag_out[cam].value = False # set to false to consume
                    np.copyto(self._local_img_out[cam], self._shmem_img_out[cam].as_numpy())
                    body_landmarks[cam] = self._shmem_landmarks_out[cam]['body'].as_numpy()
                else:
                    # no changes, terminate
                    return

        # perform triangulation
        try:
            body_landmarks_3d = triangulate(self.P_left,
                                            self.P_right,
                                            body_landmarks['left']*self.stereo_size,
                                            body_landmarks['right']*self.stereo_size)
        except np.linalg.LinAlgError as ex:
            self.get_logger().log('Triangulation failed. {}'.format(ex), LoggingSeverity.WARN)
            return

        # pack into message
        msg = BodyLandmarksStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.param_output_frame.value
        
        for i, row in enumerate(body_landmarks_3d):
            landmark = Point()
            landmark.x = row[0]
            landmark.y = row[1]
            landmark.z = row[2]
            msg.landmarks.append(landmark)

        # publish
        self.pub_body_landmarks.publish(msg)

def triangulate(P1, P2, points1, points2):
    '''
    Perform triangulation using direct linear transform.

    Parameters
    ==========
    P1 : 4x4 ArrayLike
        Projection matrix of camera 1.
    P2 : 4x4 ArrayLike
        Projection matrix of camera 2.
    points1 : nx2 ArrayLike
        2D points in camera 1.
    points2: nx2 ArrayLike
        2D points in camera 2.

    Returns
    =======
    points3D : nx3 ArrayLike
        Triangulated 3D points.
    '''
    A = np.array([[point1[1]*P1[2,:] - P1[1,:],
                    P1[0,:] - point1[0]*P1[2,:],
                    point2[1]*P2[2,:] - P2[1,:],
                    P2[0,:] - point2[0]*P2[2,:]]
                    for point1, point2 in zip(points1, points2)])

    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    return Vh[:, 3, 0:3]/Vh[:, 3, 3, np.newaxis]


def main(args=None):
    rclpy.init(args=args)

    node = PoseEstimatorNode(node_name='pose_estimator')

    rclpy.spin(node)
    rclpy.shutdown()