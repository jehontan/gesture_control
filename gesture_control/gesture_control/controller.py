import rclpy
from rclpy.logging import LoggingSeverity
import numpy as np
import ros2_numpy as ros2np

from pose_estimator.pose_estimation import BodyLandmarks
from gesture_control_interfaces.msg import BodyLandmarksStamped, HandLandmarksStamped
from geometry_msgs.msg import PoseStamped, Point

import tf2_ros
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from .utils import SimpleMovingAverage, euclidean_distance
from .knn import KNNClassifier
from .embedder import embed_hand_pose
import sys

class PointToNavigateController(rclpy.node.Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup attributes
        self._pointing = False
        self._goal_point = SimpleMovingAverage(10)

        # setup params
        self.param_fixed_frame = self.declare_parameter('fixed_frame_id', value='odom')
        self.param_hand_pose_dataset = self.declare_parameter('hand_pose_dataset', value='~/hand_pose_dataset.csv')
        self.param_hand_pose_labels = self.declare_parameter('hand_pose_labels', value='~/hand_pose_labels.txt')

        # setup subscribers
        self.sub_pose = self.create_subscription(
            BodyLandmarksStamped,
            'body_landmarks',
            self.body_landmarks_callback,
            rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value
        )

        self.sub_hands = self.create_subscription(
            HandLandmarksStamped,
            'hand_landmarks',
            self.hand_landmarks_callback,
            rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value
        )

        # setup publishers
        self.pub_goal = self.create_publisher(
            PoseStamped,
            'goal_pose',
            rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value
        )

        # setup tf
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # load hand pose dataset
        self.init_knn()

    def init_knn(self):
        '''
        Initialize KNN classifier for hand poses.
        '''
        try:
            # load labels
            self.pose_labels = []
            with open(self.param_hand_pose_labels.value, 'r') as f:
                for label in f:
                    self.pose_labels.append(label.strip())

            # load dataset
            dataset = np.genfromtxt(self.param_hand_pose_dataset.value, delimiter=',')
            Y_train = dataset[:,0]
            X_train = dataset[:,1:]

            self.pose_knn = KNNClassifier(X_train, Y_train, 5)
            self.get_logger().log('Hand pose classifier initialized.', LoggingSeverity.INFO)
        except Exception as e:
            self.get_logger().log(e, LoggingSeverity.FATAL)
            sys.exit('Could not initialize pose classifier.')

    def body_landmarks_callback(self, msg) -> None:
        '''
        Get pointing direction from 3D body landmarks.

        Pointing direction is calulated from wrist to root of index finger.

        Parameters
        ==========
        msg : gesture_control_interfaces.msg.BodyPose3D
            Body pose message. Landmark points in meters, relative to robot coordinate system.
        '''
        
        if self._pointing:
            target_frame = self.param_fixed_frame.value
            source_frame = msg.header.frame_id

            # get landmarks of interest as numpy homogeneous points
            lm1 = ros2np.numpify(msg.landmarks[BodyLandmarks.RIGHT_WRIST], hom=True)
            lm2 = ros2np.numpify(msg.landmarks[BodyLandmarks.RIGHT_INDEX], hom=True)

            # transform to odom frame
            try:
                tf = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except TransformException:
                self.get_logger().log('Could not transform pose.', LoggingSeverity.WARN)
                return

            tf = ros2np.numpify(tf) # 4x4 homogeneous tranformation matrix
            lm1 = tf@lm1
            lm2 = tf@lm2
            
            # compute direction, cast from wrist to z = 0 (in odom frame)
            v = lm2[:3] - lm1[:3] # direction vector
            
            # discard if operator is pointing to high, leading to points far away
            if np.arctan(v[1]/v[2]) >= np.deg2rad(60):
                self.get_logger().log('Operator is pointing too high!', LoggingSeverity.WARN)
                return

            l = -lm1[2]/v[2] # length of ray cast
            point = l*v + lm1[:3] # point to navigate to on z = 0

            ave = self._goal_point.update(point)

            # compare point to average, point should be stable within 0.3 m
            d = euclidean_distance(point, ave)
            if self._goal_point.is_full() and d < 0.3:
                self.get_logger().log('Pointed goal: ({}, {}, {})'.format(*ave), LoggingSeverity.INFO)

                # create pose
                goal = PoseStamped()
                goal.header.stamp = self.get_clock().now().to_msg()
                goal.header.frame_id = self.param_fixed_frame.value # change the reference frame
                
                goal.pose.point = ros2np.msgify(Point, ave)

                # publish pose
                self.pub_goal.publish(goal)

                # reset SMA filter
                self._goal_point.clear()

    def hand_landmarks_callback(self, msg):
        '''
        Detect hand gestures.
        '''
        landmarks = np.empty((21, 3))
        for i, landmark in enumerate(msg.landmarks):
            landmarks[i,:] = ros2np.numpify(landmark)
        
        # classify the pose
        embedding = embed_hand_pose(landmarks)
        pose = self.pose_knn.predict(embedding)
        
        self._pointing = self.pose_labels[pose] == 'pointing'

        if self._pointing:
            self.get_logger().log('Pointing detected.', LoggingSeverity.INFO)

def main(args=None):
    rclpy.init(args=args)
    
    node = PointToNavigateController(node_name='gesture_controller')

    rclpy.spin(node)
    rclpy.shutdown()