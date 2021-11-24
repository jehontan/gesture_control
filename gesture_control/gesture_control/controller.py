import rclpy
import numpy as np
import ros2_numpy as ros2np

from pose_estimator.pose_estimation import BodyLandmarks
from gesture_control_interfaces.msg import BodyPoseStamped
from geometry_msgs.msg import PoseStamped, Point

import tf2_ros
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

class PointToNavigateController(rclpy.node.Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup params
        self.param_fixed_frame = self.declare_parameter('fixed_frame_id', value='odom')

        # setup subscribers
        self.sub_pose = self.create_subscription(
            BodyPoseStamped,
            'body_pose',
            self.pose_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
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

    def pose_callback(self, msg) -> None:
        '''
        Interpret pose as a point.

        Pointing direction is calulated from wrist to root of index finger.

        Parameters
        ==========
        msg : gesture_control_interfaces.msg.BodyPose3D
            Body pose message. Landmark points in meters, relative to robot coordinate system.
        '''
        
        target_frame = self.param_fixed_frame.value
        source_frame = msg.header.frame_id

        # get landmarks of interest as numpy homogeneous points
        lm1 = ros2np.numpify(msg.landmarks[BodyLandmarks.RIGHT_WRIST], hom=True)
        lm2 = ros2np.numpify(msg.landmarks[BodyLandmarks.RIGHT_INDEX], hom=True)

        # transform to odom frame
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except TransformException:
            self.get_logger().warning('Could not transform pose.')
            return

        tf = ros2np.numpify(tf) # 4x4 homogeneous tranformation matrix
        lm1 = tf@lm1
        lm2 = tf@lm2
        
        # compute direction, cast from wrist to z = 0 (in odom frame)
        v = lm2[:3] - lm1[:3] # direction vector
        
        # discard if operator is pointing to high, leading to points far away
        if np.arctan(v[1]/v[2]) >= np.deg2rad(60):
            self.get_logger().warning('Operator is pointing too high!')
            return

        l = -lm1[2]/v[2] # length of ray cast
        goal_point = l*v + lm1[:3] # point to navigate to on z = 0

        # create pose
        goal = PoseStamped()
        goal.header.stamp = msg.header.stamp # make the timestamp the same
        goal.header.frame_id = target_frame # change the reference frame
        
        goal.pose.point = ros2np.msgify(Point, goal_point)

        # publish pose
        self.pub_goal.publish(goal)

def main(args=None):
    rclpy.init(args=args)
    
    node = PointToNavigateController(node_name='gesture_controller')

    rclpy.spin(node)
    rclpy.shutdown()