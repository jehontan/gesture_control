import rclpy
from rclpy.node import Node
from typing import List
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from gesture_control_interfaces.msg import BodyLandmarksStamped, HandLandmarksStamped

class PoseVisualizerNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sub = self.create_subscription(
            BodyLandmarksStamped,
            'body_landmarks',
            self.landmark_callback,
            rclpy.qos.QoSPresetProfiles.SYSTEM_DEFAULT.value
        )

        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        plt.ion()
        plt.show()

    def landmark_callback(self, msg):
        points = np.array([(l.x, l.y, l.z) for l in msg.landmarks])
        self.ax.cla()
        self.ax.scatter(points[:,0], points[:,1], points[:,2])
        self.ax.set_xlim(-2,2)
        self.ax.set_ylim(-2,2)
        self.ax.set_zlim(0,4)
        plt.draw()
        plt.pause(0.01)


def main(args=None):
    rclpy.init(args=args)

    node = PoseVisualizerNode(node_name='pose_visualizer')

    rclpy.spin(node)
    rclpy.shutdown()