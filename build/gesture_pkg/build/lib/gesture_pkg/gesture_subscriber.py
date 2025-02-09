#!/usr/bin/env python3

from typing import Tuple
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import String

class GestureSubscriberNode(Node):
    def __init__(self):
        super().__init__("gesture_sub")
        self.sub = self.create_subscription(String, "gesture_direction", self.subscriber_callback, 10)

    def subscriber_callback(self, msg):

        print("receive direction........." + str(msg.data))

def main(args=None):
    rclpy.init()
    gesture_sub = GestureSubscriberNode()
    print("Waiting for data to be published..........")

    try:
        rclpy.spin(gesture_sub)
    except KeyboardInterrupt():
        gesture_sub.destroy_node()

if __name__ == '__main__':
    main()
