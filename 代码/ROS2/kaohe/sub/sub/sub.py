import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class sub(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("我是%s,订阅开始。" % name)
        
        self.msg = ""
        self.bridge = CvBridge()

        self.sub_str = self.create_subscription(String,"string",self.sub_str_callback,10)
        self.sub_mat = self.create_subscription(Image,"mat",self.sub_mat_callback,10)

    def sub_str_callback(self,msg):
        self.msg = msg.data
        self.get_logger().info("%s" % self.msg)
        self.get_logger().info(f"数据类型名: {type(msg.data).__name__}")
    def sub_mat_callback(self,img):
        self.get_logger().info("订阅到原数据类型为Mat的数据。")
        cv_img = self.bridge.imgmsg_to_cv2(img,"bgr8")
        cv2.imshow("img",cv_img)
        cv2.waitKey(0)


def main(args=None):
    rclpy.init(args=args)
    node = sub("lcg1")
    rclpy.spin(node)
    rclpy.shutdown()
