import signal
import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 15 * 20, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(self.flatten(self.features(x)))

class BasicDriveNode(Node):
    def __init__(self):
        super().__init__('basic_drive_node')
        self.bridge = CvBridge()

        self.device = torch.device('cpu')
        self.model = BasicNet().to(self.device)
        model_path = '/home/ubuntu/basic_drive_model.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor()
        ])

        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.linear_speed = 0.18

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_angular_z = self.model(input_tensor).item()

            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.angular.z = float(pred_angular_z)
            self.pub.publish(twist)
        except Exception:
            pass

    def stop(self):
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        for _ in range(10):
            self.pub.publish(stop_twist)
            time.sleep(0.02)

def main(args=None):
    rclpy.init(args=args)
    node = BasicDriveNode()

    def sigint_handler(sig, frame):
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        rclpy.spin(node)
    except SystemExit:
        pass

if __name__ == '__main__':
    main()
