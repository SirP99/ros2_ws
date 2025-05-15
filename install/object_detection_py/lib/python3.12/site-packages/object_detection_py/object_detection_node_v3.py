#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.subscription = self.create_subscription(
             CompressedImage,
            'image_raw/compressed',  # Módosítsd a megfelelő topicra
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # Használj megfelelő modellt
        self.get_logger().info('Object Detection Node has been started.')

    def listener_callback(self, msg):
        try:
            # Konvertálás ROS Image üzenetből OpenCV képpé
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Kép konvertálási hiba: {e}')
            return

        # Objektumdetekció
        results = self.model(cv_image)

        # Eredmények megjelenítése
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{self.model.names[cls]} {conf:.2f}'
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Kép megjelenítése
        cv2.imshow('Object Detection', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
