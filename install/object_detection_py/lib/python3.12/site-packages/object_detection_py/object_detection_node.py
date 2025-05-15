import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

import cv2
import torch
import numpy as np
import threading
import time
import os

from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # YOLOv8 modell betöltése (automatikusan letölti, ha szükséges)
        self.model = YOLO('yolov8n.pt')  # Használhatsz 'yolov8s.pt', 'yolov8m.pt' stb. modelleket is

        # CvBridge inicializálása
        self.bridge = CvBridge()

        # Kép előfizetés
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',  # Cseréld le a megfelelő képtémára
            self.image_callback,
            10
        )

        # Legfrissebb kép tárolása
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Megjelenítő szál indítása
        self.running = True
        self.display_thread = threading.Thread(target=self.display_image)
        self.display_thread.start()

    def image_callback(self, msg):
        # ROS Image üzenet konvertálása OpenCV formátumra
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f'Kép konvertálási hiba: {e}')

    def display_image(self):
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 800, 600)

        while rclpy.ok() and self.running:
            frame = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    self.latest_frame = None

            if frame is not None:
                # Objektumdetektálás
                results = self.model.predict(frame, imgsz=640, conf=0.5)

                # Eredmények megjelenítése
                annotated_frame = results[0].plot()
                cv2.imshow("Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.display_thread.join()

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

if __name__ == '__main__':
    main()
