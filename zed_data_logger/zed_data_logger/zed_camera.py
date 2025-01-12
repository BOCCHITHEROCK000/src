import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class PersonDetectionNode(Node):
    def __init__(self):
        super().__init__('person_detection_node')

        # YOLOv8 모델 로드 (객체 탐지용)
        self.model = YOLO('yolov8n.pt')  # YOLOv8 객체 탐지 모델 파일 경로

        self.bridge = CvBridge()

        # 구독자 설정
        self.create_subscription(
            Image,
            '/zed/zed_node/left/image_rect_color',
            self.image_callback,
            10
        )

        # 카메라 화면을 한 번만 띄우기 위해 초기화
        self.window_name = 'YOLOv8 Person Detection'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # 윈도우 크기 조정 가능하도록 설정

    def image_callback(self, msg):
        # ROS 이미지 메시지를 OpenCV 형식으로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLOv8을 사용하여 이미지에서 사람을 감지
        results = self.model(cv_image)

        # 결과에서 사람 클래스만 필터링 (class_id == 0 -> 사람)
        person_class_id = 0  # YOLO 모델에서 'person' 클래스의 ID는 0입니다
        for result in results:
            boxes = result.boxes  # 예측된 바운딩 박스
            confidences = boxes.conf.cpu().numpy()  # 신뢰도
            class_ids = boxes.cls.cpu().numpy()  # 클래스 ID

            # 각 객체에 대해 바운딩 박스를 그리기
            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = box  # 바운딩 박스 좌표
                confidence = confidences[i]
                class_id = int(class_ids[i])

                if class_id == person_class_id:  # 사람이 감지된 경우
                    # 바운딩 박스 그리기
                    color = (255, 0, 0)  # 초록색 박스
                    cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # 신뢰도 표시
                    label = f'{confidence * 100:.2f}%'
                    font_scale = 0.6
                    font_thickness = 2
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    label_x = int(x1)
                    label_y = int(y1) - 10
                    cv2.rectangle(cv_image, (label_x, label_y - 10), (label_x + label_size[0], label_y - label_size[1] - 10), color, -1)
                    cv2.putText(cv_image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # 결과 이미지 표시 (바운딩 박스 포함된 이미지)
        cv2.imshow(self.window_name, cv_image)
        cv2.waitKey(1)  # 화면 갱신을 위한 간격 설정

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
