import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame

class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)  # 이벤트 주기를 0.1초로 빠르게 설정
        self.linear_x = 0.0
        self.steering_angle = 0.39

        pygame.init()
        pygame.display.set_mode((300, 300))

    def publish_cmd_vel(self):
        # pygame 이벤트를 빠르게 처리
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.linear_x = min(self.linear_x + 0.5, 2.5)
                elif event.key == pygame.K_s:
                    self.linear_x = max(self.linear_x - 0.5, 0.0)
                elif event.key == pygame.K_a:
                    self.steering_angle = min(self.steering_angle + 0.2, 0.4)  # 각속도 반영을 빠르게 하기 위해 증가폭을 더 크게 설정
                elif event.key == pygame.K_d:
                    self.steering_angle = max(self.steering_angle - 0.2, -0.4)

        # Twist 메시지를 발행
        msg = Twist()
        msg.linear.x = self.linear_x
        msg.angular.z = self.steering_angle
        self.publisher.publish(msg)

        self.get_logger().info(f'Published: Linear X = {msg.linear.x}, Steering Angle (angular.z) = {msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()

if __name__ == '__main__':
    main()
