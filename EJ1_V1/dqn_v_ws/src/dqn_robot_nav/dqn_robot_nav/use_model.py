import rclpy
from rclpy.node import Node
import pickle
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import os

def preprocess_scan(ranges, expected_len=360, max_dist=10.0, normalize=False):
    arr = np.array(ranges, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=max_dist, posinf=max_dist, neginf=0.0)
    if arr.size != expected_len:
        x_old = np.linspace(0, 1, arr.size)
        x_new = np.linspace(0, 1, expected_len)
        arr = np.interp(x_new, x_old, arr)
    if normalize:
        arr = arr / max_dist
    return arr

class DQNNavigator(Node):
    def __init__(self):
        super().__init__('dqn_navigator')

        # Ruta al modelo - CAMBIA si es necesario
        model_path = 'model.pkl'
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model file not found: {model_path}')
            raise FileNotFoundError(model_path)
        self.get_logger().info(f'Loading model from: {model_path}')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Infer expected input length (opcional)
        self.expected_input = getattr(self.model, 'n_features_in_', 360)
        # Some sklearn models carry n_outputs_ and n_features_in_
        try:
            self.get_logger().info(f'Model expects {self.expected_input} features.')
        except Exception:
            pass

        # Configure normalization flag: set True si al entrenar normalizaste por max_dist
        self.normalize_input = False

        # Publisher / Subscriber
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info('DQN Navigator node initialized.')

    def scan_callback(self, scan_msg):
        state = preprocess_scan(scan_msg.ranges, expected_len=self.expected_input, normalize=self.normalize_input)

        # wrap into 2D batch shape [1, n_features] for sklearn-like models
        X = np.expand_dims(state, axis=0)

        try:
            # If model is sklearn-like with predict returning (1, n_actions)
            q_values = self.model.predict(X)
            # Some models return (n_actions,) or (1, n_actions)
            q_values = np.array(q_values).squeeze()
        except Exception as e:
            self.get_logger().error(f'Predict failed: {e}')
            return

        action = int(np.argmax(q_values))
        twist = Twist()
        # Map actions -> velocities (ajusta magnitudes si lo necesitas)
        if action == 0:
            twist.linear.x = 0.15
            twist.angular.z = 0.0
        elif action == 1:
            twist.linear.x = 0.0
            twist.angular.z = 0.6
        elif action == 2:
            twist.linear.x = 0.0
            twist.angular.z = -0.6
        else:
            # Si tu modelo tiene más acciones, maneja aquí
            twist.linear.x = 0.0

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = DQNNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
