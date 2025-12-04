# environment.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import numpy as np
from typing import Tuple
import math

class TurtleBot3Env(Node):

    def __init__(self):
        super().__init__('turtlebot3_env')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.reset_world_client = self.create_client(Empty, '/reset_world')


        self.scan_data = None
        self.position = (0.0, 0.0)
        self.yaw = 0.0
        self.goal_position = (1.5, -1.5)

        self.actions = {
            0: (0.0,  0.0),    
            

            1: (0.05,  0.8),    # left Curva
            2: (0.05, -0.8),    # Right Curva 
            
            # Cspeed
            3: (-0.1, 0.0),   
            4: (0.1,  0.0),
        }

        self.collision_threshold = 0.15
        self.goal_tolerance = 0.20

        self.v_ref_base = 10    
        self.k_heading  = 0.40  


    # ---------- Callback ----------
    def scan_callback(self, msg: LaserScan):
        self.scan_data = list(msg.ranges)

    def odom_callback(self, msg: Odometry):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)



    def set_goal(self, goal: Tuple[float, float]):
        self.goal_position = (float(goal[0]), float(goal[1]))

    def step(self, action: int):

        rx, ry = self.position
        gx, gy = self.goal_position
        
        goal_heading = math.atan2(gy - ry, gx - rx)
        heading_error = goal_heading - self.yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        v_ref = self.v_ref_base
        w_ref = self.k_heading * heading_error


        dv, dw = self.actions[action]
        v_cmd = v_ref + dv
        w_cmd = w_ref + dw

        if self.scan_data is not None:
            scan = np.array(self.scan_data, dtype=np.float32)
            scan[~np.isfinite(scan)] = 3.5
            scan[scan <= 0.01] = 3.5
            
            n = len(scan)
            idx_limit = n // 6
            front_sector = np.concatenate((scan[-idx_limit:], scan[:idx_limit]))
            min_front = float(np.min(front_sector))
        else:
            min_front = 3.5

        hard_stop_dist = 0.25   
        slow_down_dist = 0.85 
        
        speed_factor = 1.0

        if min_front < hard_stop_dist:
            v_cmd = 0.0
            w_cmd = 1.5 
        elif min_front < slow_down_dist:

            speed_factor = (min_front - hard_stop_dist) / (slow_down_dist - hard_stop_dist)
            
            v_cmd = v_cmd * speed_factor
            
            if min_front < 0.50 and abs(w_cmd) < 0.5:
                w_cmd *= 1.5 

        v_cmd = max(0.0, min(v_cmd, 0.22))
        w_cmd = max(-2.5, min(w_cmd, 2.5))

        self.send_velocity(v_cmd, w_cmd)
        rclpy.spin_once(self, timeout_sec=0.05)



        done = False
        reward = 0.0

        if self.is_collision():
            reward = -100.0
            done = True
            self.send_velocity(0.0, 0.0)
            
        elif self.is_goal_reached():
            reward = 200.0
            done = True
            self.send_velocity(0.0, 0.0)
            
        else:
            reward = -0.05

        return self.get_state(), reward, done

    def is_collision(self) -> bool:
        if self.scan_data is None: return False
        scan = np.array(self.scan_data, dtype=np.float32)
        scan[~np.isfinite(scan)] = 3.5
        scan[scan <= 0.01] = 3.5
        return float(scan.min()) < self.collision_threshold

    def is_goal_reached(self) -> bool:
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        return (abs(dx) <= self.goal_tolerance and abs(dy) <= self.goal_tolerance)

    def distance_to_goal(self) -> float:
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        return math.sqrt(dx**2 + dy**2)

    def send_velocity(self, linear: float, angular: float):
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(twist)

    def reset(self):
        self.send_velocity(0.0, 0.0)
        self.reset_world()
        rclpy.spin_once(self, timeout_sec=0.5)

    def reset_world(self):
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Reset world service not available')
            return
        request = Empty.Request()
        future = self.reset_world_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def get_state(self):
        return {
            'scan_data': self.scan_data,
            'position': self.position,
            'goal_position': self.goal_position,
            'yaw': self.yaw
        }