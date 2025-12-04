#!/usr/bin/env python3
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
    """ROS2 Environment wrapper for TurtleBot3 navigation"""
    
    def __init__(self):
        super().__init__('turtlebot3_env')
        
        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        
        # Gazebo service for resetting world
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        
        # State variables
        self.scan_data = None
        self.position = (0.0, 0.0)
        self.yaw = 0.0
        self.last_position = (0.0, 0.0)
        
        # Posición inicial (se llenará con la primera odom)
        self.start_position = None
        
        # Goal position (la fija el training con set_goal)
        self.goal_position = (-0.5, -1.5)

        # Action space: 5 discrete actions
        self.actions = {
            0: (0.22, 0.0),    # Adelante
            1: (0.0,  1.0),    # Girar izquierda en el sitio
            2: (0.0, -1.0),    # Girar derecha en el sitio
            3: (0.18,  0.5),   # Adelante + curva suave izquierda
            4: (0.18, -0.5),   # Adelante + curva suave derecha
        }




        self.collision_threshold = 0.15  # meters
        
        # Tolerancia para considerar llegada por eje
        self.goal_tolerance = 0.2       # metros por eje
        
    # ---------- Callbacks ----------
    
    def scan_callback(self, msg: LaserScan):
        """Store latest LiDAR scan"""
        self.scan_data = list(msg.ranges)
    
    def odom_callback(self, msg: Odometry):
        """Store latest odometry data"""
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        
        # Guardar primera posición como start_position
        if self.start_position is None:
            self.start_position = self.position
        
        # Extraer yaw
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + 
                         orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + 
                             orientation_q.z * orientation_q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
    
    # ---------- API principal ----------
    
    def set_goal(self, goal: Tuple[float, float]):
        """Fijar explícitamente la meta desde el training node"""
        self.goal_position = (float(goal[0]), float(goal[1]))
        self.get_logger().info(
            f"[ENV] Goal set to {self.goal_position}"
        )
    
    def step(self, action: int):
        """
        Execute action and return next state, reward, done
        
        Args:
            action: Action index
            
        Returns:
            next_state, reward, done
        """
        # Ejecutar acción
        linear_vel, angular_vel = self.actions[action]
        self.send_velocity(linear_vel, angular_vel)
        
        # Esperar actualización de estado
        rclpy.spin_once(self, timeout_sec=0.1)
        
        # Chequear condiciones de término
        done = False
        reward = 0.0
        
        # 1. Colisión
        if self.is_collision():
            reward = -100.0
            done = True
            self.get_logger().info("Collision detected!")
            
        # 2. Meta alcanzada
        elif self.is_goal_reached():
            reward = 200.0
            done = True
            self.get_logger().info(
                f"Goal reached! pos={self.position}, goal={self.goal_position}"
            )
            
        # 3. Recompensa continua
        else:
            reward = self.compute_reward(action)
        
        return self.get_state(), reward, done
    
    def compute_reward(self, action: int) -> float:
        """
        Reward shaping sencillo y estable:
        - Progreso hacia la meta (principal)
        - Penalización por proximidad a obstáculos
        - Penalización suave por giros puros
        - Penalización por tiempo (episodios demasiado largos)
        - Bonus por cercanía al objetivo
        - Bonus por alineación al objetivo
        - Penalización si se queda estancado (no progresa)
        """
        # ===== 1) DISTANCIA ACTUAL =====
        current_dist = self.distance_to_goal()

        # ===== 1a) PROGRESO LINEAL HACIA LA META (TERMINO PRINCIPAL) =====
        if hasattr(self, 'last_distance') and self.last_distance is not None:
            progress = self.last_distance - current_dist   # >0 si se acerca
            progress_reward = progress * 25.0              # peso fuerte pero razonable
        else:
            progress_reward = 0.0

        # Guardamos distancia para el siguiente paso
        self.last_distance = current_dist

        # ===== 1b) PENALIZACIÓN POR ESTANCAMIENTO =====
        # Si el cambio en distancia es casi cero, penalizamos un poco
        stagnation_penalty = 0.0
        if abs(progress) < 0.005:      # muy poco cambio de distancia
            stagnation_penalty = -0.05 # lo suficientemente pequeño pero constante

        # ===== 2) PROXIMIDAD A OBSTÁCULOS =====
        if self.scan_data:
            scan = np.array(self.scan_data, dtype=np.float32)
            scan[~np.isfinite(scan)] = 3.5
            scan[scan <= 0.01] = 3.5
            min_obstacle_dist = float(scan.min())
        else:
            min_obstacle_dist = 3.5
        
        if min_obstacle_dist < 0.3:
            danger_zone = 0.3
            obstacle_penalty = -2.0 * (danger_zone - min_obstacle_dist)
        else:
            obstacle_penalty = 0.0
        
        # ===== 3) PENALIZACIÓN POR ROTACIÓN PURA =====
        # Penalizamos SOLO giros puros (acciones 1 y 2), no las curvas con avance
        action_penalty = -0.3 if action in [1, 2] else 0.0

        # ===== 4) PENALIZACIÓN POR TIEMPO =====
        time_penalty = -0.01   # un poco más suave que antes

        # ===== 5) BONUS POR CERCANÍA AL OBJETIVO =====
        close_bonus = 0.0
        close_radius = 0.5          # radio donde empieza a recibir bonus
        if current_dist < close_radius:
            k_close = 5.0           # fuerza del bonus de cercanía
            close_bonus = (close_radius - current_dist) * k_close

        # ===== 6) BONUS POR ALINEACIÓN AL OBJETIVO =====
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        angle_to_goal = math.atan2(dy, dx)
        relative_angle = angle_to_goal - self.yaw

        # Normalizar a [-pi, pi]
        relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
        angle_error = abs(relative_angle)  # 0 = perfecto, pi = totalmente al revés

        alignment_score = 1.0 - (angle_error / math.pi)  # 1 bien, 0 mal
        alignment_score = max(0.0, alignment_score)

        # Ahora SIN escalar por distancia (para no crear pozos raros lejos)
        k_align = 2.0
        alignment_bonus = alignment_score * k_align

        # ===== 7) SUMA TOTAL =====
        total_reward = (
            progress_reward
            + stagnation_penalty
            + obstacle_penalty
            + action_penalty
            + time_penalty
            + close_bonus
            + alignment_bonus
        )

        return float(total_reward)


    def is_collision(self) -> bool:
        """Check if robot has collided with obstacle usando LiDAR limpio"""
        if self.scan_data is None:
            return False
        
        scan = np.array(self.scan_data, dtype=np.float32)
        scan[~np.isfinite(scan)] = 3.5
        scan[scan <= 0.01] = 3.5
        
        min_distance = float(scan.min())
        return min_distance < self.collision_threshold
    
    def is_goal_reached(self) -> bool:
        """Check if robot has reached goal con tolerancia por eje"""
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        
        return (abs(dx) <= self.goal_tolerance and
                abs(dy) <= self.goal_tolerance)
    
    def distance_to_goal(self) -> float:
        """Compute Euclidean distance to goal"""
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        return math.sqrt(dx**2 + dy**2)
    
    def send_velocity(self, linear: float, angular: float):
        """Send velocity command to robot"""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)
    
    def reset(self, random_goal: bool = True) -> None:
        """
        Reset environment for new episode.
        
        NOTA: aquí ya NO se cambia la meta.
        La meta la define el training con set_goal().
        """
        # Parar robot
        self.send_velocity(0.0, 0.0)
        
        # Resetear mundo en Gazebo
        self.reset_world()
        
        # Esperar actualización de estado
        rclpy.spin_once(self, timeout_sec=0.5)
        
        # Actualizar distancia inicial a la meta vigente
        self.last_distance = self.distance_to_goal()
    
    def reset_world(self):
        """Reset Gazebo world using /reset_world service"""
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Reset world service not available')
            return
        
        request = Empty.Request()
        future = self.reset_world_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None:
            self.get_logger().info('World reset successfully')
        else:
            self.get_logger().error('Failed to reset world')
    
    def get_state(self):
        """Return raw components for the StateProcessor to consume"""
        return {
            'scan_data': self.scan_data,
            'position': self.position,
            'goal_position': self.goal_position,
            'yaw': self.yaw
        }
