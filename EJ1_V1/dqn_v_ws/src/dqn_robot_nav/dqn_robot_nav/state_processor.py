# state_processor.py
import numpy as np
from collections import deque
from typing import Tuple

class StateProcessor:

    
    def __init__(self, n_lidar_bins: int = 20, n_stack: int = 3):

        self.n_lidar_bins = n_lidar_bins
        self.n_stack = n_stack
        self.max_lidar_range = 3.5  # Max
        
        self.state_buffer = deque(maxlen=n_stack)
        
    def reset(self):
        """Limpiar la memoria visual (Llamar al inicio de cada episodio)"""
        self.state_buffer.clear()

    def process_lidar(self, scan_data: list) -> np.ndarray:
        """
        Procesa el LiDAR con visión de 270 GRADOS.
        Cobertura: -135° (atrás-der) a +135° (atrás-izq).
        """
        if scan_data is None:
            return np.ones(self.n_lidar_bins, dtype=np.float32)

        scan_array = np.array(scan_data, dtype=np.float32)
        
        scan_array[np.isinf(scan_array)] = self.max_lidar_range
        scan_array[np.isnan(scan_array)] = self.max_lidar_range
        scan_array = np.clip(scan_array, 0, self.max_lidar_range)
        
        # --- LÓGICA DE RECORTE (270°) ---
        limit_angle = 135  
        n_points = len(scan_array)
        
        idx_left = int(n_points * (limit_angle / 360.0))
        idx_right = int(n_points * ((360 - limit_angle) / 360.0))
        
        # (0 a +135°)
        left_chunk = scan_array[0 : idx_left]
        # (-135° a 360°)
        right_chunk = scan_array[idx_right : ]
        
        # 
        wide_scan = np.concatenate((right_chunk, left_chunk))
        
        # --- BINNINg
        points_per_bin = len(wide_scan) // self.n_lidar_bins
        binned_scan = []
        
        for i in range(self.n_lidar_bins):
            start = i * points_per_bin
            end = (i + 1) * points_per_bin if i < self.n_lidar_bins - 1 else len(wide_scan)
            
            chunk = wide_scan[start:end]
            
            if len(chunk) > 0:
                val = np.min(chunk)
            else:
                val = self.max_lidar_range
            binned_scan.append(val)
        
        return np.array(binned_scan) / self.max_lidar_range
    
    def compute_goal_info(self, 
                         current_pos: Tuple[float, float],
                         goal_pos: Tuple[float, float],
                         current_yaw: float) -> np.ndarray:
        """
        Calcular distancia y ángulo relativo al goal normalizados.
        """
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        
        dist = np.sqrt(dx**2 + dy**2)
        
        angle_to_goal = np.arctan2(dy, dx)
        relative_angle = angle_to_goal - current_yaw
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
        
        dist_norm = np.clip(dist / 5.0, 0, 1)  # Asumimos max 5 metros
        angle_norm = relative_angle / np.pi    # [-1, 1]
        
        return np.array([dist_norm, angle_norm], dtype=np.float32)
    
    def get_state(self,
                  scan_data: list,
                  current_pos: Tuple[float, float],
                  goal_pos: Tuple[float, float],
                  current_yaw: float) -> np.ndarray:

        lidar_state = self.process_lidar(scan_data)
        goal_state = self.compute_goal_info(current_pos, goal_pos, current_yaw)
        
        current_frame = np.concatenate([lidar_state, goal_state])
        
        if len(self.state_buffer) == 0:
            for _ in range(self.n_stack):
                self.state_buffer.append(current_frame)
        else:
            self.state_buffer.append(current_frame)
            
        # a signgle vector: 
        # [Frame_t-2, Frame_t-1, Frame_t]
        stacked_state = np.concatenate(self.state_buffer)
        
        return stacked_state.astype(np.float32)