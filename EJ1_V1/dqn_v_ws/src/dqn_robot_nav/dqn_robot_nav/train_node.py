#!/usr/bin/env python3
# train_node.py

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from dqn_robot_nav.dqn_agent import DQNAgent
from dqn_robot_nav.environment import TurtleBot3Env
from dqn_robot_nav.state_processor import StateProcessor

from datetime import datetime
import os
import math
import random

class DQNTrainingNode(Node):
    """
    DQN training node for TurtleBot3 navigation.
    ESTRATEGIA: FLUID AVOIDANCE + LOGGING MEJORADO
    """

    def __init__(self):
        super().__init__('dqn_training_node')

        self.obstacle_centers = [
            (1.0,  1.0), (1.0,  0.0), (1.0, -1.0),
            (0.0,  1.0), (0.0,  0.0), (0.0, -1.0),
            (-1.0,  1.0), (-1.0,  0.0), (-1.0, -1.0),
        ]
        self.forbidden_radius = 0.45 

        self.n_episodes = 250     
        self.max_steps_per_episode = 1000
        self.action_size = 5      

        self.env = TurtleBot3Env()
        
        self.n_lidar_bins = 20  
        self.n_stack = 3        
        self.state_size = (self.n_lidar_bins + 2) * self.n_stack 
        
        self.state_processor = StateProcessor(
            n_lidar_bins=self.n_lidar_bins, 
            n_stack=self.n_stack
        )
        
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001,    
            gamma=0.95,             
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.996,    
            memory_size=5000,
            batch_size=64
        )

        # ---------- Logging  ----------
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_mean_loss = []
        self.success_count = 0
        self.collision_count = 0
        
        self.last_action = 0  #<---

        self.k_progress = 0.3     
        self.k_perp = 0.05        

        self.current_goal = None
        self.successes_for_current_goal = 0
        self.successes_to_change = 2  

        self.results_dir = f"results_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)


    def _is_valid_goal(self, goal):
        gx, gy = goal
        for (ox, oy) in self.obstacle_centers:
            dx = gx - ox
            dy = gy - oy
            if math.hypot(dx, dy) < self.forbidden_radius:
                return False
        return True

    def sample_random_goal_in_circle(self, center=(-0.5, 0.5), radius=0.5, max_tries=100):
        cx, cy = center
        for _ in range(max_tries):
            r = radius * math.sqrt(random.random())
            theta = random.uniform(0.0, 2.0 * math.pi)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            candidate = (x, y)
            if self._is_valid_goal(candidate):
                return candidate
        return center

    def _compute_line_metrics(self, p_start, p_goal, p_robot):
        sx, sy = p_start; gx, gy = p_goal; rx, ry = p_robot
        vx = gx - sx; vy = gy - sy
        wx = rx - sx; wy = ry - sy
        line_len = math.sqrt(vx*vx + vy*vy) + 1e-6
        proj = (wx*vx + wy*vy) / line_len
        cross = abs(wx*vy - wy*vx)
        d_perp = cross / line_len
        return proj, d_perp

    def _compute_heading_alignment(self, yaw, robot_pos, goal_pos):
        rx, ry = robot_pos; gx, gy = goal_pos
        hx = math.cos(yaw); hy = math.sin(yaw)
        dx = gx - rx; dy = gy - ry
        dist = math.hypot(dx, dy) + 1e-6
        dx /= dist; dy /= dist
        return hx*dx + hy*dy 

    # ---------- Main training ----------

    def train(self):
        self.get_logger().info("Starting DQN training (Enhanced Logging)...")

        for episode in range(1, self.n_episodes + 1):
            self.env.reset()
            self.state_processor.reset()
            self.last_action = 0
            
            for _ in range(5): 
                rclpy.spin_once(self.env, timeout_sec=0.1)

            env_state = self.env.get_state()
            start_pos = tuple(env_state['position'])

            # Goal Management
            if (self.current_goal is None) or (self.successes_for_current_goal >= self.successes_to_change):
                self.current_goal = self.sample_random_goal_in_circle()
                self.successes_for_current_goal = 0
                self.get_logger().info(f"[NEW GOAL] {tuple(round(x,3) for x in self.current_goal)}")
            
            self.env.set_goal(self.current_goal)

            # Metrics Init
            env_state = self.env.get_state()
            goal_pos = tuple(env_state['goal_position'])
            proj0, _ = self._compute_line_metrics(start_pos, goal_pos, start_pos)
            prev_proj = proj0

            state = self.state_processor.get_state(
                env_state['scan_data'], env_state['position'],
                env_state['goal_position'], env_state['yaw']
            )

            episode_reward = 0.0
            episode_loss_sum = 0.0
            episode_loss_count = 0
            ep_success = False

            # --------- Step Loop ----------
            for step in range(self.max_steps_per_episode):
                
                # 1.  decision
                action = self.agent.act(state, training=True)
            
                next_raw, base_reward, done = self.env.step(action)
                robot_pos = tuple(next_raw['position'])
                goal_pos = tuple(next_raw['goal_position'])
                yaw = next_raw['yaw']

                proj, d_perp = self._compute_line_metrics(start_pos, goal_pos, robot_pos)
                delta_proj = proj - prev_proj
                prev_proj = proj

                # ----------------------------------REWARD SHhaping
                #obstacle detection
                min_obs_dist = 3.5
                if self.env.scan_data:
                    scan = np.array(self.env.scan_data, dtype=np.float32)
                    scan[~np.isfinite(scan)] = 3.5
                    scan[scan <= 0.01] = 3.5 
                    min_obs_dist = float(scan.min())
                
                # B. Obstacle Penalty (cubic/Suave)
                obstacle_penalty = 0.0
                safe_dist = 1.0
                if min_obs_dist < safe_dist:
                    obstacle_penalty = -4.0 * ((safe_dist - min_obs_dist) / min_obs_dist)
                
                # C. Heading 
                alignment = self._compute_heading_alignment(yaw, robot_pos, goal_pos)
                if min_obs_dist < 0.6:
                    k_heading_dynamic = 0.5 
                else:
                    k_heading_dynamic = 2.0 
                
                if alignment > 0:
                    reward_heading = k_heading_dynamic * alignment
                else:
                    reward_heading = 0.0

                # D. Forward Incentive 
                reward_velocity = 0.0
                if action in [0, 4]: # Base o Fast
                    reward_velocity = 0.2 
                elif action == 3: # Brake
                    reward_velocity = -0.1 

                # E. Wiggle Penalty
                wiggle_penalty = 0.0
                if (action == 1 and self.last_action == 2) or (action == 2 and self.last_action == 1):
                    wiggle_penalty = -0.5
                
                self.last_action = action

                
                shaped_reward = (
                    base_reward
                    + (self.k_progress * delta_proj)    
                    + reward_heading                    
                    + obstacle_penalty                  
                    + reward_velocity                   
                    + wiggle_penalty                    
                    - (self.k_perp * d_perp)            
                )

            
                next_state = self.state_processor.get_state(
                    next_raw['scan_data'], next_raw['position'],
                    next_raw['goal_position'], next_raw['yaw']
                )

                self.agent.remember(state, action, shaped_reward, next_state, done)
                
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                    episode_loss_sum += loss
                    episode_loss_count += 1

                episode_reward += shaped_reward
                state = next_state

                
                if step % 50 == 0 or done:
                    dist = self.env.distance_to_goal()
                    
                    
                    pos_str = f"({robot_pos[0]:.2f}, {robot_pos[1]:.2f})"
                    goal_str = f"({goal_pos[0]:.2f}, {goal_pos[1]:.2f})"
                    
                    prog_str = f"{self.successes_for_current_goal}/{self.successes_to_change}"
                    
                    self.get_logger().info(
                        f"[EP {episode} ST {step}] "
                        f"Pos={pos_str}->Goal={goal_str} | "
                        f"Dist={dist:.2f} | "
                        f"R_Tot={shaped_reward:.2f} | "
                        f"Succ={self.success_count} ({prog_str}) | "
                        f"Coll={self.collision_count}"
                    )

                if done:
                    if self.env.is_goal_reached():
                        ep_success = True
                        self.success_count += 1
                        self.successes_for_current_goal += 1
                        self.get_logger().info(f"*** GOAL REACHED! Total: {self.success_count} ***")
                    if self.env.is_collision():
                        self.collision_count += 1
                        self.get_logger().info("--- COLLISION ---")
                    break
                
                rclpy.spin_once(self.env, timeout_sec=0.01)

            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(step + 1)
            mean_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0.0
            self.episode_mean_loss.append(mean_loss)

            if episode % 10 == 0 or episode == 1:
                avg_reward = np.mean(self.episode_rewards[-10:])
                success_rate = self.success_count / episode * 100.0
                self.get_logger().info(
                    f"SUMMARY EP {episode} | AvgR: {avg_reward:.2f} | "
                    f"Eps: {self.agent.epsilon:.3f} | Success: {success_rate:.1f}%"
                )

            if episode % 50 == 0:
                self.agent.save(os.path.join(self.results_dir, f"model_ep{episode}.pkl"))

        self.agent.save(os.path.join(self.results_dir, "model_final.pkl"))
        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(10, 8))
        plt.subplot(3,1,1); plt.plot(self.episode_rewards); plt.title('Reward')
        plt.subplot(3,1,2); plt.plot(self.episode_steps); plt.title('Steps')
        plt.subplot(3,1,3); plt.plot(self.episode_mean_loss); plt.title('Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_results.png'))

def main(args=None):
    rclpy.init(args=args)
    trainer = DQNTrainingNode()
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(trainer, 'env'): trainer.env.send_velocity(0.0, 0.0)
        trainer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()