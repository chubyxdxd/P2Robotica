#!/usr/bin/env python3
# train_node.py
import rclpy
from rclpy.node import Node
import numpy as np
from dqn_robot_nav.dqn_agent import DQNAgent
from dqn_robot_nav.environment import TurtleBot3Env
from dqn_robot_nav.state_processor import StateProcessor
import matplotlib.pyplot as plt
from datetime import datetime
import os
import math

class DQNTrainingNode(Node):
    """Main training node for DQN navigation with fixed goal sequence (Option B)"""
    
    def __init__(self):
        super().__init__('dqn_training_node')
        
        # ---------- Training parameters ----------
        self.n_episodes = 700
        self.max_steps_per_episode = 1200
        self.state_size = 12  # 10 LiDAR bins + 2 goal info
        self.action_size = 5
        
        # Initialize components
        self.env = TurtleBot3Env()
        self.state_processor = StateProcessor(n_lidar_bins=10)
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.005,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,    # más exploración residual
            epsilon_decay=0.997, # decay más lento
            memory_size=2000,
            batch_size=64
        )
        
        # ---------- Logging ----------
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_mean_loss = []
        self.success_count = 0
        self.collision_count = 0
        
        # ---------- Fixed goal sequence (Option B) ----------
        # Caja de trabajo: [-1.5, 1.5] x [-1.5, 1.5]
        # Define aquí la ruta que quieres que vaya desbloqueando.
        self.goal_list = [
            (-1.0, -0.5),
            (0.0, -0.5),
            (-1.0, 0.5),
            (0.0, 0.5),
            (1.0, 0.5),
            (-1.5, 1.0),
            (-1.5, 0.0),
            (-1.5, -1.0),
            (-0.5, 1.0),
            (-0.5, 0.0),
            (-0.5, -1.0),
            (1.5, 1.0),
            (1.5, 0.0),
            (1.5, -1.0),
            (1.0, -0.5),
            # Goal 9: arriba de la zona original (comentario que ya tenías)
        ]
        
        self.current_goal_idx = 0
        self.current_goal = self.goal_list[self.current_goal_idx]
        
        # Cuántos éxitos se necesitan para avanzar al siguiente goal
        self.successes_to_advance = 3
        self.successes_for_current_goal = 0
        
        # Directorio de resultados
        self.results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
    
    # ---------- Helpers ----------
    
    def get_processed_state(self):
        env_state = self.env.get_state()
        return self.state_processor.get_state(
            env_state['scan_data'],
            env_state['position'],
            env_state['goal_position'],
            env_state['yaw']
        )
    
    def set_goal_for_episode(self):
        """Fija el goal actual en el environment según current_goal_idx"""
        self.current_goal = self.goal_list[self.current_goal_idx]
        self.env.set_goal(self.current_goal)
        self.get_logger().info(
            f"[GOAL] Using goal_idx={self.current_goal_idx} | "
            f"goal={tuple(round(x, 3) for x in self.current_goal)} | "
            f"successes_for_this_goal={self.successes_for_current_goal}"
        )
    
    def maybe_advance_goal(self):
        """Si hay suficientes éxitos para el goal actual, avanza al siguiente."""
        if self.successes_for_current_goal >= self.successes_to_advance:
            if self.current_goal_idx < len(self.goal_list) - 1:
                old_idx = self.current_goal_idx
                self.current_goal_idx += 1
                self.successes_for_current_goal = 0
                
                # Al cambiar de goal, damos un boost de exploración
                old_eps = self.agent.epsilon
                self.agent.epsilon = max(self.agent.epsilon, 0.3)
                
                self.get_logger().info(
                    f"[GOAL ADVANCE] Goal {old_idx} mastered "
                    f"(>= {self.successes_to_advance} successes). "
                    f"Advancing to goal_idx={self.current_goal_idx}. "
                    f"Epsilon {old_eps:.3f} -> {self.agent.epsilon:.3f}"
                )
            else:
                # Ya estamos en el último goal
                self.get_logger().info(
                    "[GOAL ADVANCE] Last goal in list already. "
                    "Staying on this goal."
                )
    
    # ---------- Main training ----------
    
    def train(self):
        """Main training loop"""
        self.get_logger().info("Starting DQN training with fixed goal sequence (Option B)...")
        
        # Flag por si se completan todos los goals antes de llegar a n_episodes
        all_goals_mastered = False
        
        for episode in range(1, self.n_episodes + 1):
            # 1) Elegir/fijar goal para este episodio según la secuencia
            self.set_goal_for_episode()
            
            # 2) Resetear entorno (no cambia goal)
            self.env.reset()
            rclpy.spin_once(self.env, timeout_sec=0.5)
            
            # Log meta de este episodio
            self.get_logger().info(
                f"[EP {episode}] Goal_position="
                f"{tuple(round(x, 3) for x in self.env.goal_position)} "
                f"(goal_idx={self.current_goal_idx})"
            )
            
            state = self.get_processed_state()
            episode_reward = 0.0
            step = 0
            
            episode_loss_sum = 0.0
            episode_loss_count = 0
            
            # Flag de éxito en este episodio
            ep_success = False
            
            for step in range(self.max_steps_per_episode):
                # Seleccionar y ejecutar acción
                action = self.agent.act(state, training=True)
                next_raw, reward, done = self.env.step(action)
                
                # Construir next_state
                next_state = self.state_processor.get_state(
                    next_raw['scan_data'],
                    next_raw['position'],
                    next_raw['goal_position'],
                    next_raw['yaw']
                )
                
                # Guardar experiencia y entrenar
                self.agent.remember(state, action, reward, next_state, done)
                
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                    episode_loss_sum += loss
                    episode_loss_count += 1
                
                episode_reward += reward
                state = next_state
                
                # Debug EP/STEP
                dist = self.env.distance_to_goal()
                if self.env.scan_data is not None:
                    scan = np.array(self.env.scan_data, dtype=np.float32)
                    scan[~np.isfinite(scan)] = 3.5
                    scan[scan <= 0.01] = 3.5
                    min_obs = float(scan.min())
                else:
                    min_obs = -1.0
                
                self.get_logger().info(
                    f"[EP {episode} STEP {step}] "
                    f"pos={tuple(round(x, 3) for x in self.env.position)} | "
                    f"goal={tuple(round(x, 3) for x in self.env.goal_position)} | "
                    f"dist={dist:.3f} | "
                    f"min_lidar={min_obs:.3f} | "
                    f"reward={reward:.2f} | done={done}"
                )
                
                if done:
                    if self.env.is_goal_reached():
                        ep_success = True
                        self.success_count += 1
                        self.successes_for_current_goal += 1
                        self.get_logger().info(
                            f"[EP {episode}] SUCCESS at goal_idx={self.current_goal_idx} | "
                            f"successes_for_this_goal={self.successes_for_current_goal}"
                        )
                    if self.env.is_collision():
                        self.collision_count += 1
                    break
                
                rclpy.spin_once(self.env, timeout_sec=0.01)
            
            # Estadísticas por episodio
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(step + 1)
            
            if episode_loss_count > 0:
                mean_loss = episode_loss_sum / episode_loss_count
            else:
                mean_loss = 0.0
            self.episode_mean_loss.append(mean_loss)
            
            # Si hubo éxito, ver si avanzamos al siguiente goal
            if ep_success:
                self.maybe_advance_goal()
            
            # ✅ NUEVO: parar si ya dominó TODOS los goals
            if (
                self.current_goal_idx == len(self.goal_list) - 1 and
                self.successes_for_current_goal >= self.successes_to_advance
            ):
                self.get_logger().info(
                    "[TRAIN] All goals mastered "
                    f"(last goal_idx={self.current_goal_idx}, "
                    f"successes_for_this_goal={self.successes_for_current_goal}). "
                    "Stopping training early."
                )
                all_goals_mastered = True
                # rompemos el loop de episodios
                break
            
            # Logging resumen por episodio
            if episode % 10 == 0 or episode == 1:
                avg_reward = np.mean(self.episode_rewards[-10:])
                success_rate = self.success_count / episode * 100.0
                
                self.get_logger().info(
                    f"Episode: {episode}/{self.n_episodes} | "
                    f"Steps: {step+1} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Reward (10): {avg_reward:.2f} | "
                    f"Epsilon: {self.agent.epsilon:.3f} | "
                    f"Success Rate: {success_rate:.1f}% | "
                    f"Mean Loss: {mean_loss:.4f} | "
                    f"Goal_idx: {self.current_goal_idx} | "
                    f"Goal_successes: {self.successes_for_current_goal}"
                )
            
            # Guardar modelo cada 50 episodios (se mantiene igual)
            if episode % 50 == 0:
                model_path = os.path.join(self.results_dir, f"model_episode_{episode}.pkl")
                self.agent.save(model_path)
        
        # Guardado final (tanto si llegó a n_episodes como si salió antes)
        final_model_path = os.path.join(self.results_dir, "model_final.pkl")
        self.agent.save(final_model_path)
        self.plot_results()
    
    # ---------- Plots ----------
    
    def plot_results(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 8))
        
        # 1) Reward
        plt.subplot(3, 1, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # 2) Steps
        plt.subplot(3, 1, 2)
        plt.plot(self.episode_steps)
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        # 3) Mean loss
        plt.subplot(3, 1, 3)
        plt.plot(self.episode_mean_loss)
        plt.title('Mean TD Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'training_results.png')
        self.get_logger().info(f"Results saved to {self.results_dir}")
        plt.savefig(save_path)

def main(args=None):
    rclpy.init(args=args)
    trainer = DQNTrainingNode()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.get_logger().info("Training interrupted by user")
    finally:
        trainer.env.send_velocity(0.0, 0.0)
        trainer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
