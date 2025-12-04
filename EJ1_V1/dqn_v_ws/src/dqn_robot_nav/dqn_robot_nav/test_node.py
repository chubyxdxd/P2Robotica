#!/usr/bin/env python3
# test_node.py

import rclpy
from rclpy.node import Node
import numpy as np
import sys
import os
import time

from dqn_robot_nav.dqn_agent import DQNAgent
from dqn_robot_nav.environment import TurtleBot3Env
from dqn_robot_nav.state_processor import StateProcessor

class DQNTestNode(Node):
    """
    Test trained DQN agent.
    MODO: META FIJA (Definida en environment.py)
    """
    
    def __init__(self, model_path: str):
        super().__init__('dqn_test_node')
        
        self.n_lidar_bins = 20
        self.n_stack = 3
        self.state_size = (self.n_lidar_bins + 2) * self.n_stack
        self.action_size = 5
        
        self.env = TurtleBot3Env()
        self.state_processor = StateProcessor(
            n_lidar_bins=self.n_lidar_bins,
            n_stack=self.n_stack
        )
        
        self.agent = DQNAgent(
            state_size=self.state_size, 
            action_size=self.action_size
        )
        
        if os.path.exists(model_path):
            self.agent.load(model_path)
            self.agent.epsilon = 0.0  # Sin aleatoriedad (Greedy)
            self.get_logger().info(f"Model loaded successfully from {model_path}")
        else:
            self.get_logger().error(f"Model path does not exist: {model_path}")
            sys.exit(1)

    def get_processed_state(self):
        env_state = self.env.get_state()
        return self.state_processor.get_state(
            env_state['scan_data'],
            env_state['position'],
            env_state['goal_position'],
            env_state['yaw']
        )
    
    def test(self, n_episodes: int = 10):
        """Run test episodes con META FIJA"""
        self.get_logger().info(f"Starting Test Session: {n_episodes} Episodes")
        
        fixed_goal = self.env.goal_position
        self.get_logger().info(f"Target Goal Fixed at: {fixed_goal}")

        successes = 0
        total_rewards = []
        
        for episode in range(n_episodes):

            #starting
            self.env.reset()              
            self.state_processor.reset()  
            
            for _ in range(5):
                rclpy.spin_once(self.env, timeout_sec=0.1)
            
            
            goal = self.env.goal_position
            self.get_logger().info(f"--- EP {episode+1} START. Going to {goal} ---")
            
            state = self.get_processed_state()
            episode_reward = 0
            
            for step in range(1000):
                action = self.agent.act(state, training=False)
                
                _, reward, done = self.env.step(action)
                state = self.get_processed_state()
                
                episode_reward += reward
                
                if step % 50 == 0:
                    dist = self.env.distance_to_goal()
                    self.get_logger().info(f"Step {step} | Dist to goal: {dist:.2f}")

                if done:
                    if self.env.is_goal_reached():
                        successes += 1
                        self.get_logger().info(
                            f">>> SUCCESS! Steps: {step+1}, Reward: {episode_reward:.2f}"
                        )
                    else:
                        self.get_logger().info(
                            f"XXX COLLISION/FAIL. Steps: {step+1}, Reward: {episode_reward:.2f}"
                        )
                    break
                
                rclpy.spin_once(self.env, timeout_sec=0.01)
            
            total_rewards.append(episode_reward)
        
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info(f"FINAL RESULTS ({n_episodes} eps) on Fixed Goal {fixed_goal}:")
        self.get_logger().info(f"Success Rate: {successes/n_episodes*100:.1f}%")
        self.get_logger().info(f"Avg Reward: {np.mean(total_rewards):.2f}")
        self.get_logger().info("="*50)

def main(args=None):
    rclpy.init(args=args)
    
    if len(sys.argv) < 2:
        print("Usage: ros2 run dqn_robot_nav test_node <model_path>")
        return
    
    model_path = sys.argv[1]
    tester = DQNTestNode(model_path)
    
    try:
        tester.test(n_episodes=5)
    except KeyboardInterrupt:
        pass
    finally:
        tester.env.send_velocity(0.0, 0.0)
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()