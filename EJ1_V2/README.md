- EJERCICIO_1 V2

1. REQUISITOS

- Instalar Ubuntu 22.04  
- Instalar ROS2 Humble  
- Instalar Gazebo Classic  
- Instalar TurtleBot3  
- Instalar dependencias Python:


2. WORKSPACE

2.1 Copiar el workspace del repositorio (dqn_navigation_ws)

2.2 Realizar el Colcon build al ws, para generar los paquetes "build", "log", "install"

2.3 Compilar encontrando dentro del ws
   source install/setup.bash


3. Iniciar Simulación

3.1. Abrir terminal y ejecutar:

   ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

4. TESTEO DEL MODELO ENTRENADO

4.1. En nueva terminal:

   cd ~/dqn_navigation_ws
   source install/setup.bash
4.2. Ejecutar:

   ros2 run dqn_robot_nav test_node /path/to/model_final.pkl (tambien se encuentra en el repositorio con el nombre de: results_20251204_052059)


5. ESTRUCTURA DEL PAQEUTE:

dqn_robot_nav/
│── train_node.py        → Entrenamiento + metas secuenciales
│── test_node.py         → Evaluación del agente
│── environment.py       → Acciones, recompensas, colisiones, odom/LiDAR
│── dqn_agent.py         → DQN, memoria, target network, replay
│── state_processor.py   → Procesamiento del estado (LiDAR → bins)
│── package.xml
│── setup.py
results_20251204_052059/
│── model_episode_50.pkl
│── model_episode_100.pkl
│── model_episode_150.pkl
│── model_episode_200.pkl
│── model_episode_250.pkl
│── model_episode_300.pkl
│── model_episode_350.pkl
│── model_episode_400.pkl
│── model_episode_450.pkl
│── model_episode_500.pkl
│── model_final.pkl (este es el modelo a cargar)
│── training_results.png



