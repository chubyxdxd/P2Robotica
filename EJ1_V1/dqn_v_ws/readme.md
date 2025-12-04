# Documentación Matemática: Navegación Autónoma con DQN

Este documento detalla la formulación matemática utilizada para el entrenamiento del agente robótico (TurtleBot3) mediante Deep Q-Networks (DQN).

## 0. Procedimientos para demo del modelo en UBUNTU

Para poner a prueba el modelo entrenado, abre dos terminales independientes y ejecuta los siguientes comandos:

**TERMINAL 1: Compilación y Ejecución del Nodo de Test**

```bash
# Ejecutar dentro del workspace

colcon build
source install/setup.bash

ros2 run dqn_robot_nav test_node results_log_20251204_153624/model_final.pkl
```
**TERMINAL 2: Entorno de Gazebo**

```bash
# Iniciar Gazebo con el escenario del TurtleBot3
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```


## 1. Definición del Espacio de Estados y Acciones

### 1.1 Espacio de Acciones ($\mathcal{A}$)
El agente opera en un espacio de acciones discreto. Sea $a_t \in \mathcal{A} = \{0, 1, 2, 3, 4\}$, las velocidades lineal ($v$) y angular ($\omega$) se definen como:

$$
\text{Action Mapping:} \quad (v, \omega) = 
\begin{cases} 
(0.0, 0.0) & \text{si } a_t = 0 \text{ (Stop)} \\
(v_{ref} + 0.05, 0.8) & \text{si } a_t = 1 \text{ (Left Curve)} \\
(v_{ref} + 0.05, -0.8) & \text{si } a_t = 2 \text{ (Right Curve)} \\
(v_{ref} - 0.1, 0.0) & \text{si } a_t = 3 \text{ (Slow/Back)} \\
(v_{ref} + 0.1, 0.0) & \text{si } a_t = 4 \text{ (Fast Forward)}
\end{cases}
$$

Adicionalmente, se aplica un controlador proporcional para la corrección de rumbo base ($w_{ref}$) en función del error angular hacia el objetivo.

### 1.2 Espacio de Estados ($\mathcal{S}$)
El estado $s_t$ es un vector compuesto por la concatenación de $k$ *frames* temporales (Stacking) para capturar la dinámica temporal. Dado un vector de características $x_t$ que incluye datos del LiDAR discretizado y coordenadas relativas polares al objetivo:

$$s_t = [x_t, x_{t-1}, ..., x_{t-k+1}]$$

La dimensión total es $dim(s_t) = (N_{lidar} + 2) \times N_{stack}$.

---

## 2. Ingeniería de Recompensas (Reward Shaping)

La función de recompensa total $R_{total}$ en el instante $t$ es una suma ponderada de múltiples incentivos y penalizaciones diseñada para fomentar un movimiento suave y eficiente hacia el objetivo.

$$R_{total} = r_{base} + r_{prog} + r_{head} + r_{obs} + r_{vel} + r_{wiggle} - r_{perp}$$

A continuación se detalla cada componente:

### 2.1 Progreso Proyectado ($r_{prog}$)
Se premia el avance efectivo sobre la línea ideal que conecta el punto de inicio ($P_{start}$) y la meta ($P_{goal}$). Sea $\vec{v} = P_{goal} - P_{start}$ y $\vec{w} = P_{robot} - P_{start}$. La proyección escalar $proj$ se calcula como:

$$proj_t = \frac{\vec{w} \cdot \vec{v}}{||\vec{v}||}$$

La recompensa es proporcional al incremento de esta proyección:

$$r_{prog} = k_{prog} \cdot (proj_t - proj_{t-1})$$

### 2.2 Penalización por Desviación Perpendicular ($r_{perp}$)
Para mantener al robot cerca de la ruta óptima, se penaliza la distancia perpendicular ($d_{\perp}$) a la recta ideal:

$$d_{\perp} = \frac{|w_x v_y - w_y v_x|}{||\vec{v}||}$$
$$r_{perp} = k_{perp} \cdot d_{\perp}$$

### 2.3 Evasión de Obstáculos ($r_{obs}$)
Se utiliza una función de penalización suave (no binaria) basada en la distancia mínima al obstáculo ($d_{min}$). Si $d_{min}$ es menor que una distancia segura $d_{safe}$:

$$r_{obs} = -4.0 \cdot \left( \frac{d_{safe} - d_{min}}{d_{min}} \right)$$

Esta función hiperbólica aumenta drásticamente la penalización a medida que el robot se acerca al obstáculo.

### 2.4 Alineación de Rumbo ($r_{head}$)
Se incentiva que la orientación del robot ($\theta_{yaw}$) coincida con el vector hacia el objetivo. Sea $\vec{h} = [\cos(\theta), \sin(\theta)]$ y $\vec{g}$ el vector unitario hacia la meta:

$$alignment = \vec{h} \cdot \vec{g}$$
$$r_{head} = \begin{cases} k_{heading} \cdot alignment & \text{si } alignment > 0 \\ 0 & \text{e.o.c.} \end{cases}$$

Donde $k_{heading}$ es dinámico (se reduce si hay obstáculos muy cerca para priorizar la evasión).

### 2.5 Estabilidad de Control ($r_{wiggle}$)
Para evitar oscilaciones rápidas (zig-zag), se penaliza si el robot alterna inmediatamente entre izquierda ($a=1$) y derecha ($a=2$):

$$r_{wiggle} = -0.5 \quad \text{si } (a_t=1 \land a_{t-1}=2) \lor (a_t=2 \land a_{t-1}=1)$$

---

## 3. Algoritmo Deep Q-Network (DQN)

El objetivo es encontrar la política óptima $\pi^*$ que maximice el retorno esperado acumulado $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$.

### 3.1 Aproximación de la Función Q
Se utiliza una red neuronal (MLP) para aproximar la función de valor acción $Q(s, a; \theta) \approx Q^*(s, a)$, donde $\theta$ son los pesos de la red.

* **Arquitectura:** Perceptrón Multicapa.
* **Capas Ocultas:** 2 capas de 128 neuronas.
* **Activación:** ReLU ($f(x) = \max(0, x)$).
* **Optimizador:** Adam.

### 3.2 Ecuación de Bellman y Cálculo del Objetivo
Para estabilizar el entrenamiento, se utilizan dos redes:
1.  **Red Principal ($Q$):** Parámetros $\theta$, se actualiza en cada paso.
2.  **Red Objetivo ($Q_{target}$):** Parámetros $\theta^-$, se actualiza cada $C$ pasos ($\theta^- \leftarrow \theta$).

El valor objetivo $y_i$ para una transición $(s, a, r, s', done)$ se calcula como:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-) \cdot (1 - done)$$

Donde $\gamma = 0.95$ es el factor de descuento.

### 3.3 Función de Pérdida (Loss Function)
La red se entrena minimizando el Error Cuadrático Medio (MSE) entre la predicción actual y el objetivo de Bellman sobre un *minibatch* de experiencia $B$:

$$\mathcal{L}(\theta) = \frac{1}{|B|} \sum_{(s,a,r,s') \in B} \left( y - Q(s, a; \theta) \right)^2$$

### 3.4 Política de Exploración ($\epsilon$-greedy)
Durante el entrenamiento, la acción se selecciona mediante:

$$a_t = \begin{cases} \text{random}(\mathcal{A}) & \text{con probabilidad } \epsilon \\ \arg\max_a Q(s_t, a; \theta) & \text{con probabilidad } 1 - \epsilon \end{cases}$$

El valor de $\epsilon$ decae exponencialmente en cada episodio:
$$\epsilon_{k+1} = \max(\epsilon_{min}, \epsilon_k \cdot \epsilon_{decay})$$


