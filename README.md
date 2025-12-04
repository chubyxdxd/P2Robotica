# P2Robotica

**Examen Segundo Parcial de Robótica**

------------------------------------------------------------------------

## Ejercicio 1

------------------------------------------------------------------------

## Ejercicio 2

Para poder correr este ejercicio, se debe entrar a la carpeta **`EJ2`**,
donde se encuentra:

-   el *workspace* (`EJ2_ws`)
-   el archivo **`.ino`**, correspondiente al código cargable en la
    **ESP32** como *micro-ros-agent*.

### Adaptación del envío de imágenes

Dado que no contamos con una Kinect, adaptamos el envío de datos tipo
**Image** usando una **cámara RGB**, publicando en el tópico:

    /kinect/image_raw

El nodo de la cámara se llama `camera_node.py`.

### Ejecutar el nodo de la cámara

Para correr el ejecutable:

    ros2 run kinet webcam_node

Esto abrirá la cámara correspondiente.\
Asegúrate de tener una cámara conectada y configurada correctamente en
`VideoCapture`.

------------------------------------------------------------------------

### Nodo de interpretación de gestos

Para ejecutar el nodo encargado de interpretar gestos con **Mediapipe**,
usa:

    ros2 run kinet p2

Este nodo:

-   Tiene un **suscriptor** que recibe la imagen
-   Muestra la imagen interpretada
-   Publica el tópico:
```{=html}
```
    /gesture_command

------------------------------------------------------------------------

### Inicializar el micro-ROS Agent en la ESP32

Una vez cargado el código en la ESP32, con **Docker instalado** y las
conexiones de pines a **LEDs y botón** realizadas, se debe ejecutar:

    docker run -it --rm --privileged -v /dev:/dev --net=host microros/micro-ros-agent:humble serial --dev /dev/ttyUSB0

Esto debería crear e inicializar el nodo, publicador y subscriptor.\
Luego, **presionar el botón de reset por 5 segundos y soltar**.

El nodo de micro-ROS publica en el tópico:

    /cmd_vel

------------------------------------------------------------------------

### Ejecutar el simulador TurtleBot

Para que el robot en **Gazebo** se mueva según los gestos, se debe abrir
el simulador ejecutando:

    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

------------------------------------------------------------------------

## 📘 Ejercicio 3
