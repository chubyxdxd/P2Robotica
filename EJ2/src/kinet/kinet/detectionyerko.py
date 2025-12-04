#!/usr/bin/env python3

import sys
import numpy as np
print("🔥 Python ejecutándose desde:", sys.executable)
print("🔥 NumPy version:", np.__version__)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import mediapipe as mp


class GestureDetector(Node):
    def __init__(self):
        super().__init__("gesture_detector")

        self.bridge = CvBridge()

        # ================================
        # PARÁMETROS AJUSTADOS
        # ================================
        self.PARAMS = {
            'UMBRAL_LADO_ESTIRADO': 40,
            'UMBRAL_LADO_NORMAL': 15,
            'TOLERANCIA_VERTICAL_NORMAL': 40,
            'UMBRAL_ARRIBA': 40,
            'UMBRAL_ABAJO': 70,
            'ESTABILIDAD_MINIMA': 3,
        }

        # Mediapipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        # Publicador de comandos
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Suscriptor
        self.subscription = self.create_subscription(
            Image, "/kinect/image_raw", self.image_callback, 10
        )

        self.last_gesture = None
        self.stable_count = 0

        self.get_logger().info("🟦 Nodo de detección de gestos listo.")

    # =========================================================
    # CALLBACK DE IMAGEN
    # =========================================================
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            gesture = self.detectar_brazos(lm, h, w)
            final_gesture = self.aplicar_estabilidad(gesture)
            self.controlar_robot(final_gesture)

            # Mostrar texto en pantalla
            cv2.putText(frame, f"Gesto: {final_gesture}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 3)

        cv2.imshow("GESTURE CONTROL", frame)
        cv2.waitKey(1)

    # =========================================================
    # DETECCIÓN DE GESTOS
    # =========================================================
    def detectar_brazos(self, lm, h, w):
        # Obtención de puntos
        LW = (int(lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x * w),
              int(lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y * h))
        RW = (int(lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
              int(lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * h))
        LS = (int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
              int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        RS = (int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
              int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))

        # Imagen espejada
        mirror = True
        if mirror:
            LW = (w - LW[0], LW[1])
            RW = (w - RW[0], RW[1])
            LS = (w - LS[0], LS[1])
            RS = (w - RS[0], RS[1])

        # Distancias laterales
        diff_izq = LS[0] - LW[0]
        diff_der = RW[0] - RS[0]

        # UMBRALES
        ext = self.PARAMS["UMBRAL_LADO_ESTIRADO"]
        norm = self.PARAMS["UMBRAL_LADO_NORMAL"]
        tol = self.PARAMS["TOLERANCIA_VERTICAL_NORMAL"]

        # AMBOS LADOS
        if diff_izq > ext and diff_der > ext:
            return "AMBOS LADOS"

        # IZQUIERDA
        if diff_izq > norm and abs(LW[1] - LS[1]) < tol:
            return "IZQUIERDA"

        # DERECHA
        if diff_der > norm and abs(RW[1] - RS[1]) < tol:
            return "DERECHA"

        # ARRIBA
        if LW[1] < LS[1] - self.PARAMS["UMBRAL_ARRIBA"] and \
           RW[1] < RS[1] - self.PARAMS["UMBRAL_ARRIBA"]:
            return "ARRIBA"

        # ABAJO
        if LW[1] > LS[1] + self.PARAMS["UMBRAL_ABAJO"] and \
           RW[1] > RS[1] + self.PARAMS["UMBRAL_ABAJO"]:
            return "ABAJO"

        return "BUSCANDO..."

    # =========================================================
    # ESTABILIZADOR DE DETECCIÓN
    # =========================================================
    def aplicar_estabilidad(self, new_gesture):
        if new_gesture == self.last_gesture:
            self.stable_count += 1
        else:
            self.last_gesture = new_gesture
            self.stable_count = 1

        if self.stable_count >= self.PARAMS["ESTABILIDAD_MINIMA"]:
            return new_gesture
        else:
            return "BUSCANDO..."

    # =========================================================
    # CONTROL DEL ROBOT
    # =========================================================
    def controlar_robot(self, gesture):
        cmd = Twist()

        if gesture == "IZQUIERDA":
            cmd.angular.z = 1.0

        elif gesture == "DERECHA":
            cmd.angular.z = -1.0

        elif gesture == "ARRIBA":
            cmd.linear.x = 0.3

        elif gesture == "ABAJO":
            cmd.linear.x = -0.3

        elif gesture == "AMBOS LADOS":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        else:
            return  # no publicar mientras está buscando

        self.cmd_pub.publish(cmd)
        self.get_logger().info(f"📡 Enviado comando: {gesture}")


# =========================================================
# MAIN
# =========================================================
def main(args=None):
    rclpy.init(args=args)
    node = GestureDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
