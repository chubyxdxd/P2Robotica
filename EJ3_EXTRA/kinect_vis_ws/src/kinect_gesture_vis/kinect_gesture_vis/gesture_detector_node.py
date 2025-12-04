import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

class GestureDetector(Node):
    def __init__(self):
        super().__init__('gesture_detector')
        
        self.bridge = CvBridge()
        
        self.PARAMS = {
            'UMBRAL_LADO_ESTIRADO': 40,
            'TOLERANCIA_VERTICAL_ESTIRADO': 60,
            'UMBRAL_LADO_NORMAL': 15,
            'TOLERANCIA_VERTICAL_NORMAL': 30,
            'UMBRAL_ARRIBA': 40,
            'UMBRAL_ABAJO': 80,
            'CONFIANZA_DETECCION': 0.5,
            'ESTABILIDAD_MINIMA': 3,
        }
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.PARAMS['CONFIANZA_DETECCION'],
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.last_gesture = None
        self.stable_count = 0
        
        self.subscription = self.create_subscription(
            Image,
            '/kinect/image_raw',
            self.image_callback,
            10
        )
        
        self.get_logger().info("🟦 Detector de Gestos - Corregido espejo y umbrales ajustados")
    
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        
        if result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(255,0,0), thickness=2)
            )
            
            lm = result.pose_landmarks.landmark
            
            gesture = self.detectar_brazos_estirados(lm, h, w, frame)
            final_gesture = self.aplicar_estabilidad(gesture)
            
            self.mostrar_informacion(frame, lm, h, w, final_gesture)
        
        cv2.imshow("GESTURES", frame)
        cv2.waitKey(1)
    
    def detectar_brazos_estirados(self, lm, h, w, frame):
        left_wrist = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        LW = (int(left_wrist.x * w), int(left_wrist.y * h))
        RW = (int(right_wrist.x * w), int(right_wrist.y * h))
        LS = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        RS = (int(right_shoulder.x * w), int(right_shoulder.y * h))

        mirror = True
        if mirror:
            LW = (w - LW[0], LW[1])
            RW = (w - RW[0], RW[1])
            LS = (w - LS[0], LS[1])
            RS = (w - RS[0], RS[1])

        cv2.line(frame, LS, LW, (0,255,255), 2)
        cv2.line(frame, RS, RW, (0,255,255), 2)
        
        diff_izq = LS[0] - LW[0]
        diff_der = RW[0] - RS[0]
        
        cv2.putText(frame, f"Diff I: {diff_izq}", (LW[0], LW[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(frame, f"Diff D: {diff_der}", (RW[0], RW[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        
        umbral_ext = self.PARAMS['UMBRAL_LADO_ESTIRADO']
        umbral_normal = self.PARAMS['UMBRAL_LADO_NORMAL']
        tol_ext = self.PARAMS['TOLERANCIA_VERTICAL_ESTIRADO']
        tol_norm = self.PARAMS['TOLERANCIA_VERTICAL_NORMAL']
        
        if diff_izq > umbral_ext and diff_der > umbral_ext:
            return "AMBOS LADOS"
        
        if diff_izq > umbral_normal and abs(LW[1] - LS[1]) < tol_norm:
            return "IZQUIERDA"
        
        if diff_der > umbral_normal and abs(RW[1] - RS[1]) < tol_norm:
            return "DERECHA"
        
        if LW[1] < LS[1] - self.PARAMS['UMBRAL_ARRIBA'] and \
           RW[1] < RS[1] - self.PARAMS['UMBRAL_ARRIBA']:
            return "ARRIBA"
        
        if LW[1] > LS[1] + self.PARAMS['UMBRAL_ABAJO'] and \
           RW[1] > RS[1] + self.PARAMS['UMBRAL_ABAJO']:
            return "ABAJO"
        
        return "BUSCANDO..."
    
    def mostrar_informacion(self, frame, lm, h, w, gesture):
        cv2.putText(frame, f"GESTO: {gesture}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    
    def aplicar_estabilidad(self, new_gesture):
        if new_gesture == self.last_gesture:
            self.stable_count += 1
        else:
            self.last_gesture = new_gesture
            self.stable_count = 1
        
        if self.stable_count >= self.PARAMS['ESTABILIDAD_MINIMA']:
            return new_gesture
        else:
            return f"{new_gesture} ({self.stable_count}/{self.PARAMS['ESTABILIDAD_MINIMA']})"

def main(args=None):
    rclpy.init(args=args)
    node = GestureDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
