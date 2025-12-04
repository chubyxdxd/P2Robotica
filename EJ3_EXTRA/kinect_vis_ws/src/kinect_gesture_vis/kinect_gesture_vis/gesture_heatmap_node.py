import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray 
from cv_bridge import CvBridge
import cv2
import numpy as np

class GestureHeatMap(Node):
    def __init__(self):
        super().__init__('gesture_heat_map')
        
        # Suscriptor a la cámara
        self.subscription = self.create_subscription(
            Image,
            '/kinect/depth/image_raw', 
            self.listener_callback,
            10)
        
        self.publisher_ = self.create_publisher(Float32MultiArray, 'cmd_distances', 10)
        
        self.bridge = CvBridge()
        self.get_logger().info('Nodo Heatmap publicando en /cmd_distances iniciado...')

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            depth_array = np.array(cv_image, dtype=np.float32)
            cv2.normalize(depth_array, depth_array, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_array.astype(np.uint8)
            depth_uint8 = cv2.bitwise_not(depth_uint8)
            heatmap_img = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

            height, width = cv_image.shape
            step_y = height // 3
            step_x = width // 3

            # Colores
            BLANCO = (255, 255, 255)
            AZUL   = (255, 0, 0)
            VERDE  = (0, 255, 0)
            AMARILLO = (0, 255, 255)
            ROJO   = (0, 0, 255)

            block_distances = [0.0, 0.0, 0.0] 

            for i in range(3):      
                for j in range(3):  
                    
                    y_start = i * step_y
                    y_end = (i + 1) * step_y
                    x_start = j * step_x
                    x_end = (j + 1) * step_x

                    avg_distance = 0.0
                    text_color = BLANCO 

                    if i == 0:
                        roi = cv_image[y_start:y_end, x_start:x_end]
                        valid_pixels = roi[roi > 0]
                        if len(valid_pixels) > 0:
                            avg_distance = np.mean(valid_pixels)
                        text_color = BLANCO

                    else:
                        block_y_start = 1 * step_y
                        block_y_end   = 3 * step_y
                        
                        roi_block = cv_image[block_y_start:block_y_end, x_start:x_end]
                        valid_pixels = roi_block[roi_block > 0]
                        
                        if len(valid_pixels) > 0:
                            avg_distance = np.mean(valid_pixels)
                        
                        block_distances[j] = float(avg_distance)

                        if avg_distance < 2900:
                            text_color = ROJO
                        elif avg_distance < 3500:
                            text_color = AMARILLO
                        elif avg_distance <= 4000:
                            text_color = VERDE
                        else: 
                            text_color = AZUL

                    cv2.rectangle(heatmap_img, (x_start, y_start), (x_end, y_end), (200, 200, 200), 1)
                    text = f"{int(avg_distance)} mm"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = x_start + (step_x - text_size[0]) // 2
                    text_y = y_start + (step_y + text_size[1]) // 2
                    cv2.putText(heatmap_img, text, (text_x+2, text_y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                    cv2.putText(heatmap_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            msg_distances = Float32MultiArray()
            msg_distances.data = block_distances 
            self.publisher_.publish(msg_distances)
            

            cv2.imshow("Kinect Logic Blocks", heatmap_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error procesando imagen: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = GestureHeatMap()
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