#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <geometry_msgs/msg/twist.h>
#include <std_msgs/msg/string.h>

// Pin definitions
#define LED_FORWARD 33
#define LED_BACKWARD 25
#define LED_LEFT 26
#define LED_RIGHT 27
#define LED_STOP 14
#define BTN_EMERGENCY 12

// Robot parameters
#define LINEAR_SPEED 0.5   // m/s
#define ANGULAR_SPEED 1.0  // rad/s

rcl_publisher_t publisher;
rcl_subscription_t subscriber;
geometry_msgs__msg__Twist twist_msg;
std_msgs__msg__String feedback_msg;
int flag = 0;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

void error_loop(){
  while(1){
    delay(100);
  }
}

void subscription_callback(const void * msgin) {  
  const std_msgs__msg__String * msg = (const std_msgs__msg__String *)msgin;

  // Reset movement
  twist_msg.linear.x = 0.0;
  twist_msg.angular.z = 0.0;

  // Reset LEDs
  digitalWrite(LED_LEFT, LOW);
  digitalWrite(LED_RIGHT, LOW);
  digitalWrite(LED_FORWARD, LOW);
  digitalWrite(LED_BACKWARD, LOW);
  digitalWrite(LED_STOP, LOW);
   // === COMPARACIONES CORRECTAS ===
  if (flag == 1)
  {
      twist_msg.linear.x = 0.0;
      twist_msg.angular.z = 0.0;
  }
  else if (strcmp(msg->data.data, "IZQUIERDA") == 0) {
      twist_msg.linear.x = 0.0;
      twist_msg.angular.z = 1.0;
      digitalWrite(LED_LEFT, HIGH);
  }
  else if (strcmp(msg->data.data, "DERECHA") == 0) {
      twist_msg.linear.x = 0.0;
      twist_msg.angular.z = -1.0;
      digitalWrite(LED_RIGHT, HIGH);
  }
  else if (strcmp(msg->data.data, "ADELANTE") == 0) {
      twist_msg.angular.z = 0.0;
      twist_msg.linear.x = 0.3;
      digitalWrite(LED_FORWARD, HIGH);
  }
  else if (strcmp(msg->data.data, "ATRAS") == 0) {
      twist_msg.angular.z = 0.0;
      twist_msg.linear.x = -0.3;
      digitalWrite(LED_BACKWARD, HIGH);
  }
  else if (strcmp(msg->data.data, "PARAR") == 0) {
      twist_msg.angular.z = 0.0;
      twist_msg.linear.x = 0.0;
      digitalWrite(LED_STOP, HIGH);
  }

  Serial.print("Received: ");
  Serial.println(msg->data.data);
}

void timer_callback(rcl_timer_t * timer, int64_t last_call_time) {  
  RCLC_UNUSED(last_call_time);
  if (timer != NULL) {
    // Publish the message
    bool emergency = digitalRead(BTN_EMERGENCY);
    if(emergency){
      flag ^=1;
  }
    RCSOFTCHECK(rcl_publish(&publisher, &twist_msg, NULL));
  }
}

void setup() {
  Serial.begin(115200);
  
  // Configure button pins
  pinMode(BTN_EMERGENCY, INPUT_PULLDOWN);
  pinMode(LED_RIGHT, OUTPUT);
  pinMode(LED_LEFT, OUTPUT);
  pinMode(LED_STOP, OUTPUT);
  pinMode(LED_FORWARD, OUTPUT);
  pinMode(LED_BACKWARD, OUTPUT);
  // Configure micro-ROS transport
  set_microros_transports();
  
  delay(2000);
  
  allocator = rcl_get_default_allocator();
  
  // Create init_options
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  
  // Create node
  RCCHECK(rclc_node_init_default(&node, "microros_button_controller", "", &support));
  
  // Create publisher
  RCCHECK(rclc_publisher_init_default(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
    "cmd_vel"));
  
  // Create subscriber
  RCCHECK(rclc_subscription_init_default(
    &subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "gesture_command"));
  
  // Create timer (100ms = 10Hz)
  const unsigned int timer_timeout = 100;
  RCCHECK(rclc_timer_init_default(
    &timer,
    &support,
    RCL_MS_TO_NS(timer_timeout),
    timer_callback));
  
  // Create executor
  RCCHECK(rclc_executor_init(&executor, &support.context, 2, &allocator));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));
  RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &feedback_msg, 
    &subscription_callback, ON_NEW_DATA));
  
  // Allocate memory for subscriber message
  feedback_msg.data.data = (char *) malloc(100 * sizeof(char));
  feedback_msg.data.size = 0;
  feedback_msg.data.capacity = 100;
  
  Serial.println("micro-ROS node initialized!");
}

void loop() {
  delay(10);
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10)));
}
