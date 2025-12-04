#include <Arduino.h>
#include <micro_ros_platformio.h>

#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

#include <geometry_msgs/msg/vector3.h>


#define PUL_PIN 21   // PUL (STEP)
#define DIR_PIN 20   // Direction
#define EN_PIN  19   // Enable (activo en LOW normalmente)


rcl_publisher_t publisher_cmd_nema;
geometry_msgs__msg__Vector3 cmd_nema_msg;

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

#define RCCHECK(fn) { rcl_ret_t rc = fn; if(rc != RCL_RET_OK){while(1){delay(100);}} }
#define RCSOFTCHECK(fn) { rcl_ret_t rc = fn; (void)rc; }


const float SPEED_REF = 0.8f;


const unsigned long PULSE_TOGGLE_US = 500;  

const unsigned long DIR_INTERVAL_MS = 4000; 


bool current_direction = true;   
unsigned long last_dir_switch_ms = 0;


bool pul_state = false;
unsigned long last_pulse_toggle_us = 0;


void timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
  (void) last_call_time;
  if (timer == NULL) return;

  unsigned long now_ms = millis();


  if (now_ms - last_dir_switch_ms >= DIR_INTERVAL_MS) {
    current_direction = !current_direction;
    last_dir_switch_ms = now_ms;

    digitalWrite(DIR_PIN, current_direction ? HIGH : LOW);
  }

  float velocidad = SPEED_REF;

  float sentido = current_direction ? 1.0f : -1.0f;

  cmd_nema_msg.x = velocidad;   // velocidad
  cmd_nema_msg.y = sentido;     // sentido
  cmd_nema_msg.z = 0.0f;        // no usado

  RCSOFTCHECK(rcl_publish(&publisher_cmd_nema, &cmd_nema_msg, NULL));
}

// ----------------- Setup -----------------
void setup() {
  Serial.begin(115200);
  delay(2000);

  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN,  OUTPUT);


  digitalWrite(PUL_PIN, LOW);
  digitalWrite(DIR_PIN, HIGH); 
  digitalWrite(EN_PIN,  LOW);   

  last_dir_switch_ms    = millis();
  last_pulse_toggle_us  = micros();
  current_direction     = true;
  pul_state             = false;


  set_microros_serial_transports(Serial);

  allocator = rcl_get_default_allocator();


  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

  RCCHECK(rclc_node_init_default(
      &node,
      "microros_nema_controller",
      "",
      &support));

  RCCHECK(rclc_publisher_init_default(
      &publisher_cmd_nema,
      &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Vector3),
      "cmd_nema"));

  const unsigned int timer_timeout_ms = 100; // 10 Hz
  RCCHECK(rclc_timer_init_default(
      &timer,
      &support,
      RCL_MS_TO_NS(timer_timeout_ms),
      timer_callback));

  // Executor (solo un timer)
  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));

  Serial.println("micro-ROS NEMA controller started!");
}

void loop() {

  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(5)));
  

  unsigned long now_us = micros();
  if (now_us - last_pulse_toggle_us >= PULSE_TOGGLE_US) {
    last_pulse_toggle_us = now_us;

    pul_state = !pul_state;
    digitalWrite(PUL_PIN, pul_state);
  }


  delay(1);
}

