#!/usr/bin/env python3

import cv2
import math
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def cal_distance(landmark_1, landmark_2):
    """Calculate Euclidean distance between two 3D landmarks."""
    return math.sqrt(
        (landmark_2.x - landmark_1.x) ** 2 +
        (landmark_2.y - landmark_1.y) ** 2 +
        (landmark_2.z - landmark_1.z) ** 2
    )


def cal_three_point_angle(middle_point, point_1, point_2):
    """Calculate the angle between three points using dot product formula."""
    vector_1 = [middle_point.x - point_1.x, middle_point.y - point_1.y, middle_point.z - point_1.z]
    vector_2 = [middle_point.x - point_2.x, middle_point.y - point_2.y, middle_point.z - point_2.z]
    
    dot_product = sum(vector_1[idx] * vector_2[idx] for idx in range(len(vector_1)))
    angle = math.acos(dot_product / (
        math.sqrt(sum([vector_1[i] ** 2 for i in range(3)])) *
        math.sqrt(sum([vector_2[i] ** 2 for i in range(3)]))
    ))

    return 180 - math.degrees(angle)


def compute_all_distance(hand_landmarks):
    """Compute the maximum distance between any two landmarks on the hand."""
    max_distance = 0
    num_landmarks = len(hand_landmarks.landmark)
    for i in range(num_landmarks):
        for j in range(i + 1, num_landmarks):
            distance = cal_distance(hand_landmarks.landmark[i], hand_landmarks.landmark[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance


def gesture_stop(hand_landmarks):
    """Detect if the gesture is an open palm."""
    five_finger_palm = []
    for i in range(1, len(hand_landmarks.landmark), 4):
        knuckle_distances = [
            cal_distance(hand_landmarks.landmark[i + j], hand_landmarks.landmark[0]) for j in range(4)
        ]
        
        if knuckle_distances[3] > max(knuckle_distances[:3]) and (knuckle_distances[3] / knuckle_distances[2]) > 0.95:
            five_finger_palm.append(True)
        else:
            five_finger_palm.append(False)

    return all(finger == True for finger in five_finger_palm)


def gesture_okay(hand_landmarks):
    """Detect if the gesture is 'Okay' (thumb and index making a circle)."""
    angles = [
        cal_three_point_angle(hand_landmarks.landmark[3], hand_landmarks.landmark[4], hand_landmarks.landmark[2]),
        cal_three_point_angle(hand_landmarks.landmark[7], hand_landmarks.landmark[8], hand_landmarks.landmark[6]),
        cal_three_point_angle(hand_landmarks.landmark[11], hand_landmarks.landmark[12], hand_landmarks.landmark[10]),
        cal_three_point_angle(hand_landmarks.landmark[15], hand_landmarks.landmark[16], hand_landmarks.landmark[14]),
        cal_three_point_angle(hand_landmarks.landmark[19], hand_landmarks.landmark[20], hand_landmarks.landmark[18])
    ]

    dist_index_thumb = cal_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    #cv2.putText(frame, str(int(dist_index_thumb * 1000)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 128), 3)

    return all([
        angles[0] > 15, angles[1] > 20, angles[2] < 10, angles[3] < 10, angles[4] < 10, dist_index_thumb < 50
    ])


def thumb_direction(hand_landmarks):
    """Determine the direction of the thumb (up, down, left, right)."""
    direction_vector_x = (hand_landmarks.landmark[4].x - hand_landmarks.landmark[2].x) * 1000
    direction_vector_y = (hand_landmarks.landmark[4].y - hand_landmarks.landmark[2].y) * 1000
    direction_vector_z = (hand_landmarks.landmark[4].z - hand_landmarks.landmark[2].z) * 1000

    print(f'x: {direction_vector_x}, y: {direction_vector_y}, z: {direction_vector_z}')

    if direction_vector_y < 0 and abs(direction_vector_x) > abs(direction_vector_y):
        return 1 if direction_vector_x > 0 else 2  # Left or Right
    elif direction_vector_x > 0 and abs(direction_vector_y) > abs(direction_vector_x):
        return 3 if direction_vector_y < 0 else 4  # Up or Down
    elif gesture_stop(hand_landmarks):
        return 0
    elif gesture_okay(hand_landmarks):
        return 5
    else:
        return -1  # No direction detected


class GesturePublisherNode(Node):
    def __init__(self):
        """ROS 2 Publisher Node for gesture detection."""
        super().__init__("gesture_pub")
        self.pub = self.create_publisher(String, "gesture_direction", 10)
        self.timer = self.create_timer(0.5, self.publisher_gesture)
        self.get_logger().info("Gesture Node publisher is running............")
        
    def publisher_gesture(self, gesture):
        """Publish gesture data to the topic."""
        msg = String()
        msg.data = gesture
        self.pub.publish(msg)


def main(args=None):
    """Main function to initialize ROS, OpenCV, and Mediapipe Hand Gestures."""
    rclpy.init(args=args)
    node = GesturePublisherNode()

    # Initialize the Hands model from Mediapipe
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )
    
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_detected = hands.process(frame_rgb)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = frame.shape

        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )

                # Gesture Recognition       
                thumb_direction_result = thumb_direction(hand_landmarks)

                if thumb_direction_result == 1:
                    cv2.putText(frame, "Thumb Left!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    node.publisher_gesture("left")
                elif thumb_direction_result == 2:
                    cv2.putText(frame, "Thumb Right!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    node.publisher_gesture("right")
                elif thumb_direction_result == 3:
                    cv2.putText(frame, "Thumb Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    node.publisher_gesture("up")
                elif thumb_direction_result == 4:
                    cv2.putText(frame, "Thumb Down!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    node.publisher_gesture("down")
                elif thumb_direction_result == 5:
                    cv2.putText(frame, "Okay Gesture Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    node.publisher_gesture("okay")
                elif thumb_direction_result == 0:
                    cv2.putText(frame, "Open Palm Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    node.publisher_gesture("stop")
                else:
                    cv2.putText(frame, "No Thumb Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    node.publisher_gesture("none")

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()
