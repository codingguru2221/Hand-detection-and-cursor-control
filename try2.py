import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Screen width and height for cursor normalization
screen_width, screen_height = pyautogui.size()

# Threshold for fingertip distance to trigger click
click_threshold = 30  # Adjust threshold as necessary
last_click_time = 0
click_cooldown = 0.5  # Cooldown time in seconds

def detect_click(thumb_tip, index_finger_tip):
    """Detects a click based on the distance between thumb tip and index fingertip."""
    global last_click_time
    distance = ((index_finger_tip[0] - thumb_tip[0]) ** 2 +
                (index_finger_tip[1] - thumb_tip[1]) ** 2) ** 0.5
    current_time = time.time()
    if distance < click_threshold and (current_time - last_click_time) > click_cooldown:
        pyautogui.click()
        last_click_time = current_time

try:
    while True:
        success, image = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with the hand detection model
        results = hands.process(image)

        # Check if hands were detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract coordinates from the thumb tip (index 4) and index fingertip (index 8)
                thumb_tip_x, thumb_tip_y = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
                index_finger_x, index_finger_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y

                # Normalize coordinates to screen dimensions
                normalized_thumb_tip = (int(thumb_tip_x * screen_width), int(thumb_tip_y * screen_height))
                normalized_index_finger_tip = (int(index_finger_x * screen_width), int(index_finger_y * screen_height))

                # Detect click based on thumb and index finger distance
                detect_click(normalized_thumb_tip, normalized_index_finger_tip)

                # Draw hand landmarks on the image
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the processed image
        cv2.imshow('Hand Detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanly release resources
    cap.release()
    cv2.destroyAllWindows()
