import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Screen width and height for cursor normalization
screen_width, screen_height = pyautogui.size()

# Threshold for fingertip distance to trigger click
click_threshold = 0.1

while True:
    success, image = cap.read()

    # Convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image with the hand detection model
    results = hands.process(image)

    # Check if hands were detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract coordinates from the wrist landmark (index 8) and index fingertip (index 8)
            wrist_x, wrist_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
            index_finger_x, index_finger_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y

            # Normalize coordinates to screen dimensions
            normalized_wrist_x = int(wrist_x * screen_width)
            normalized_wrist_y = int(wrist_y * screen_height)
            normalized_index_finger_x = int(index_finger_x * screen_width)
            normalized_index_finger_y = int(index_finger_y * screen_height)

            # Move cursor to normalized wrist coordinates
            pyautogui.moveTo(normalized_wrist_x, normalized_wrist_y)

            # Calculate distance between wrist and index fingertip
            distance = ((normalized_index_finger_x - normalized_wrist_x) ** 2 + (normalized_index_finger_y - normalized_wrist_y) ** 2) ** 0.5

            # Trigger click if distance is below threshold
            if distance < click_threshold:
                pyautogui.click()

            # Draw hand landmarks on the image
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the processed image
    cv2.imshow('Hand Detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanly release resources
cap.release()
cv2.destroyAllWindows()