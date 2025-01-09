import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

# Function to smooth cursor movements
def smooth_coordinates(prev_coords, new_coords, smoothing_factor=0.7):
    return (smoothing_factor * prev_coords[0] + (1 - smoothing_factor) * new_coords[0],
            smoothing_factor * prev_coords[1] + (1 - smoothing_factor) * new_coords[1])

# Function to check if thumb and index finger are touching
def is_touching(thumb_tip, index_finger_tip, threshold=0.02):
    distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y]))
    return distance < threshold

prev_cursor_x, prev_cursor_y = 0, 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Get coordinates of index finger tip and thumb tip
            index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

            # Convert coordinates to screen size
            cursor_x = int(index_finger_tip.x * screen_width)
            cursor_y = int(index_finger_tip.y * screen_height)

            # Smooth the cursor movements
            cursor_x, cursor_y = smooth_coordinates((prev_cursor_x, prev_cursor_y), (cursor_x, cursor_y))
            prev_cursor_x, prev_cursor_y = cursor_x, cursor_y

            # Move the cursor
            pyautogui.moveTo(cursor_x, cursor_y)

            # Check if index finger and thumb are touching
            if is_touching(thumb_tip, index_finger_tip):
                pyautogui.click()

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()