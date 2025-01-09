import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Get coordinates of index finger tip and thumb tip
            index_finger_tip = landmarks[8]
            thumb_tip = landmarks[4]

            # Convert coordinates to screen size
            cursor_x = int(index_finger_tip.x * screen_width)
            cursor_y = int(index_finger_tip.y * screen_height)

            # Move the cursor
            pyautogui.moveTo(cursor_x, cursor_y)

            # Check if index finger and thumb are close enough to click
            distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
            if distance < 0.02:
                pyautogui.click()

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
