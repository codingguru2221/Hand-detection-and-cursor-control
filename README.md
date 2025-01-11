# Hand Detection Application

This application utilizes computer vision and hand tracking to control mouse clicks using hand gestures. It leverages the OpenCV library for video capture and the MediaPipe library for hand detection.

## Features

- **Real-time Hand Detection**: The application captures video from the webcam and processes each frame to detect hands using MediaPipe.
  
- **Cursor Control**: The position of the cursor is controlled by the thumb and index finger positions. The coordinates are normalized to the screen dimensions for accurate cursor movement.

- **Click Triggering**: A click is triggered when the distance between the thumb tip and index fingertip is below a specified threshold, allowing for a natural clicking gesture.

- **Click Cooldown**: To prevent multiple clicks from being registered too quickly, a cooldown period is implemented.

- **User-Friendly Interface**: The processed video feed is displayed in a window, showing the detected hand landmarks for visual feedback.

- **Exit Functionality**: The application can be exited gracefully by pressing the 'q' key.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- PyAutoGUI

## Installation

1. Clone the repository or download the files.
2. Install the required libraries using pip:

   ```bash
   pip install opencv-python mediapipe pyautogui
   ```

3. Run the application:

   ```bash
   python try.py
   ```

## Usage

- Position your hand in front of the webcam.
- Move your thumb and index finger to control the cursor.
- Bring your thumb and index finger close together to trigger a click.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
