import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np


# Initialize Mediapipe Hand model with detection and tracking confidence
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# To track whether a key is currently being held down
current_key = None
last_press_time = time.time()  # Track the last press time
press_delay = 0.2  # Delay between key presses (in seconds)

# Counters for each gesture (click count)
gesture_counts = {
    'up': 0,
    'down': 0,
    'left': 0,
    'right': 0
}

# Define font and text size
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.9
font_thickness = 2

# Load logo
overlay_img = cv2.imread("llama.jpg", cv2.IMREAD_UNCHANGED)
overlay_img = cv2.resize(overlay_img, (50, 50)) 

# Function to detect specific gestures based on hand landmarks
def detect_gesture(hand_landmarks, frame_width, is_mirror=False):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    if is_mirror:
        thumb_tip.x = 1 - thumb_tip.x
        index_tip.x = 1 - index_tip.x
        middle_tip.x = 1 - middle_tip.x
        ring_tip.x = 1 - ring_tip.x
        pinky_tip.x = 1 - pinky_tip.x
        wrist.x = 1 - wrist.x

    # Gesture: Up (all fingers extended and well spread)
    if (all([
        index_tip.y < wrist.y,  # Above wrist
        middle_tip.y < wrist.y,
        ring_tip.y < wrist.y,
        pinky_tip.y < wrist.y,
    ]) and abs(thumb_tip.x - pinky_tip.x) > 0.3):
        return 'up'

    # Gesture: Down (fist gesture, all fingers curled)
    if (
        all([index_tip.y > wrist.y, middle_tip.y > wrist.y, ring_tip.y > wrist.y, pinky_tip.y > wrist.y])
        and abs(index_tip.x - thumb_tip.x) < 0.1
    ):
        return 'down'

    # Gesture: Left (index and thumb extended, others curled)
    if (
        index_tip.y < wrist.y
        and middle_tip.y > index_tip.y
        and abs(index_tip.x - thumb_tip.x) > 0.1
        and (index_tip.x < thumb_tip.x if not is_mirror else index_tip.x > thumb_tip.x)
    ):
        return 'left'

    # Gesture: Right (index and thumb extended, others curled)
    if (
        index_tip.y < wrist.y
        and middle_tip.y > index_tip.y
        and abs(index_tip.x - thumb_tip.x) > 0.1
        and (index_tip.x > thumb_tip.x if not is_mirror else index_tip.x < thumb_tip.x)
    ):
        return 'right'

    return None

# Function to simulate a key click (press and release the key)
def click_key(direction):
    global current_key, last_press_time

    current_time = time.time()
    if current_time - last_press_time >= press_delay:
        if direction == 'up':
            if current_key != 'up':
                pyautogui.press('up')
                current_key = 'up'
                gesture_counts['up'] += 1
        elif direction == 'down':
            if current_key != 'down':
                pyautogui.press('down')
                current_key = 'down'
                gesture_counts['down'] += 1
        elif direction == 'left':
            if current_key != 'left':
                pyautogui.press('left')
                current_key = 'left'
                gesture_counts['left'] += 1
        elif direction == 'right':
            if current_key != 'right':
                pyautogui.press('right')
                current_key = 'right'
                gesture_counts['right'] += 1
        last_press_time = current_time

# Function to press and hold the "down" key
def press_key(direction):
    global current_key, last_press_time

    current_time = time.time()
    if current_time - last_press_time >= press_delay:
        if direction == 'down':
            if current_key != 'down':
                pyautogui.keyDown('down')
                current_key = 'down'
                gesture_counts['down'] += 1
        last_press_time = current_time

# Function to release the key
def release_key():
    global current_key
    if current_key:
        pyautogui.keyUp(current_key)
        current_key = None

# Function to draw gesture counts on the frame
def draw_background(frame, text_position, color, transparency=0.6):
    x, y = text_position
    text = f"Up: {gesture_counts['up']} | Down: {gesture_counts['down']} | Left: {gesture_counts['left']} | Right: {gesture_counts['right']}"
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 50), (x + text_width + 10, y + text_height + 10), color, -1)
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)

    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

# Function to draw performance percentage
def draw_performance(frame):
    total_gestures = sum(gesture_counts.values())
    if total_gestures == 0:
        performance_percentage = 0
    else:
        performance_percentage = (total_gestures / (4 * max(gesture_counts.values(), default=1))) * 100

    performance_text = f"Performance: {performance_percentage:.2f}%"
    cv2.putText(frame, performance_text, (10, frame.shape[0] - 10), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    

# Main loop to process video frames
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame from webcam.")
        break

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (600, 400))

    # Draw a thin border around the frame
    border_thickness = 50
    frame_height, frame_width, _ = frame.shape
    border_color = (201, 170, 136)  # Border colour

    # Draw the border (top-left and bottom-right corners)
    cv2.rectangle(frame, (0, 0), (frame_width-1, frame_height-1), border_color, border_thickness)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_gesture = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_gesture = detect_gesture(hand_landmarks, frame_width=frame.shape[1], is_mirror=True)

    if detected_gesture:
        if detected_gesture == 'down':
            press_key(detected_gesture)
        else:
            click_key(detected_gesture)
    else:
        release_key()

    draw_background(frame, (10, 50), (0, 0, 0))
    draw_performance(frame)
    
    # Logo on the bottom-right corner
    h, w, _ = overlay_img.shape  # Dimensions of the overlay image
    x_offset = frame.shape[1] - w - 10  # Bottom-right corner with 10px margin
    y_offset = frame.shape[0] - h - 10

    if overlay_img.shape[2] == 4:
        bgr_img = overlay_img[:, :, :3]
        alpha_channel = overlay_img[:, :, 3] / 255.0

        for c in range(0, 3):  
            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                bgr_img[:, :, c] * alpha_channel
    else:
        frame[y_offset:y_offset+h, x_offset:x_offset+w] = overlay_img


    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import pyautogui
# import time
# import numpy as np

# # Import your new Simon Says Python code
# import simon_game as simon_game

# simon_game.startGame()


# # Initialize Mediapipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# # Webcam capture
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Gesture tracking / cooldown
# current_key = None
# last_press_time = time.time()
# press_delay = 0.5  # Increase this if you want more time between inputs

# # Count each gesture
# gesture_counts = {
#     'up': 0,
#     'down': 0,
#     'left': 0,
#     'right': 0
# }

# # Font settings for on-screen text
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 0.9
# font_thickness = 2

# # Optional logo overlay
# overlay_img = cv2.imread("llama.jpg", cv2.IMREAD_UNCHANGED)
# overlay_img = cv2.resize(overlay_img, (50, 50))

# def detect_gesture(hand_landmarks, is_mirror=False):
#     """
#     Return 'up', 'down', 'left', 'right' if recognized, otherwise None.
#     """
#     thumb_tip = hand_landmarks.landmark[4]
#     index_tip = hand_landmarks.landmark[8]
#     middle_tip = hand_landmarks.landmark[12]
#     ring_tip = hand_landmarks.landmark[16]
#     pinky_tip = hand_landmarks.landmark[20]
#     wrist = hand_landmarks.landmark[0]

#     # Mirror the x-coordinates if needed
#     if is_mirror:
#         thumb_tip.x = 1 - thumb_tip.x
#         index_tip.x = 1 - index_tip.x
#         middle_tip.x = 1 - middle_tip.x
#         ring_tip.x = 1 - ring_tip.x
#         pinky_tip.x = 1 - pinky_tip.x
#         wrist.x = 1 - wrist.x

#     # Up: all fingers above wrist & a wide spread
#     if (all([index_tip.y < wrist.y,
#              middle_tip.y < wrist.y,
#              ring_tip.y < wrist.y,
#              pinky_tip.y < wrist.y]) 
#         and abs(thumb_tip.x - pinky_tip.x) > 0.3):
#         return "up"

#     # Down: fist-like
#     if (all([index_tip.y > wrist.y,
#              middle_tip.y > wrist.y,
#              ring_tip.y > wrist.y,
#              pinky_tip.y > wrist.y])
#         and abs(index_tip.x - thumb_tip.x) < 0.1):
#         return "down"

#     # Left: index + thumb extended, others not
#     if (index_tip.y < wrist.y
#         and middle_tip.y > index_tip.y
#         and abs(index_tip.x - thumb_tip.x) > 0.1
#         and (index_tip.x < thumb_tip.x if not is_mirror else index_tip.x > thumb_tip.x)):
#         return "left"

#     # Right: index + thumb extended, others not
#     if (index_tip.y < wrist.y
#         and middle_tip.y > index_tip.y
#         and abs(index_tip.x - thumb_tip.x) > 0.1
#         and (index_tip.x > thumb_tip.x if not is_mirror else index_tip.x < thumb_tip.x)):
#         return "right"

#     return None

# def click_key(gesture):
#     """
#     Map gestures to Simon color indices.
#     up -> 0, right -> 1, down -> 2, left -> 3
#     """
#     global current_key, last_press_time
#     current_time = time.time()
#     if current_time - last_press_time < press_delay:
#         return  # too soon, ignore

#     if gesture == 'up':
#         simon_game.addToPlayerSequence(0)
#         gesture_counts['up'] += 1
#         current_key = 'up'
#     elif gesture == 'right':
#         simon_game.addToPlayerSequence(1)
#         gesture_counts['right'] += 1
#         current_key = 'right'
#     elif gesture == 'down':
#         simon_game.addToPlayerSequence(2)
#         gesture_counts['down'] += 1
#         current_key = 'down'
#     elif gesture == 'left':
#         simon_game.addToPlayerSequence(3)
#         gesture_counts['left'] += 1
#         current_key = 'left'

#     last_press_time = current_time

# def release_key():
#     """
#     If you were holding a key, release it.
#     In this logic, we don't really need to hold any keys down,
#     so it's mostly a no-op.
#     """
#     global current_key
#     current_key = None

# def draw_info(frame):
#     """
#     Draw gesture counts and a simple performance metric on the video feed.
#     """
#     # Gesture counts
#     text_counts = (f"Up: {gesture_counts['up']} | "
#                    f"Down: {gesture_counts['down']} | "
#                    f"Left: {gesture_counts['left']} | "
#                    f"Right: {gesture_counts['right']}")
#     (tw, th), _ = cv2.getTextSize(text_counts, font, font_scale, font_thickness)

#     # Semi-transparent background
#     x, y = 10, 50
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (x - 10, y - 50), 
#                   (x + tw + 10, y + th + 10), 
#                   (0, 0, 0), -1)
#     alpha = 0.6
#     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#     cv2.putText(frame, text_counts, (x, y), font, 
#                 font_scale, (255, 255, 255), font_thickness)

#     # Performance
#     total_g = sum(gesture_counts.values())
#     if total_g == 0:
#         perf = 0
#     else:
#         m = max(gesture_counts.values())
#         perf = (total_g / (4 * m)) * 100
#     perf_text = f"Performance: {perf:.2f}%"
#     cv2.putText(frame, perf_text, (10, frame.shape[0] - 10), 
#                 font, 0.9, (255, 255, 255), 2)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame from webcam.")
#         break

#     # Mirror flip
#     frame = cv2.flip(frame, 1)
#     frame = cv2.resize(frame, (640, 480))

#     # Convert for Mediapipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb_frame)

#     detected_gesture = None
#     if results.multi_hand_landmarks:
#         # We only look at the first hand for simplicity
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             detected_gesture = detect_gesture(hand_landmarks, is_mirror=True)
#             if detected_gesture:
#                 break  # Just use the first recognized hand/gesture

#     if detected_gesture:
#         click_key(detected_gesture)
#     else:
#         release_key()

#     draw_info(frame)

#     # Draw an optional logo in bottom-right
#     if overlay_img is not None:
#         h, w, _ = overlay_img.shape
#         x_offset = frame.shape[1] - w - 10
#         y_offset = frame.shape[0] - h - 10

#         # If PNG has alpha channel
#         if overlay_img.shape[2] == 4:
#             bgr_img = overlay_img[:, :, :3]
#             alpha_channel = overlay_img[:, :, 3] / 255.0
#             for c in range(3):
#                 frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = (
#                     frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel)
#                     + bgr_img[:, :, c] * alpha_channel
#                 )
#         else:
#             frame[y_offset:y_offset+h, x_offset:x_offset+w] = overlay_img

#     cv2.imshow("Hand Gesture Control", frame)

#     # Quit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
