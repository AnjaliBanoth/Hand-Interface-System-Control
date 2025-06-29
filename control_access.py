#here we gonna create an gesture based navigation so lets begin with intresting code
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import math
import time

# Load icons if u need to add commands
click_icon = cv2.imread("click.png", cv2.IMREAD_UNCHANGED)
scroll_icon = cv2.imread("scroll.png", cv2.IMREAD_UNCHANGED)
drag_icon = cv2.imread("drag.png", cv2.IMREAD_UNCHANGED)

def overlay_icon(frame, icon, x, y):
    if icon is None:
        return
    icon = cv2.resize(icon, (60, 60))
    h, w, _ = icon.shape
    for i in range(h):
        for j in range(w):
            if icon[i, j][3] != 0:
                frame[y + i, x + j] = icon[i, j][:3]
    return frame

def fingers_up(lm_list):
    fingers = []
    fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)  # thumb
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if lm_list[tip][1] < lm_list[tip - 2][1] else 0)
    return fingers

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

prev_click_time = 0
dragging = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    show_icon = None

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_state = fingers_up(lm_list)
            index = lm_list[8]
            thumb = lm_list[4]
            middle = lm_list[12]

            handedness = results.multi_handedness[i].classification[0].label

            if handedness == "Left":
                # Left hand: clicks and drag
                dist_thumb_index = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
                dist_thumb_middle = math.hypot(middle[0] - thumb[0], middle[1] - thumb[1])

                # Right click
                if dist_thumb_index < 30 and time.time() - prev_click_time > 0.5:
                    pyautogui.click(button='right')
                    prev_click_time = time.time()
                    show_icon = click_icon

                # Left click
                if dist_thumb_middle < 30 and time.time() - prev_click_time > 0.5:
                    pyautogui.click(button='left')
                    prev_click_time = time.time()
                    show_icon = click_icon

                # Drag
                if dist_thumb_index < 30 and not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                    show_icon = drag_icon
                elif dist_thumb_index > 50 and dragging:
                    pyautogui.mouseUp()
                    dragging = False

            elif handedness == "Right":
                # Right hand: navigation and scroll
                if finger_state[1] == 1:
                    cursor_x = np.interp(index[0], [0, w], [0, screen_w])
                    cursor_y = np.interp(index[1], [0, h], [0, screen_h])
                    cur_x, cur_y = pyautogui.position()
                    smooth_x = cur_x + (cursor_x - cur_x) * 0.4
                    smooth_y = cur_y + (cursor_y - cur_y) * 0.4
                    pyautogui.moveTo(smooth_x, smooth_y)

                # Scroll
                if sum(finger_state) == 0:
                    pyautogui.scroll(-30)  # closed hand = scroll down
                    show_icon = scroll_icon
                elif sum(finger_state) >= 4:
                    pyautogui.scroll(30)   # open hand = scroll up
                    show_icon = scroll_icon

    if show_icon is not None:
        overlay_icon(frame, show_icon, 10, 10)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()