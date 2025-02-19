import cv2
import time
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Define HandDetector class
class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image to find hands
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        # List to store landmarks
        lm_list = []
        if self.results.multi_hand_landmarks:
            # Get the specified hand
            my_hand = self.results.multi_hand_landmarks[hand_no]
            
            # Get the coordinates of landmarks
            for id, lm in enumerate(my_hand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                
                # Draw circles on landmarks
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lm_list

    def fingers_up(self, lm_list):
        fingers = []
        
        # Check the thumb
        if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Check the other fingers
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    def find_distance(self, p1, p2, img, draw=True):

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

# Constants for camera width and height
wCam, hCam = 640, 480

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Change 0 to 1 if using an external camera
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Create an instance of the HandDetector class
detector = HandDetector(detection_confidence=0.7)

# Audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    # Process the frame to find hands and landmarks
    img = detector.find_hands(img)
    # Try to unpack the result from find_position method
    position_data = detector.find_position(img, draw=True)

    # Check if position_data contains two elements (lm_list and bbox)
    if position_data and len(position_data) == 2:
        lm_list, bbox = position_data
    else:
        # If not enough data, skip to the next iteration
        continue

    # Check if there are landmarks detected
    if lm_list:
        # Calculate the area of the hand bounding box
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

        # Filter based on the area size
        if 250 < area < 1000:
            # Find the distance between the thumb and index finger
            length, img, line_info = detector.find_distance(4, 8, img)

            # Map the distance to the volume bar range
            vol_bar = np.interp(length, [50, 200], [400, 150])
            vol_per = np.interp(length, [50, 200], [0, 100])

            # Smooth the volume percentage
            smoothness = 10
            vol_per = round(vol_per / smoothness) * smoothness

            # Determine which fingers are up
            fingers = detector.fingers_up()

            # Adjust the volume if the pinky finger is down
            if fingers[4] == 0:
                volume.SetMasterVolumeLevelScalar(vol_per / 100, None)
                # Draw a filled green circle if the volume is adjusted
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                color_vol = (0, 255, 0)
            else:
                color_vol = (255, 0, 0)

    # Draw the volume bar and text
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_per)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Display the current volume level as text
    c_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {int(c_vol)}%', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color_vol, 3)

    # Calculate frames per second (FPS)
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Display the image with hand gestures and volume control
    cv2.imshow("Hand Gesture Control", img)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
