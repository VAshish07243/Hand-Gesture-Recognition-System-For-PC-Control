import cv2
import time
import numpy as np
import mediapipe as mp

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

# Constants for camera width and height
wCam, hCam = 640, 480

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Change 0 to 1 if using an external camera
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Create an instance of the HandDetector class
detector = HandDetector(detection_confidence=0.7)

# Initial zoom level and zoom bar position
zoom_level = 1.0
zoomBar = 400

while True:
    # Read a frame from the video feed
    success, img = cap.read()
    
    # Check if frame capture was successful
    if not success:
        print("Failed to read frame from camera. Skipping this frame...")
        continue  # Skip this iteration and try the next frame

    # Find hands in the frame
    img = detector.find_hands(img)
    
    # Get hand landmarks
    lm_list = detector.find_position(img)

    # Process the frame if there are hand landmarks
    if len(lm_list) != 0:
        # Calculate distance between thumb and index finger tips
        x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
        x2, y2 = lm_list[8][1], lm_list[8][2]  # Index finger tip
        length = np.hypot(x2 - x1, y2 - y1)
        
        # Interpolate zoom level based on finger distance
        zoom_level = np.interp(length, [50, 300], [0.5, 3.0])
        zoomBar = np.interp(length, [50, 300], [400, 150])
        
        # Apply zoom action (cropping the frame)
        h, w, _ = img.shape
        new_h = int(h / zoom_level)
        new_w = int(w / zoom_level)
        center_y, center_x = h // 2, w // 2
        
        # Calculate the top-left and bottom-right coordinates for cropping
        top_left_y = max(0, center_y - new_h // 2)
        top_left_x = max(0, center_x - new_w // 2)
        bottom_right_y = min(h, center_y + new_h // 2)
        bottom_right_x = min(w, center_x + new_w // 2)
        
        # Crop the image to achieve zoom effect
        cropped_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        cropped_img = cv2.resize(cropped_img, (wCam, hCam))

        # Replace the current image with the cropped image
        img = cropped_img

        # Draw indicators
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (255, 0, 255), cv2.FILLED)

        # Draw circle around midpoint when close
        if length < 50:
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 255, 0), cv2.FILLED)

    # Draw zoom level bar and level
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(zoomBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'Zoom Level: {zoom_level:.2f}', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Display the image
    cv2.imshow("Hand Gesture Recognition", img)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
