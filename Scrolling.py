import cv2
import time
import numpy as np
import mediapipe as mp
import pyautogui  # For controlling the scroll

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
        self.tip_ids = [8, 12]  # Index and middle finger tips

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

# Constants for camera width and height
wCam, hCam = 640, 480

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Change 0 to 1 if using an external camera
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Create an instance of the HandDetector class
detector = HandDetector(detection_confidence=0.7)

# Variables to track pinch and scroll
scroll_start = False
prev_y_pos = None

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
        # Get the current positions of the index and middle fingers
        index_x, index_y = lm_list[8][1], lm_list[8][2]  # Index finger tip
        middle_x, middle_y = lm_list[12][1], lm_list[12][2]  # Middle finger tip

        # Calculate the distance between index and middle finger
        distance = np.hypot(index_x - middle_x, index_y - middle_y)

        # Check if index and middle finger are close enough (e.g., distance < 30 pixels)
        if distance < 30:
            if not scroll_start:  # Start scroll tracking
                scroll_start = True
                prev_y_pos = index_y
                print("Scrolling started")
            
            # Calculate movement delta
            delta_y = prev_y_pos - index_y
            prev_y_pos = index_y
            
            # Apply the scroll action based on the delta
            pyautogui.scroll(delta_y * 2)  # Adjust the factor to control scroll speed

            # Draw circles to indicate the scroll state
            cv2.circle(img, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)  # Green circle for scrolling
            cv2.circle(img, (middle_x, middle_y), 15, (0, 255, 0), cv2.FILLED)
        else:
            # Reset scrolling state if fingers are no longer pinching
            if scroll_start:
                scroll_start = False
                print("Scrolling stopped")
            
            # Draw regular circles when not scrolling
            cv2.circle(img, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (middle_x, middle_y), 15, (255, 0, 255), cv2.FILLED)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Display the image
    cv2.imshow("Hand Gesture Recognition with Pinch-based Scrolling", img)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
