import cv2
import mediapipe as mp
import time

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

def main():
    # Capture video from the default camera (0)
    cap = cv2.VideoCapture(0)
    
    # Create an instance of the hand detector
    detector = HandDetector()
    
    # Variables to calculate frames per second (FPS)
    prev_time = 0
    
    while True:
        # Read a frame from the video feed
        success, img = cap.read()
        if not success:
            print("Failed to read frame from camera.")
            break
        
        # Find hands in the frame
        img = detector.find_hands(img)
        
        # Get hand landmarks
        lm_list = detector.find_position(img)
        
        # If landmarks exist, check the fingers up
        if len(lm_list) != 0:
            fingers = detector.fingers_up(lm_list)
            print(fingers)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # Show the image
        cv2.imshow("Hand Gesture Recognition", img)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()