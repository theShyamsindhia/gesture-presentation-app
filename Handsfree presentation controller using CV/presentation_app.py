import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

class PresentationController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize presentation variables
        self.slides = []
        self.current_slide = 0
        self.prev_x = None
        self.gesture_threshold = 100
        self.last_navigation_time = 0
        self.navigation_cooldown = 0.5  # seconds
        
        # Get screen dimensions
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Define gesture zones
        self.gesture_zone_left = int(self.screen_width * 0.2)
        self.gesture_zone_right = int(self.screen_width * 0.8)
        self.gesture_zone_top = int(self.screen_height * 0.3)
        self.gesture_zone_bottom = int(self.screen_height * 0.7)
        
        # Color selection properties
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 255) # White
        ]
        self.current_color_idx = 0
        
        # Drawing properties
        self.drawing = False
        self.annotations = None
        self.prev_point = None
        
        # Window properties
        cv2.namedWindow('Presentation', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Presentation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def load_images(self, directory=None):
        """Load slides from image files"""
        if directory is None:
            root = tk.Tk()
            root.withdraw()
            directory = filedialog.askdirectory(
                title="Select Folder with Images"
            )
            if not directory:
                return False

        try:
            self.slides = []
            self.slide_positions = []
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            # Get list of image files
            image_files = []
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_files.append(os.path.join(directory, file))
            
            # Sort files by name
            image_files.sort()
            
            print(f"Loading images from: {directory}")
            for img_path in image_files:
                img = cv2.imread(img_path)
                if img is not None:
                    self._process_slide_image(img)
                    print(f"Loaded: {os.path.basename(img_path)}")
            
            if not self.slides:
                print("No valid images found in the selected directory")
                return False
            
            # Initialize annotations layer for each slide
            self.annotations = [np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8) 
                              for _ in range(len(self.slides))]
            print(f"Successfully loaded {len(self.slides)} slides")
            return True

        except Exception as e:
            print(f"Error loading images: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _process_slide_image(self, img):
        """Process and add a single slide image"""
        # Maintain aspect ratio while fitting to screen
        h, w = img.shape[:2]
        aspect = w / h
        
        # Calculate new dimensions to fit screen while maintaining aspect ratio
        if self.screen_width / self.screen_height > aspect:
            new_h = self.screen_height
            new_w = int(new_h * aspect)
        else:
            new_w = self.screen_width
            new_h = int(new_w / aspect)
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h))
        
        # Create black canvas of screen size
        canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Calculate position to center the image
        y_offset = (self.screen_height - new_h) // 2
        x_offset = (self.screen_width - new_w) // 2
        
        # Store the position and dimensions for annotations
        self.slide_positions.append({
            'x': x_offset,
            'y': y_offset,
            'width': new_w,
            'height': new_h
        })
        
        # Place image on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
        self.slides.append(canvas)

    def check_drawing_gesture(self, hand_landmarks):
        """Check for simple thumb-index pinch for drawing"""
        if not hand_landmarks:
            return False
            
        # Get landmarks for thumb and index finger
        thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
        index_tip = hand_landmarks.landmark[8]  # Index finger tip
        
        # Calculate distance between thumb tip and index tip
        pinch_distance = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 +
            (thumb_tip.y - index_tip.y) ** 2
        )
        
        # Enable drawing if the pinch distance is below a threshold
        return pinch_distance < 0.05

    def check_clear_gesture(self, hand_landmarks):
        """Check if all fingers are up to clear annotations"""
        if not hand_landmarks:
            return False
            
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        finger_pips = [6, 10, 14, 18]  # Corresponding pip joints
        
        all_fingers_up = True
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y >= hand_landmarks.landmark[pip].y:
                all_fingers_up = False
                break
                
        # Also check thumb
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_up = thumb_tip.x > thumb_ip.x
        
        return all_fingers_up and thumb_up

    def handle_gestures(self, frame, hand_landmarks):
        """Process hand gestures for navigation and drawing"""
        if not hand_landmarks:
            return frame
            
        # Get index finger tip position
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * self.screen_width)
        y = int(index_tip.y * self.screen_height)
        
        # Get current slide position
        pos = self.slide_positions[self.current_slide]
        
        # Check if cursor is within slide bounds
        in_slide_bounds = (pos['x'] <= x < pos['x'] + pos['width'] and 
                         pos['y'] <= y < pos['y'] + pos['height'])
        
        # Draw cursor
        cursor_color = self.colors[self.current_color_idx] if in_slide_bounds else (150, 150, 150)
        cv2.circle(frame, (x, y), 10, cursor_color, -1)
        
        # Handle drawing with pinch gesture
        if self.check_drawing_gesture(hand_landmarks) and in_slide_bounds:
            self.total_gestures += 1
            self.detected_gestures += 1
            print("Pinch gesture detected, drawing...")  # Debug statement
            if not self.drawing:
                self.drawing = True
                self.prev_point = (x, y)
            else:
                cv2.line(self.annotations[self.current_slide], 
                        self.prev_point, (x, y),
                        self.colors[self.current_color_idx], 3)
                self.prev_point = (x, y)
        else:
            if self.drawing:
                print("Stopped drawing.")  # Debug statement
            self.drawing = False
            self.prev_point = None
            if not self.check_drawing_gesture(hand_landmarks):
                self.false_negatives += 1
        
        # Handle clear gesture
        if self.check_clear_gesture(hand_landmarks):
            self.total_gestures += 1
            self.detected_gestures += 1
            print("Clear gesture detected, clearing annotations...")  # Debug statement
            self.annotations[self.current_slide] = np.zeros_like(self.annotations[self.current_slide])
        else:
            if not self.check_clear_gesture(hand_landmarks):
                self.false_negatives += 1
        
        # Log metrics periodically
        if self.total_gestures % 50 == 0:
            self.log_metrics()
         
        # Handle navigation in gesture zones with cooldown
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if current_time - self.last_navigation_time >= self.navigation_cooldown:
            if self.gesture_zone_top <= y <= self.gesture_zone_bottom:
                if x < self.gesture_zone_left:
                    if self.current_slide > 0:
                        self.current_slide -= 1
                        self.last_navigation_time = current_time
                elif x > self.gesture_zone_right:
                    if self.current_slide < len(self.slides) - 1:
                        self.current_slide += 1
                        self.last_navigation_time = current_time
        
        return frame

    def run(self):
        """Main presentation loop"""
        if not self.slides:
            if not self.load_images():
                print("No images loaded!")
                return
        
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.screen_width, self.screen_height))
            
            # Process hand landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Get current slide and its annotations
            display = self.slides[self.current_slide].copy()
            
            # Add annotations for current slide
            annotations = self.annotations[self.current_slide]
            # Create a transparent overlay
            overlay = display.copy()
            cv2.addWeighted(annotations, 1, overlay, 0.5, 0, overlay)
            
            # Process hand gestures
            if results.multi_hand_landmarks:
                overlay = self.handle_gestures(overlay, results.multi_hand_landmarks[0])
            
            # Show slide number
            cv2.putText(overlay, f"Slide {self.current_slide + 1}/{len(self.slides)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw pen indicator
            pen_status = "Pen: Active" if self.drawing else "Pen: Inactive"
            pen_color = self.colors[self.current_color_idx] if self.drawing else (150, 150, 150)
            cv2.putText(overlay, pen_status, 
                       (self.screen_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, pen_color, 2)
            # Draw color indicator circle
            cv2.circle(overlay, (self.screen_width - 230, 25), 10, self.colors[self.current_color_idx], -1)
            
            # Draw gesture zones
            cv2.line(overlay, (self.gesture_zone_left, 0), 
                    (self.gesture_zone_left, self.screen_height), (50, 50, 50), 2)
            cv2.line(overlay, (self.gesture_zone_right, 0), 
                    (self.gesture_zone_right, self.screen_height), (50, 50, 50), 2)
            
            cv2.imshow('Presentation', overlay)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = PresentationController()
    controller.run()
