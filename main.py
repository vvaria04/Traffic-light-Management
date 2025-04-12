import cv2
import time
from datetime import datetime
from traffic_control_system import TrafficControlSystem
from vehicle_detection import VehicleDetector
import sys
import os
import numpy as np
from collections import deque

def draw_traffic_light(frame, x, y, is_green):
    """Draw a traffic light indicator"""
    color = (0, 255, 0) if is_green else (0, 0, 255)
    cv2.circle(frame, (x, y), 15, color, -1)
    cv2.circle(frame, (x, y), 15, (255, 255, 255), 2)

def get_intersection_regions(frame_width, frame_height):
    """Get the regions of interest for each direction in the intersection."""
    # Define the width and height of each region
    region_width = frame_width // 4
    region_height = frame_height // 4
    
    # Calculate the center of the frame
    center_x = frame_width // 2
    center_y = frame_height // 2
    
    # Define regions for each direction
    # North region (moved a little more to the right)
    north = {
        'x1': center_x - region_width - region_width//4,  # Moved more to the right
        'y1': 0,
        'x2': center_x - region_width//4 - region_width//4,  # Adjusted accordingly
        'y2': region_height
    }
    
    # South region
    south = {
        'x1': center_x - region_width//2,
        'y1': frame_height - region_height,
        'x2': center_x + region_width//2,
        'y2': frame_height
    }
    
    # East region (moved more to the left)
    east = {
        'x1': frame_width - region_width - region_width//2,  # Moved more to the left
        'y1': center_y - region_height//2 - region_height//6,  # Moved slightly up
        'x2': frame_width - region_width//2,  # Adjusted accordingly
        'y2': center_y + region_height//2 - region_height//6  # Adjusted accordingly
    }
    
    # West region (made bigger and stretched down)
    west = {
        'x1': 0,
        'y1': center_y - region_height//2,  # Stretched up
        'x2': region_width + region_width//4,  # Made wider
        'y2': center_y + region_height//2 + region_height//4  # Stretched down
    }
    
    return {
        'north': north,
        'south': south,
        'east': east,
        'west': west
    }

class TrafficLightController:
    def __init__(self):
        self.MAX_GREEN_TIME = 90
        self.MIN_GREEN_TIME = 15
        self.NO_VEHICLE_TIMEOUT = 5
        self.current_green = 'east' 
        self.green_start_time = time.time()
        self.directions = ['north', 'south', 'east', 'west']
        self.last_switch_sequence = []  # Keep track of recent switches
        
    def update(self, vehicle_counts):
        current_time = time.time()
        elapsed_time = current_time - self.green_start_time
        current_count = vehicle_counts[self.current_green]
        
        # Find direction with most waiting vehicles
        max_other_count = 0
        direction_with_max = None
        
        for direction in self.directions:
            if direction != self.current_green and vehicle_counts[direction] > max_other_count:
                max_other_count = vehicle_counts[direction]
                direction_with_max = direction
        
        # Switch conditions:
        # 1. Maximum time elapsed
        # 2. Current direction has no vehicles and minimum time passed and other direction has vehicles
        # 3. Other direction has significantly more vehicles and minimum time passed
        should_switch = (
            elapsed_time >= self.MAX_GREEN_TIME or
            (elapsed_time >= self.MIN_GREEN_TIME and 
             ((current_count == 0 and max_other_count > 0) or
              (max_other_count > current_count * 1.5)))  # Only switch if 50% more vehicles waiting
        )
        
        if should_switch:
            # If there are vehicles waiting, switch to that direction
            if max_other_count > 0:
                self.current_green = direction_with_max
            else:
                # If no vehicles anywhere, rotate through directions
                # but skip the current direction
                available_directions = [d for d in self.directions if d != self.current_green]
                # Prefer directions that haven't been green recently
                for d in available_directions:
                    if d not in self.last_switch_sequence[-2:]:  # Check last 2 switches
                        self.current_green = d
                        break
                else:
                    # If all directions recently used, take the first available
                    self.current_green = available_directions[0]
            
            # Update switch history
            self.last_switch_sequence.append(self.current_green)
            if len(self.last_switch_sequence) > 4:  # Keep last 4 switches
                self.last_switch_sequence.pop(0)
            
            self.green_start_time = current_time
    
    def get_green_direction(self):
        return self.current_green
    
    def get_elapsed_time(self):
        return int(time.time() - self.green_start_time)

def main():
    # Initialize components
    traffic_system = TrafficControlSystem()
    vehicle_detector = VehicleDetector()
    light_controller = TrafficLightController()
    
    # Performance optimization settings
    PROCESS_EVERY_N_FRAMES = 5  # Process more frequently for better detection
    DISPLAY_EVERY_N_FRAMES = 2
    TARGET_WIDTH = 960
    MAX_FPS = 30
    frame_time = 1.0 / MAX_FPS
    
    # Vehicle detection settings
    MIN_VEHICLE_AREA = 400  # Minimum area to consider as vehicle
    DETECTION_THRESHOLD = 0.4  # Lower threshold for better detection
    
    frame_counter = 0
    last_counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
    last_frame_time = time.time()
    
    # Video setup
    video_path = input("Enter the path to your intersection video file: ")
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    print("Video opened successfully!")
    print("Press 'q' to quit, 'p' to pause")
    
    # Initialize frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame")
        return
        
    frame_height, frame_width = frame.shape[:2]
    scale_factor = TARGET_WIDTH / frame_width
    target_height = int(frame_height * scale_factor)
    
    # Pre-calculate regions and positions
    regions = get_intersection_regions(TARGET_WIDTH, target_height)
    traffic_lights = {
        'north': (TARGET_WIDTH // 4, 30),  # Moved more to the left
        'south': (TARGET_WIDTH // 4, target_height - 30),  # Moved more to the left
        'east': (TARGET_WIDTH - 30, target_height // 2),
        'west': (30, target_height // 2)
    }
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    # Initialize background subtractor with adjusted parameters for better standing vehicle detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,  # Increased history to better handle standing vehicles
        varThreshold=25,  # Lower threshold to detect subtle changes
        detectShadows=True  # Enable shadow detection
    )
    
    # Add motion history buffer for each region
    motion_history = {
        'north': deque(maxlen=30),  # Store last 30 frames of motion
        'south': deque(maxlen=30),
        'east': deque(maxlen=30),
        'west': deque(maxlen=30)
    }
    
    while True:
        frame_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_counter += 1
        
        # Skip frames for smoother display
        if frame_counter % DISPLAY_EVERY_N_FRAMES != 0:
            continue
            
        # Resize frame
        display_frame = cv2.resize(frame, (TARGET_WIDTH, target_height))
        
        # Process vehicle detection less frequently
        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(display_frame)
            
            for direction, region in regions.items():
                x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
                roi = fg_mask[y1:y2, x1:x2]
                
                # Count vehicles using contour detection with improved parameters
                _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)  # Lower threshold
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Calculate motion intensity in the region
                motion_intensity = np.sum(thresh) / (thresh.size * 255)
                motion_history[direction].append(motion_intensity)
                
                # Count vehicles with improved detection
                count = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > MIN_VEHICLE_AREA:
                        # Check if contour is likely a vehicle
                        x, y, w, h = cv2.boundingRect(cnt)
                        aspect_ratio = float(w) / h
                        if 0.5 < aspect_ratio < 3.0:  # Typical vehicle aspect ratio
                            count += 1
                
                # Consider standing vehicles based on motion history
                if count == 0 and len(motion_history[direction]) > 0:
                    avg_motion = sum(motion_history[direction]) / len(motion_history[direction])
                    if avg_motion > 0.1:  # If there was recent motion
                        count = last_counts[direction]  # Keep previous count
                
                last_counts[direction] = count
                traffic_system.update_vehicle_density(direction, count)
            
            light_controller.update(last_counts)
        
        # Get traffic light state
        current_green = light_controller.get_green_direction()
        remaining_time = light_controller.MAX_GREEN_TIME - light_controller.get_elapsed_time()
        
        # Draw visualization
        for direction, region in regions.items():
            x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
            # Draw detection region with different colors based on vehicle presence
            color = (0, 255, 0) if last_counts[direction] > 0 else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw vehicle count with background for better visibility
            count_text = f"{direction}: {last_counts[direction]}"
            text_size = cv2.getTextSize(count_text, font, font_scale, font_thickness)[0]
            cv2.rectangle(display_frame, (x1, y1 - 25), (x1 + text_size[0], y1 - 5), (0, 0, 0), -1)
            cv2.putText(display_frame, count_text, (x1, y1 - 10),
                       font, font_scale, (255, 255, 255), font_thickness)
            
            light_x, light_y = traffic_lights[direction]
            draw_traffic_light(display_frame, light_x, light_y, direction == current_green)
        
        # Display status with background
        status_text = f"Green: {current_green}"
        time_text = f"Time: {remaining_time}s"
        cv2.rectangle(display_frame, (5, 5), (200, 70), (0, 0, 0), -1)
        cv2.putText(display_frame, status_text, (10, 30),
                   font, font_scale, (0, 255, 0), font_thickness)
        cv2.putText(display_frame, time_text, (10, 60),
                   font, font_scale, (0, 255, 0), font_thickness)
        
        # Show frame
        cv2.imshow('Traffic Control System', display_frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)
        
        # Control frame rate
        frame_time_elapsed = time.time() - frame_start_time
        if frame_time_elapsed < frame_time:
            time.sleep(frame_time - frame_time_elapsed)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
