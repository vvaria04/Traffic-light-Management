import cv2
import numpy as np
from typing import Tuple, List

class VehicleDetector:
    def __init__(self):
        # Initialize the vehicle detection model
        self.net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        self.classes = []
        with open('coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Vehicle classes we're interested in (cars, trucks, buses, motorcycles)
        self.vehicle_classes = [2, 3, 5, 6, 7, 8]  # COCO dataset class IDs

    def detect_vehicles(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect vehicles in the frame and return their bounding boxes"""
        height, width, _ = frame.shape
        
        # Preprocess the frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and class_id in self.vehicle_classes:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        vehicle_boxes = []
        for i in indices:
            box = boxes[i]
            vehicle_boxes.append(tuple(box))
            
        return vehicle_boxes

    def count_vehicles_in_region(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> int:
        """Count vehicles in a specific region of the frame"""
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        vehicle_boxes = self.detect_vehicles(roi)
        return len(vehicle_boxes)

    def draw_detections(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw bounding boxes around detected vehicles"""
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame 