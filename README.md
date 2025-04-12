# AI-Controlled Traffic Flow System

An intelligent traffic control system that uses computer vision and adaptive algorithms to manage traffic flow at intersections based on real-time vehicle density.

## Features

- Real-time vehicle detection using background subtraction and contour analysis
- Adaptive traffic light control based on vehicle density
- Smart switching logic with minimum and maximum green light times
- Rotation system for low-traffic periods
- Visual monitoring interface with:
  - Color-coded detection regions
  - Vehicle counts for each direction
  - Traffic light status indicators
  - Remaining time display
  - Pause/resume functionality

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Other dependencies in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd traffic-control-system
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. When prompted, enter the path to your intersection video file.

3. The system will display:
   - Detection regions for each direction (north, south, east, west)
   - Vehicle counts in each region
   - Current traffic light status
   - Remaining time for current green light

4. Controls:
   - Press 'q' to quit
   - Press 'p' to pause/resume

## System Components

### Traffic Light Controller
- Maximum green light time: 90 seconds
- Minimum green light time: 15 seconds
- Smart switching based on:
  - Vehicle presence
  - Waiting time
  - Traffic density ratios
  - Historical patterns

### Vehicle Detection
- Background subtraction using MOG2
- Contour detection for vehicle counting
- Minimum area filtering to reduce false positives
- Real-time processing with frame skip optimization

### Visualization
- Color-coded detection regions:
  - Green: Vehicles present
  - Red: No vehicles
- Traffic light indicators
- Vehicle counts with background
- Status display with timing information

## Performance Optimizations

- Frame skipping for smooth display (every 2nd frame)
- Reduced processing frequency (every 5th frame)
- Background subtraction for efficient vehicle detection
- Resolution scaling (960px width)
- FPS capping (30 FPS)

## Configuration

Key parameters can be adjusted in `main.py`:

```python
# Performance settings
PROCESS_EVERY_N_FRAMES = 5
DISPLAY_EVERY_N_FRAMES = 2
TARGET_WIDTH = 960
MAX_FPS = 30

# Vehicle detection settings
MIN_VEHICLE_AREA = 400
DETECTION_THRESHOLD = 0.4

# Traffic light timings
MAX_GREEN_TIME = 90
MIN_GREEN_TIME = 15
NO_VEHICLE_TIMEOUT = 5
```

## Detection Regions

The system monitors four regions:
- North: Entry/exit point at top
- South: Entry/exit point at bottom
- East: Entry/exit point at right
- West: Entry/exit point at left

Regions are automatically scaled based on video resolution and can be adjusted by modifying the `get_intersection_regions()` function.

## Notes

- The system requires clear visibility of the intersection
- Performance depends on video quality and lighting conditions
- Background subtraction may need adjustment based on weather conditions
- System maintains at least one green light at all times
- Switching logic prevents rapid changes and ensures fair distribution

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Add your license information here] 