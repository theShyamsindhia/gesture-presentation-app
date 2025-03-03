# Gesture-Controlled Presentation App

A Python application that allows you to control presentations using hand gestures, with features for drawing and annotation.

## Features

- Load and display image-based presentations
- Hand gesture controls:
  - Pinch (thumb + index finger) to draw/annotate
  - All fingers up to clear annotations
  - Hand movement to navigate slides
- Real-time drawing with color selection
- Transparent annotation overlay
- Fullscreen presentation mode
- Performance metrics tracking

## Requirements

- Python 3.7+
- Webcam
- Required packages (install using requirements.txt)

## Installation

1. Clone or download this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Convert your presentation to images (if using PDF):
```bash
python pdf_to_images.py
```
This will create a folder named "presentation_slides" with your converted images.

2. Run the presentation app:
```bash
python presentation_app.py
```

3. Select the folder containing your presentation images when prompted.

## Controls

- **Drawing**: Pinch your thumb and index finger together
- **Clear Annotations**: Raise all fingers
- **Navigation**:
  - Move hand to left edge: Previous slide
  - Move hand to right edge: Next slide
- **Keyboard Controls**:
  - 'n': Next slide
  - 'p': Previous slide
  - 'c': Clear annotations
  - 'x': Change drawing color
  - 'l': Load new presentation
  - 'ESC': Exit presentation

## Performance Metrics

The application tracks gesture detection accuracy metrics:
- Total gestures attempted
- Successfully detected gestures
- False negatives
- Overall accuracy percentage

These metrics are displayed periodically during usage.

## Troubleshooting

1. If the webcam isn't detected, ensure it's properly connected and not in use by another application.
2. For optimal hand detection, ensure good lighting conditions.
3. Keep your hand within the camera's field of view.
4. Maintain a reasonable distance from the camera (approximately arm's length).

## Notes

- The application uses MediaPipe for hand tracking
- Annotations are saved per slide during the session
- Performance metrics help optimize gesture detection accuracy
