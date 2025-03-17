# YOLO Object Detection Demo

A real-time object detection app for iOS using YOLO (You Only Look Once) and CoreML. This app demonstrates object detection using the camera feed with bounding boxes and confidence scores.

## Features

- Real-time object detection using YOLO model
- Camera preview with live detection boxes
- Confidence scores for each detection
- Non-Maximum Suppression (NMS) to eliminate overlapping detections
- Configurable parameters:
  - Confidence threshold
  - IoU threshold for NMS
  - Minimum and maximum box sizes
- Support for 80 COCO classes
- Debug logging for coordinate transformations

## Requirements

- iOS 15.0+
- Xcode 13.0+
- Swift 5.5+
- CoreML
- Vision framework

## Installation

1. Clone the repository
2. Open `yolodemo.xcodeproj` in Xcode
3. Build and run on a compatible iOS device

## Usage

1. Launch the app
2. Grant camera permissions when prompted
3. Point the camera at objects to detect
4. Adjust detection parameters using the sliders:
   - Confidence threshold (0.1-1.0)
   - IoU threshold (0.1-1.0)
   - Min box size (0-100)
   - Max box size (100-1000)

## Technical Details

### Model
- Uses YOLO model converted to CoreML format
- Input size: 640x640
- Output: Bounding boxes with confidence scores for 80 COCO classes

### Coordinate Handling
- Properly handles aspect ratio preservation
- Scales coordinates from model space (640x640) to screen space
- Accounts for image scaling and centering offsets

### Non-Maximum Suppression
- Eliminates overlapping detections
- Uses IoU (Intersection over Union) threshold
- Preserves highest confidence detection when overlaps occur

## Debug Output

The app includes detailed debug logging for:
- Model loading and configuration
- Coordinate transformations
- Detection processing
- NMS results

## License

This project is available under the MIT license. See the LICENSE file for more info. 