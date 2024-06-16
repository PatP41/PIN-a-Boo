# PIN-a-Boo
This repository contains the code accompanying the paper "PIN-a-Boo: Leveraging segmentation and hand skeleton tracking frameworks to discern smartphone PIN entry from a video feed".

# Repository Contents

- **Determining the region of interest**:
  - [ROI with Padding](yolo_roi/generate_roi-better_error-padding.py): Determining the Region of Interest based on the smart phone's location using object detection via a general, pre-trainted [YOLOv8](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) model. Uses padding to avoid changes in windows size at the cost of slightly lower speed.
  - [ROI without Padding](yolo_roi/generate_roi-better_error_no-padding.py): Same as above but without padding (slightly faster).
  - [Prior version](yolo_roi/generate_roi-clumsy_error.py): Earlier version with some errors.
 
- **Movement path creation via finger tracking**:
  - [Main Script](mediapipe_fingertracking/combination_path_roi_v2.py): Combination of the the prior step (identifying the region of interest with YOLO), finger movement tracking using [Mediapipe](https://chuoling.github.io/mediapipe/) and drawing of the finger movement path.
  - [Prior versions](mediapipe_fingertracking/prior_versions): Includes multiple earlier versions (basic mediapipe usage, path creation).

- **Rotation and Transformation**:

  Includes multiple scripts that try to transform the finger movement path from the camera's perspective into the victim's perspective using a rotation matrix calculated in each frame based on the angles determined via the edges of the phone. Manual adjustments still necessary and not without errors.
  - [PoC transformation](rotation-transformation/transform-mediapipe-3d.py): PoC of a live transformation of the finger path using three rotational angles.
  - [2-angle rotation matrix](rotation-transformation/rotation_matrix-2_angles.py): Calculating a rotational matrix using two angles (smartphone edges).
  - [Angle calculation](rotation-transformation/edge_detection-yolo.py): Calculating rotational angles using the long and short edge of the smart phone.
  - [Other scripts](rotation-transformation/other): Contains multiple scripts to compare different line detectors.

- **Decision making**:
  - [Mapping](clustering/mapping.py): Maps the normalised finger path (in a list) to a smart phone's number block using clustering.
  - [Alternative visuals](clustering/mapping-different_visuals.py): Produces output in a different visual style.
 
- **Other scripts**:
  - [Absolute difference](other_pys/mapping.py): Uses contour detection to show the absolute difference between two frames.

- **Assets**:
  - [Development](assets/test): Contains pictures and video used for tests during the development.
  - [Evaluation](assets/evaluation): Contains videos used for the evaluation.
  - [YOLOv8](yolov8n.pt): Pre-trained YOLO model by [Ultralytics](https://github.com/ultralytics).

# Installation and Usage
The python version and dependencies are quite specific and didn't work on all systems (ARM-Chip Macs).

1. Install `pipenv`
```
pip install pipenv
```

2. Navigate to project path
```
cd /path/to/the/project
```

3. Install dependencies from Pipfile
```
pipenv install
```

4. Activate the virtual environment
```
pipenv shell
```

5. Run the file
```
python folder/whatever.py
```
