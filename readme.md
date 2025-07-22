# Bodylandmarks & Object Detection & Kinematic Analysis in Physical Education Classes

This repository contains code and Jupyter notebooks for analyzing physical education class videos. It enables:  
- **Person (ID) detection and tracking**  
- **Body landmark detection** (major joints)  
- **Object detection** (e.g., javelin with YOLO)  
- **Kinematic analysis** of javelin throwing performance

## 📦 Features

- **Person detection:** Track and assign consistent IDs to individual students.
- **Pose estimation:** Detect key human body landmarks (ankle, knee, hip, wrist, elbow, shoulder, bilaterally).
- **Object detection:** Track objects, such as the javelin, using a pretrained YOLO model.
- **Kinematic analysis:** Quantify and visualize movement metrics using pose and object detection results.
- **Interactive tools:** Streamlit app for easy visualization of all outputs.

## 🗂️ Repository Structure

| File/Folder                                               | Description                                             |
|-----------------------------------------------------------|---------------------------------------------------------|
| `Person (ID) detection_YOLO 11.ipynb`                     | Detects and tracks students (ID assignment)             |
| `Body landmarks_YOLO 11 and RTM Pose.ipynb`               | Detects body landmarks (major joints)                   |
| `Javelin object detection_YOLO 11.ipynb`                  | Detects parts of the javelin using YOLO                 |
| `Merge body landmarks, Person (ID) detection, and object detection.ipynb` | Combines tracking and detection results     |
| `Kinematic analysis_javelin throwing.ipynb`               | Performs kinematic analysis for javelin throws          |
| `Post viewer_streamlit.py`                                | Streamlit app for interactive visualization             |
| `best.pt`                                                 | YOLO model weights (required for object detection)      |
| `README.md`                                               | This documentation file                                 |

## 🛠️ Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/lekuelc/bodylandmarks_objectdetection_kinematicanalysis.git
    cd bodylandmarks_objectdetection_kinematicanalysis
    ```

2. **Install dependencies:**  
   These are the primary Python packages you’ll need:
    - `opencv-python`
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `scipy`
    - `tqdm`
    - `ultralytics` (for YOLO)
    - `torch`
    - `streamlit`
    - `deeplabcut` (for pose estimation based on RTMPose)
    - `psutil`
    - `pyyaml`
    - `torchreid`  
   
   You can install them all at once:
    ```bash
    pip install opencv-python numpy pandas matplotlib scipy tqdm ultralytics torch streamlit deeplabcut psutil pyyaml torchreid
    ```

3. **Provide your own video/data files:**  
   Recordings of students performing motor skills (e.g., javelin throwing). For best results with javelin, record from a side view.

## ▶️ Usage

**Recommended analysis pipeline:**

1. **Person (ID) detection:**  
   Run `Person (ID) detection_YOLO 11.ipynb` to detect and track students.

2. **Body landmarks detection:**  
   Run `Body landmarks_YOLO 11 and RTM Pose.ipynb` to track major joints. *Requires bounding boxes from YOLO.*

3. **Object detection:**  
   Run `Javelin object detection_YOLO 11.ipynb` to detect javelin parts (requires trained YOLO weights in `best.pt`).

4. **Merge detections:**  
   Run `Merge body landmarks, Person (ID) detection, and object detection.ipynb` to combine all results into a unified dataset.

5. **Kinematic analysis:**  
   Run `Kinematic analysis_javelin throwing.ipynb` to extract and analyze movement features (arm extension, stepping pattern, timing of release, elbow dynamics).

- Steps 3 and 5 are specific to javelin throwing, but the workflow can be adapted for other skills and equipment (in case sports equipment or other materials are involved, by training new object detection models).

**Visualization:**  
Use the Streamlit app for interactive visualization:

    ```bash
    streamlit run Post viewer_streamlit.py
    ```

## 📋 Notes

- **Video data:** Users must provide their own video footage for analysis. Both group and single-student recordings are supported.
- **Camera perspective:** For analyzing javelin throwing, students must be filmed from the side for best results.
- **Support:** For questions, open an [issue](https://github.com/lekuelc/bodylandmarks_objectdetection_kinematicanalysis/issues) in this repository.

## 🙏 Contributions

- **Pull requests:** Contributions are welcome! If you have improvements or fixes, please submit a pull request.
- **Suggestions:** Ideas for new features or changes are also appreciated—feel free to open an issue.

---