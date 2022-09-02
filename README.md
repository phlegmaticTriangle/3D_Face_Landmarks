# 3D_Face_Landmarks

<p align="center" width="100">
    <img src="https://github.com/phlegmaticTriangle/3D_Face_Landmarks/blob/9c6c67462e0d711f6810a1c6b2c0648b6fe6ddfc/results/demo_result_left.png" width="30%">
    <img src="https://github.com/phlegmaticTriangle/3D_Face_Landmarks/blob/5795e091c2425fa50692b0dccd14bc901c0ead9a/results/demo_result_center.png" width="30%">
    <img src="https://github.com/phlegmaticTriangle/3D_Face_Landmarks/blob/7f952d2c40555de42544009d7f72302dc42e98a1/results/demo_result_right.png" width="30%">
</p>

## Introduction
This project locates five soft-tissue landmarks on 3D face scans. In particular, the pronasal (nose tip), endocanthions (inner eye corners), and cheilions (mouth corners) are located. Motivation for this project stems from the desire to improve the use of face landmarks for designing and visualizing dentures. 

## Requirements
This project runs on Python3. The following libraries are also necessary:
- Joblib
- NumPy
- pandas
- PyVista
- scikit-learn
- SciPy

## Usage
1. Clone this repo
```
git clone https://github.com/phlegmaticTriangle/3D_Face_Landmarks
cd 3D_Face_Landmarks
```
2. Try out the demo
```
python3 demo.py
```

Use `--mesh` to specify the path to the face scan that is to be landmarked, and use the `--reorient` flag to specify whether the face scan needs to be reoriented before it is landmarked. (Running the code with `--reorient` is currently much slower than running without it.)

Running `python3 demo.py` should create a rendering of a 3D face scan and a file named landmarks.txt in the results directory that contains the coordinates of the five landmarks. A portrait view of the rendered 3D face scan is shown below:

<p align="center" width="100%">
    <img src="https://github.com/phlegmaticTriangle/3D_Face_Landmarks/blob/5795e091c2425fa50692b0dccd14bc901c0ead9a/results/demo_result_center.png" width="40%">
</p>

## Acknowledgement
The demo.stl face scan was generated using [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2).

This research was conducted at the [REU in Software and Data Analytics at East Carolina University](http://www.cs.ecu.edu/reu/) and was primarily supported by NSF REU grant #2050883.

I would like to given special thanks to my mentor Dr. Nic Herndon. His guidance and expertise was essential to this project.
