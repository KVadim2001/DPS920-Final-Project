# Self-Driving Car Simulator Project

## Overview
This project implements a CNN model that autonomously drives a car in a simulator using behavioral cloning.  
The model learns steering angles from recorded driving data and predicts steering in real time.

---

## Project Structure

```
Final Project/
│
├── src/
│   ├── project.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── TestSimulation.py
│
├── dataset/
├── model.h5
├── package_list.txt
└── README.md
```

---

## Dataset

The dataset is not included in this repository because it contains a large number of image files.

The `driving_log.csv` file is tied to the `IMG/` folder, which contains all recorded driving images.  
Both must remain together for the project to work correctly.

To run this project:

1. Obtain the dataset (driving_log.csv + IMG folder)
2. Place it in the dataset folder:
```
dataset/
├── IMG/
└── driving_log.csv
```

---

## Dependencies

Install dependencies using:

`pip install -r package_list.txt`

It is recommended to use a virtual environment before installing dependencies.

---

## Training

Train the model: `python src/project.py`

This will train the model and save: `model.h5`

---

## Testing (Simulator)

1. Open simulator  
2. Run: `python src/TestSimulation.py`

3. Start autonomous mode

---

## Features

- NVIDIA CNN architecture  
- Data preprocessing  
- Data augmentation (flip, brightness, zoom, pan, rotation)  
- Real-time simulator driving  

---

## Author

Vadim Kurbanbakiyev  
Seneca Polytechnic
