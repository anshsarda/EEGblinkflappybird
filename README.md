# EEG Blink Controlled Flappy Bird

This project implements a simple **EEG-based brain–computer interface (BCI)** that allows a user to control Flappy Bird using **blink signals recorded from an OpenBCI Cyton board**.

The goal of the project was to design a full pipeline that:

1. Collects EEG data
2. Trains a classifier to detect blink events
3. Streams live EEG data
4. Uses classifier output to control a game in real time

\---

# System Pipeline

```
OpenBCI Cyton
   ↓
BrainFlow Data Stream
   ↓
EEG Preprocessing
   ↓
Blink Classifier
   ↓
BCI Controller
   ↓
Flappy Bird Game Control
```

The system was designed so that the **game and EEG processing are separated**, making it easier to modify the classifier without changing the game logic.

\---

# Project Structure

```
project/
│
├── flappy.py
│   Flappy Bird game implementation (pygame)
│   Accepts jump commands from keyboard or BCI controller
│
├── bci\_controller.py
│   Interface between EEG system and the game
│   Handles classifier output and cooldown logic
│
├── cyton\_stream.py
│   Streams EEG data from the OpenBCI Cyton using BrainFlow
│
├── preprocessing.py
│   EEG preprocessing and feature extraction
│
├── classifier\_interface.py
│   Wrapper used to run the trained classifier
│
├── run\_blink.py
│   Script used to collect blink training data
│
├── scripts/train\_trca.py
│   Training script used to train the blink classifier
│
└── model.joblib
    Saved classifier used for live inference
```

\---

# Environment Setup (Windows 11)

Create and activate a Python virtual environment:

```bash
pip install virtualenv
virtualenv pyenv --python=3.11.9
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
pyenv\\Scripts\\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Install **brainda** (used for EEG model utilities):

```bash
git clone https://github.com/TBC-TJU/brainda.git
cd brainda
pip install -r requirements.txt
pip install -e .
```

\---

# Data Collection

Blink data is collected using the script:

```bash
python run\_blink.py
```

Multiple runs can be recorded by changing the run number inside `run\_blink.py`.

The recorded data is later used to train the blink detection classifier.

\---

# Training the Classifier

After collecting blink data, train the classifier with:

```bash
python scripts/train\_trca.py
```

This produces a trained model that is saved and used during live gameplay.

\---

# Running the Game

Start the Flappy Bird game:

```bash
python flappy.py
```

Controls:

* **Spacebar** – manual jump
* **BCI blink detection** – jump triggered by EEG classifier

During gameplay the game loop repeatedly calls:

```
should\_jump()
```

The BCI controller returns `True` when a blink is detected, causing the bird to flap.

\---

# Real-Time Control Loop

```
Game Loop
   ↓
bci\_controller.should\_jump()
   ↓
Read EEG window (cyton\_stream.py)
   ↓
Preprocess signal (preprocessing.py)
   ↓
Run classifier (classifier\_interface.py)
   ↓
Return jump command
```

\---

# Summary

* EEG data collection using **OpenBCI Cyton**
* Blink signal classification
* Real-time EEG streaming
* Integration with a **pygame-based Flappy Bird game**
