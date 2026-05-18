# Early Failure Prediction System Using BiLSTM

## Overview

This project presents a deep learning-based Early Failure Prediction System for system log analysis using a Bidirectional Long Short-Term Memory (BiLSTM) architecture. The model analyzes sequential HDFS log events and forecasts the likelihood of system failure within the next *k* events.

Unlike traditional anomaly detection systems that identify failures only after they occur, this project focuses on proactive failure forecasting by learning temporal patterns that precede anomalies.

---

# Key Features

- BiLSTM-based sequential log modeling
- Early failure forecasting using future event windows
- Sliding window sequence generation
- Weighted imbalance handling
- Focal Loss for rare anomaly learning
- Threshold optimization using validation F1-score
- Dynamic forecasting outputs:

```text
x% chance of system failure likely in next k events
```

- End-to-end automated training and testing pipeline

---

# Dataset

## HDFS Log Dataset

The project uses the HDFS Log Dataset obtained from LogHub and accessed through Hugging Face.

The dataset contains:
- structured log event sequences
- block execution traces
- anomaly labels

Each log sequence represents ordered system events occurring during block execution in a distributed Hadoop environment.

---

# Project Architecture

## Pipeline Overview

```text
Raw HDFS Logs
      ↓
Log Parsing
      ↓
Event Sequence Extraction
      ↓
Feature Engineering
      ↓
Sliding Window Forecast Generation
      ↓
Sequence Padding
      ↓
BiLSTM Training
      ↓
Threshold Optimization
      ↓
Failure Forecasting
```

---

# Feature Engineering

The preprocessing pipeline includes:

- Event sequence extraction
- Sequence length computation
- Event frequency counting
- Error count extraction
- Forecasting window generation
- Sequence padding

These transformations convert raw system logs into structured sequential inputs suitable for BiLSTM-based forecasting.

---

# Model Architecture

The model consists of:

- Embedding Layer
- Bidirectional LSTM (BiLSTM)
- Max Pooling Layer
- Dropout Layer
- Fully Connected Classification Layer

The BiLSTM processes log sequences in both forward and backward directions to better capture temporal dependencies and failure escalation patterns.

---

# Class Imbalance Handling

The dataset is highly imbalanced, with significantly more normal sequences than failure sequences.

To address this:

- Weighted learning mechanisms were used
- Focal Loss was implemented
- Threshold optimization was performed using validation F1-score

This improves anomaly sensitivity and minority-class learning.

---

# Forecasting Strategy

Instead of directly classifying current anomalies, the model predicts:

```text
whether system failure is likely within the next k events
```

This is achieved using:
- sliding forecasting windows
- future-window labeling

---

# Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

Example performance:

| Metric | Value |
|---|---|
| Accuracy | 0.85 |
| Precision | 0.81 |
| Recall | 0.87 |
| F1 Score | 0.84 |

---

# Project Structure

```text
├── dataset_loader.py
├── log_parser.py
├── sequence_builder.py
├── model.py
├── utils.py
├── train_model.py
├── test_model.py
├── architecture.jpeg
├── best_threshold.json
└── README.md
```

---

# Installation

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn datasets tensorflow
```

---

# Training

Run:

```bash
python train_model.py
```

This performs:
- dataset loading
- preprocessing
- forecasting window generation
- BiLSTM training
- threshold tuning
- model evaluation

---

# Testing and Forecasting

Run:

```bash
python test_model.py
```

Example output:

```text
91.24% chance of system failure likely in next 5 events
```

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- TensorFlow (sequence padding utilities)
- Hugging Face Datasets

---

# Applications

- Predictive maintenance
- Cloud infrastructure monitoring
- Distributed system reliability
- Failure forecasting
- Intelligent anomaly detection

---

# Future Improvements

- Attention-based Transformers
- Real-time streaming inference
- Explainable AI for anomaly interpretation
- Multi-horizon forecasting
- Hybrid CNN-BiLSTM architectures

---

# Authors
Aarav Jhawar,
Dhruv Nyati,
Phani Gande
