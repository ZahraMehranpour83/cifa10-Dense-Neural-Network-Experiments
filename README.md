# cifa10-Dense Neural Network Experiments

---

CIFAR-10 Dense Neural Network Experiments

This repository contains three separate Jupyter Notebooks exploring different Dense Neural Network (fully-connected) architectures for the CIFAR-10 image classification task.
Each notebook progressively modifies the model to test the effect of architectural changes, optimization strategies, and regularization techniques.


---

📂 Notebooks Overview

Notebook	Description	Main Features

01_CIFAR10_BasicDense.ipynb	A simple dense network for CIFAR-10 classification.	- Flatten + Rescaling layer<br>- Multiple Dense layers with ReLU activation<br>- Adam optimizer, default learning rate<br>- Evaluates training & test accuracy
02_CIFAR10_DeepDense.ipynb	A deeper and wider dense network for improved capacity.	- Larger Dense layers (1024, 512, 256 neurons)<br>- EarlyStopping callback (patience=5)<br>- Adam optimizer, default learning rate<br>- Focused on increasing network depth/width
03_CIFAR10_RegularizedDense.ipynb	A more robust architecture with regularization and checkpoints.	- Batch Normalization layers<br>- Dropout layers (0.2–0.4)<br>- Reduced learning rate (0.0005)<br>- EarlyStopping + ModelCheckpoint callbacks<br>- Best model loading before evaluation



---

🧪 Experiment Goals

The three models were designed to test how architecture depth, width, and regularization affect CIFAR-10 classification performance:

1. Model 1: Baseline — minimal preprocessing, no regularization.


2. Model 2: Increased model capacity — deeper & wider Dense layers.


3. Model 3: Balanced complexity — added Batch Normalization, Dropout, tuned learning rate, and checkpointing.




---

📊 Performance Comparison

Model	Best Train Accuracy	Best Validation Accuracy	Test Accuracy

Basic Dense	~High 80% range	~High 70% range	Lower than Val
Deep Dense	Improved train acc, slight overfitting	Slight Val improvement	Similar to Val
Regularized Dense	Balanced high Val accuracy	Best generalization	Highest Test Accuracy


(Exact values depend on random initialization and training runs — see each notebook output for details.)


---

⚙️ Requirements

Python 3.8+

TensorFlow 2.x

NumPy

Matplotlib


Install dependencies:

pip install tensorflow numpy matplotlib


---

🚀 Usage

Clone the repo:

git clone https://github.com/yourusername/cifar10-dense-experiments.git
cd cifar10-dense-experiments

Run any notebook:

jupyter notebook 01_CIFAR10_BasicDense.ipynb


---

📌 Notes

All models use CIFAR-10 dataset directly from tf.keras.datasets.

The focus is fully-connected (Dense) layers, not CNNs.

Regularization (BatchNorm + Dropout) in Model 3 helps reduce overfitting.

Learning rate tuning significantly affects convergence speed and stability.



---
