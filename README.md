# Logistic Regression from Scratch using NumPy

This repository contains a **from-scratch implementation of Logistic Regression** using **NumPy**. The project demonstrates how to implement the key components of Logistic Regression, including:

- Sigmoid activation function
- Binary Cross-Entropy loss
- Mini-Batch Gradient Descent
- Model evaluation using accuracy and confusion matrix

The model is trained on a dataset split into **Train (70%)**, **Validation (10%)**, and **Test (20%)** sets.

---

## Setup

Clone the repository and install the required packages:

---

## Data

Place your dataset CSV file inside the `data/` folder with the filename:

```
data/dataset.csv
```

The notebook expects the last column to be the target variable (`0` or `1`) and all other columns as features. The notebook automatically performs **Min-Max scaling** based on training data.

---

## Implementation

The implementation is contained in the Jupyter Notebook:

```
logistic_regression_from_scratch.ipynb
```

Key features:

* `CustomLogisticRegression` class for modularity
* Supports **full batch** and **mini-batch gradient descent**
* Weight initialization options: Normal, Uniform, or Zeros
* Prediction and evaluation functions with custom confusion matrix format `[[TP, FP], [FN, TN]]`

---

## Training and Evaluation

* **Full Batch Gradient Descent**: Trains on the entire training set each epoch
* **Mini-Batch Gradient Descent**: Trains on smaller batches (e.g., batch size = 64)
* Validation set is used to monitor performance during training
* Final evaluation is performed on the test set

Example:

```python
model = CustomLogisticRegression(n_features)
w, b, losses = model.train(X_train, y_train, batch_size=64, epochs=100, lr=1.0)
y_pred = model.predict(X_val)
accuracy, confusion_matrix = model.evaluate(y_val, y_pred)
```

---

## Results

After training, the notebook prints:

* **Validation Accuracy**
* **Confusion Matrix** in format `[[TP, FP], [FN, TN]]`
* **Final Test Set Accuracy**
* **Test Set Confusion Matrix**

---

## Visualization

Training loss per epoch is visualized using Matplotlib for both **full batch** and **mini-batch** training:

```python
plt.plot(losses_full, label='Full Batch Loss')
plt.plot(losses_mini, label='Mini Batch Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.show()
```

