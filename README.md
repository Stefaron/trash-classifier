# Trash Classification Model with ResNet50

This repository contains a Python implementation of a Trash Classification model based on ResNet50. The project involves preprocessing, fine-tuning a pre-trained ResNet50 model, and training on the TrashNet dataset. Below is a detailed guide on the project's setup and usage.

---

## Dataset

The dataset used in this project is the [TrashNet Dataset](https://github.com/garythung/trashnet). It is automatically downloaded and extracted by the script.

### Dataset URL:
`https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip`

The dataset is split into six categories:
1. Glass
2. Paper
3. Cardboard
4. Plastic
5. Metal
6. Trash

---

## Project Structure

```
.
├── data/                        # Directory for the dataset
├── best_model.keras             # Checkpoint of the best model during training
├── train.py                     # Main script for training the model
└── README.md                    # Project documentation
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have TensorFlow installed:
   ```bash
   pip install tensorflow
   ```

---

## Model Development Pipeline

### 1. Data Preparation

- The dataset is downloaded and extracted automatically into the `./data` directory.
- Data is processed using `pandas` to create a DataFrame containing file paths and labels.
- The dataset is split into training and validation sets using `train_test_split` from `sklearn`.

### 2. Data Augmentation

The training data is augmented using `ImageDataGenerator` with the following techniques:
- Rotation: up to 60 degrees
- Shifting: up to 15% of width and height
- Zoom: up to 20%
- Shear, brightness adjustment, horizontal/vertical flipping
- Preprocessing with ResNet50's `preprocess_input` function

### 3. Model Architecture

The model is built using the pre-trained ResNet50 as the backbone with the following modifications:
- Global Average Pooling Layer
- Dropout Layer (50%)
- Dense Output Layer with 6 classes (softmax activation)

### 4. Training Details

- Optimizer: Adam (Learning rate: 0.0001)
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Class Weights: Computed using `sklearn.utils.compute_class_weight`

#### Callbacks:
- ReduceLROnPlateau: Reduce learning rate on plateau
- EarlyStopping: Stop training early if validation loss doesn't improve
- ModelCheckpoint: Save the best model

---

## Training the Model

Run the script to start training:
```bash
python train.py
```
The script will train the model and save the best version to `best_model.keras`.

---

## Results

The training process logs metrics such as accuracy and loss for both training and validation datasets. Additionally, the script uses callbacks to ensure the best model is saved based on validation loss.

---

## Future Work

- Experiment with other pre-trained models (e.g., Inception, EfficientNet)
- Fine-tune hyperparameters for improved accuracy
- Evaluate the model on a separate test set
- Deploy the model using Flask/Django for real-world applications

---

## Requirements

- Python 3.7+
- TensorFlow
- Pandas
- NumPy
- scikit-learn
- Matplotlib (optional for visualizing results)

Install all required packages using the following command:
```bash
pip install tensorflow pandas numpy scikit-learn
```

---

## References

- [TrashNet Dataset](https://github.com/garythung/trashnet)
- [ResNet50 Documentation](https://keras.io/api/applications/resnet/#resnet50-function)
