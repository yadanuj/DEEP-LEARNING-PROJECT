# DEEP-LEARNING-PROJECT
*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: ANUJ YADAV

*INTERN ID*: CT04DH1868

*DOMAIN*: DATA SCIENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

# Data-Science-Project-Task-2

🚀 Fashion MNIST Classification using CNN - TensorFlow



📌 Overview
This project showcases a complete deep learning pipeline for image classification using the Fashion MNIST dataset. It utilizes Convolutional Neural Networks (CNN) built with TensorFlow and Keras to accurately classify fashion items into 10 distinct categories. The project includes data preprocessing, model building, training, evaluation, and visualization of results through metrics and sample predictions.

🎯 Objective
The goal of this project is to:

Implement a CNN from scratch using TensorFlow/Keras.

Train the model on Fashion MNIST, a dataset of grayscale clothing images.

Evaluate model performance through metrics and visualizations.

Predict and visualize sample outputs for qualitative analysis.

🧰 Technologies Used
Tool	              
Python 3.x	          
TensorFlow 2.x	      
Keras	              
NumPy	              
Matplotlib	          
Seaborn	              
Scikit-Learn	      

🖼️ Dataset: Fashion MNIST
Fashion MNIST is a dataset of 28x28 grayscale images of 70,000 fashion products from 10 categories, with 7,000 images per category.

Class Labels
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot

⚙️ Workflow
1️⃣ Load & Preprocess Data
Dataset loaded from tf.keras.datasets.fashion_mnist.

Images normalized to range [0, 1] for better training.

Reshaped to (28, 28, 1) to match CNN input format.

2️⃣ Build CNN Model
Architecture:

2 Convolutional layers with ReLU activation.

MaxPooling layers to reduce dimensionality.

Flattening followed by Dense layers.

Final Dense layer with softmax for 10-class classification.

Compiled with Adam optimizer and sparse_categorical_crossentropy loss.

3️⃣ Train the Model
Trained over 5 epochs with validation data.

Metrics tracked: Accuracy and Loss for both training and validation.

4️⃣ Evaluate Performance
Plotted accuracy and loss curves.

Generated confusion matrix using Scikit-Learn and visualized it with Seaborn.

5️⃣ Predict and Visualize Results
Displayed sample predictions with true labels.

Correct predictions shown in green, incorrect ones in red.

💻 Sample Code Highlights
```# Load and preprocess
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

📊 Results
Final Accuracy: Achieved high accuracy on both training and validation datasets.

Confusion Matrix: Helped identify model strengths and weaknesses in classifying similar items.

Sample Predictions: Visual confirmation of correct and incorrect classifications.

📈 Future Improvements

Add Dropout layers for regularization.
Train for more epochs to improve accuracy.
Explore data augmentation to enhance robustness.
Deploy model using Flask or Streamlit.

#Output: 
<img width="1908" height="861" alt="Image" src="https://github.com/user-attachments/assets/ff60b504-97de-4b46-963d-f7d06df06ff3" />
<img width="1541" height="643" alt="Image" src="https://github.com/user-attachments/assets/633073b8-fbd6-4ac9-80a3-2be4929304f7" />
<img width="1052" height="755" alt="Image" src="https://github.com/user-attachments/assets/59b9d047-cbea-4ce9-a518-7d571dc7c2a9" />
<img width="1736" height="1065" alt="Image" src="https://github.com/user-attachments/assets/6adf8f31-2ac7-4d51-af5d-5e4451b8989d" />
