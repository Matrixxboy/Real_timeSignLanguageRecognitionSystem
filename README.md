## Author 
[@Matrixxboy](https://github.com/Matrixxboy)

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
# Hand Sign Recognition

**Description**

This project implements hand sign recognition using Python and deep learning techniques. It allows users to capture hand gestures and classifies them using a Convolutional Neural Network (CNN) trained with TensorFlow and Keras.


### Key Technologies

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow, Keras
- **Image Processing**: OpenCV, cvzone
- **Computer Vision**: MediaPipe (optional)
- **Web Framework (Optional)**: Flask (for browser-based deployment)

### Structure

- **final_data_collection.py**: Captures hand sign images for training and prediction.
- **main.py**: Predicts hand signs from new images or a webcam stream.
- **AtoZ.h5**: Saves the trained CNN model for future use.
- **README.md**: This file (you're editing it now!).
- **(Optional) main.py**: Flask application code (if deploying as a web app).

## Setup

Important Notes (Recommended Versions and Setup):
For optimal performance and to avoid potential compatibility issues, it is highly recommended to use the following software 

### versions:
- ![Python 3.11.0](https://img.shields.io/badge/python-3.11.0-blue)
- ![TensorFlow 2.12.0](https://img.shields.io/badge/tensorflow-2.12.0-orange)
- ![Keras 2.12.0](https://img.shields.io/badge/keras-2.12.0-blueviolet)
- ![NumPy 1.23.5](https://img.shields.io/badge/numpy-1.23.5-blue)
- ![opencv-python 4.10.0.84](https://img.shields.io/badge/opencv--python-4.10.0.84-green)
- ![Flask 2.3.2](https://img.shields.io/badge/flask-2.3.2-blue)
- ![Jinja2 3.1.2](https://img.shields.io/badge/jinja2-3.1.2-blueviolet)

It's also strongly advised to create a virtual environment to manage project dependencies and prevent conflicts with other Python projects on your system. This isolates the project's required libraries. You can create one using these commands (in your project directory):

```bash
python3 -m venv .venv       # Create a virtual environment (named .venv)
```
```bash
source .venv/bin/activate  # Activate the virtual environment (Linux/macOS)
```
```bash
.venv\Scripts\activate      # Activate the virtual environment (Windows)
```
After activating the virtual environment, then install the requirements:

```bash
pip install -r requirements.txt
```
This guide explains how to set up and run a hand sign recognition project using Python libraries.

 
### Prerequisites:
- **Terminal**: Open a terminal window on your computer (Command Prompt on Windows, Terminal on Mac/Linux).
- **Python and Git**: Ensure you have Python 3.11.0 and Git installed on your system. You can download them from their respective official websites.

### Steps:
1.	**Clone the Repository**:
- Replace <your_username> with your actual GitHub username in the following command:

```bash
git clone https://github.com/Matrixxboy/Real_timeSignLanguageRecognitionSystem.git
```
This will download the project files from your GitHub repository to your local machine.

2.	**Install Dependencies**:
- Navigate to the project directory using the following command:
```bash
cd Real_timeSignLanguageRecognitionSystem
```

- If you haven't already, make sure you've activated your virtual environment (see "Important Notes" above).
- Install all the required libraries listed in the requirements.txt file. This file specifies the versions needed for smooth project operation. Here's the installation command:
```bash
pip install -r requirements.txt
```

3.	**Capture Training Data**:
- Run the final_data_collection.py script to collect images of different hand signs for training the model.
```bash
python final_data_collection.py
```
- Press `S` key to save the frame(square image).
- Press `Q` key to terminate.

This script will likely guide you through performing specific hand gestures in front of your webcam. The captured images will be saved in separate folders for each sign class, making it easier to train the model on each sign. 

You can modify the following code to save captured data to custom folders.
``` bash
folder = "Data/Alphabates/S"
```

4.	**Train the Model**:
- Train the Convolutional Neural Network (CNN) model on the captured images using the [Trainable Machine](https://teachablemachine.withgoogle.com/).

5.	**Optional**: Run the Flask App (Web Interface):
- If you want to interact with the model in a web browser, run the Flask application using main.py.

```bash
python main.py
```

This will launch a web server(http://127.0.0.1:5000/), allowing you to upload images or use your webcam for real-time hand sign prediction.

6.	**Example Usage (Prediction)**:
Once you've trained the model, you can use the main.py script for hand sign detection:

Prediction from an image:

```bash
prediction, index = classifier.getPrediction(img_white, draw=False)  # Classify the hand gesture
label = labels[index]  # Get the label corresponding to the predicted index
print(f"Detection: Label Index = {index}, Label Name = {label}")

detector = HandDetector(maxHands=1)  # Initialize hand detector
classifier = Classifier("Final_Model/AtoZ.h5", "Final_Model/AtoZ.txt") # Initialize classifier
with open("Final_Model/AtoZ.txt", 'r') as f: #open   file
    labels = [line.strip() for line in f] #read labels file
```

Prediction from webcam:
 ```bash
detector = HandDetector(maxHands=1)  # Initialize hand detector
classifier = Classifier("Final_Model/AtoZ.h5", "Final_Model/AtoZ.txt") # Initialize classifier
with open("Final_Model/AtoZ.txt", 'r') as f: #open   file
    labels = [line.strip() for line in f] #read labels file

```

This will activate your webcam and display the predicted hand sign in real-time.

 
### Key Changes:
- "Important Notes" at the very top: This section now highlights the recommended versions and, most importantly, the use of a virtual environment. This is crucial for avoiding many common setup problems.
- **Virtual Environment Instructions**: Added clear instructions on how to create and activate a virtual environment on different operating systems.
- **Emphasis on Virtual Environment**: Added reminders to activate the virtual environment before installing requirements.

## Explanation of Technologies

- **TensorFlow**: A powerful open-source library for numerical computation and machine learning. It provides low-level APIs for building custom neural networks (for advanced users) and higher-level APIs like Keras (which you're using) for easier model development.

- **Keras**: A high-level API built on top of TensorFlow, designed for user-friendliness and rapid prototyping of neural networks. It provides pre-built layers (like convolutional and pooling layers) and simplifies model building.

- **Convolutional Neural Networks (CNNs)**: A type of neural network particularly adept at processing image data. CNNs learn features directly from images by applying filters (kernels) to extract patterns and progressively build higher-level representations.

<p align="center">
  <img width="460" height="250" src="https://github.com/Matrixxboy/Real_timeSignLanguageRecognitionSystem/blob/main/images/Convolutional%20Neural%20Networks_CNNs.jpg?raw=true">
</p>

# Neural Networks: The Core of this Project

**Description**

A neural network is a computational model inspired by the structure and function of biological neural networks (the brain). It consists of interconnected nodes (neurons) organized in layers that process information.

### Basic Components:

1. **Neurons (Nodes)**: The basic units of a neural network. Each neuron receives input signals, performs a simple computation, and produces an output signal.

<p align="center">
  <img width="460" height="280" src="https://github.com/Matrixxboy/Real_timeSignLanguageRecognitionSystem/blob/main/images/neurons(nodes).jpg?raw=true">
</p>

  - **Inputs (x1, x2, x3)**: Values received from other neurons or external sources.
  - **Weights (w1, w2, w3)**: Represent the strength of each input connection.
  - **Summation (∑)**: The weighted inputs are summed together.
  - **Activation Function (f)**: Introduces non-linearity, allowing the neuron to make decisions.
  - **Output (y)**: The result of the neuron's computation.


2. **Connections (Weights)**: Connections between neurons have associated weights that determine the strength of the connection. During training, these weights are adjusted to improve the network's performance.

3. **Layers**: Neurons are organized into layers:
  - **Input Layer**: Receives the initial data (e.g., pixel values of an image).
  - **Hidden Layers**: Intermediate layers that perform computations and extract features from the input. A network can have multiple hidden layers (deep learning).
  - **Output Layer**: Produces the final result (e.g., probabilities for different hand sign classes).

<p align="center">
  <img width="460" height="370" src="https://github.com/Matrixxboy/Real_timeSignLanguageRecognitionSystem/blob/main/images/layers.png?raw=true">
</p>
## How Neural Networks Learn (Training)

1.	**Forward Propagation**: Input data is passed through the network, layer by layer, until it reaches the output layer.
2.	**Loss Function**: A loss function measures the difference between the network's predictions and the actual target values (labels).
3.	**Backpropagation**: The error (loss) is propagated back through the network, and the weights are adjusted to minimize the error. This is done using optimization algorithms like gradient descent.
4.	**Activation Functions**: These functions introduce non-linearity into the network, allowing it to learn complex patterns. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh.

## Types of Neural Networks (Relevant to Image Recognition)

- Convolutional Neural Networks (CNNs): Designed specifically for image data. They use convolutional layers to extract local features from images, making them very effective for image recognition tasks. Your project uses CNNs.
 
  - Convolutional Layers: Apply filters (kernels) to the input image to detect features like edges, corners, and textures.
  - Pooling Layers: Reduce the spatial dimensions of the feature maps, reducing the number of parameters and computations.
  - Fully Connected Layers: Perform high-level reasoning and classification based on the extracted features.

<p align="center">
  <img width="500" height="270" src="https://github.com/Matrixxboy/Real_timeSignLanguageRecognitionSystem/blob/main/images/The-structure-of-a-CNN-consisting-of-convolutional-pooling-and-fully-connected-layers.png?raw=true">
</p>

## Why Square Images for Neural Networks (Especially CNNs)?

While not strictly required in all cases, square images are often preferred for several reasons, particularly in the context of CNNs and when using tools like TensorFlow/Keras:
1.	**Consistency in Convolutional Operations**: Convolutional layers use square filters (kernels) to scan the image. Using square input images ensures that the filters operate consistently across both dimensions. This simplifies the computations and the architecture of the network.

 <p align="center">
  <img width="460" height="400" src="https://github.com/Matrixxboy/Real_timeSignLanguageRecognitionSystem/blob/main/images/Screenshot%202024-1.png?raw=true">
</p>

2.	**Simplifying Resizing/Preprocessing**: If you have images of varying aspect ratios, you need to resize or pad them to a consistent size before feeding them to the network. Resizing to a square shape is the simplest and most common approach. It avoids distortions that might occur when resizing to non-square dimensions.

 <p align="center">
  <img width="260" height="260" src="https://github.com/Matrixxboy/Real_timeSignLanguageRecognitionSystem/blob/main/images/Screenshot%202024-12.png?raw=true">
</p>


3.	**Compatibility with Pre-trained Models (Not Applicable)**:Since this project uses a custom-built model, pre-trained model compatibility is not a concern. Square inputs are still preferred for other reasons.
Choose the phrasing that best fits the overall tone and level of detail in your README. The key is to clearly state that pre-trained model compatibility is not relevant to your project because you're using a model you created yourself.

4.	**Avoiding Distortion**: Non-square resizing can distort the image, making objects appear stretched or compressed. This can negatively impact the performance of the model, especially if shape is a crucial feature for recognition.

<p align="center">
  <img width="260" height="260" src="https://github.com/Matrixxboy/Real_timeSignLanguageRecognitionSystem/blob/main/images/Screenshot%202024-3.png?raw=true">
</p>

## Training Process (Trainable Machine)

1.	**Capture Images**: 
- Use final_data_collection.py to capture hand signs from different angles and lighting conditions to create a diverse dataset.
- Ensure images are square to comply with the CNN's input requirements. You can resize captured images or crop them to a square format.
2.	**Data Preprocessing**: 
- Preprocess the captured images before feeding them to the CNN. This might involve: 
    -  Resizing to a fixed size
    - Normalization (scaling pixel values)
    - Data augmentation (random flipping, rotations, etc.) to increase dataset variety and improve model generalization.
3.	**Model Training**: 
- Train the CNN using the pre-processed dataset on Trainable Machine. This involves feeding images and corresponding labels (sign classes) into the model, iteratively adjusting weights to improve prediction accuracy. Monitor training progress on a validation set to avoid overfitting.

## prediction

1. **Load Trained Model**:
- This step retrieves the model.h5 file saved during the training process. This file contains all the learned weights and biases of your CNN, essentially the "knowledge" it has acquired about hand signs.
**Imagine it this way**: You trained your model on a bunch of hand sign images, just like showing a student a bunch of flashcards. Now, you have a "teacher" (the model) who has learned to recognize different hand signs.
- The main.py script acts like a "test giver" and needs the "teacher" (model) present to assess new images.

2. **Process New Image**:
- This stage takes a new image (potentially captured from a webcam or uploaded by a user) and prepares it for the model's input. 
    - **Preprocessing**: There are two main things that might be done here: 
        - Resizing: The image might be resized to match the size the model was trained on. Just like fitting flashcards to a specific size for the student to see clearly.
        - Normalization: The pixel values in the image might be scaled or adjusted to a standard range. This helps the model interpret the image's intensity levels consistently.
    - **Think of it this way**: You want the "test" for the student (model) to be presented in the same format as the "flashcards" (training data) used before.


3. **Prediction**:
- Now comes the core prediction part: 
    - **Feeding the Image**: The pre-processed image is fed as input to the loaded CNN model. Just like giving the student a new image to identify.
    - **Obtaining Output**: The model processes the image through its layers and generates an output. This output is typically a set of probabilities, one for each potential hand sign class.
    - **Consider this**: The "teacher" (model) analyses the new image based on its learned knowledge.
    - **Probabilities as Scores**: The model doesn't give a definitive answer, but rather assigns a score (probability) to each hand sign class. Imagine the student giving a confidence score for each possible answer. A higher score indicates the model is more confident about that class.


4. **Making the Decision (Choosing the Class)**:
- The main.py script needs to interpret the model's output (probabilities) to make a final prediction about the hand sign in the image. 
    - Logic for Choosing: This often involves selecting the class with the highest probability. Essentially, the script picks the answer the "teacher" (model) seems most confident about.
    - Deciding the Winner: Based on the scores, the script decides which hand sign the image most likely represents.

## Flask Application (main.py)


- What is Flask? Flask is a lightweight web framework for Python that allows you to build web applications with a relatively simple and flexible approach. Think of it as a set of tools for creating interactive web interfaces.
- Creating the Flask App: The main.py script in your project creates a Flask application. This application essentially defines how the web interface will function and what happens when users interact with it.
- Routes: Routes are like pathways within the web application. They map specific URLs to corresponding functions that handle user requests. For example, you might have a route for uploading an image (/upload) or for accessing a live webcam feed (/webcam).

## Image Upload or Webcam Stream Processing

- When a user interacts with your web interface (e.g., uploads an image or accesses the webcam feed), the Flask app triggers the appropriate route function defined in main.py.
- **Processing**: This function utilizes the functionalities of the main.py script. It doesn't directly call main.py as a separate program, but rather imports it as a Python module and uses its functions within the Flask app.
- **Prediction**: The route function in main.py would likely: 
    - Retrieve the image data: If the user uploaded an image, this function would access the uploaded file and convert it into a format suitable for the model (e.g., a NumPy array). If using a webcam, it would capture frames from the webcam in real-time.
    - Call predict.py functions: The Flask app would use functions defined in main.py to preprocess the image (resize, normalize), feed it to the loaded model, and interpret the output probabilities.
- **Essentially**, the Flask app acts as a bridge: It takes user interaction (uploading images, accessing webcam), utilizes the prediction logic from main.py, and presents the results (predicted hand sign) back to the user in a web browser.

### Deployment (Optional)

- Deploying the Flask app to a hosting platform allows you to make your hand sign recognition project accessible online. Anyone with an internet connection could then access your web interface through a specific URL.
- Popular hosting platforms for Flask applications include Heroku, PythonAnywhere, and AWS Elastic Beanstalk. Each platform has its own deployment process and pricing structure.


### Benefits of Flask Integration

- **User Interaction**: Flask allows you to create a user-friendly interface, making your model accessible to people without needing to run Python scripts directly.
- **Real-time Predictions**: With webcam integration, your web app can analyse hand signs in real-time, providing a more interactive experience.
- **Accessibility**: Deployment lets you share your project with others, allowing them to test your model or even use it for their own purposes.
