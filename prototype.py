import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import gradio as gr

# Function to extract landmarks from the static image using Mediapipe


def extract_landmarks(image):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return hand_landmarks.landmark
    else:
        return None

# Function to preprocess the landmarks by multiplying by image width and height


def preprocess_landmarks(landmarks, image_width, image_height):
    preprocessed_landmarks = []
    for landmark in landmarks:
        preprocessed_landmarks.append(
            (landmark.x * image_width, landmark.y * image_height, landmark.z))
    return preprocessed_landmarks

# Function to predict hand gestures using your custom model


def predict_hand_gesture(landmarks):
    gestures = ["Rock", "Paper", "Scissors"]

    keras_model = tf.keras.models.load_model("./model1")

    
    prediction = keras_model.predict(landmarks)
    predicted_class = np.argmax(prediction)
    gesture_label = gestures[predicted_class]
    return gesture_label

# Gradio interface function


def classify_hand_gesture(image):
    image_width, image_height, _ = image.shape
    landmarks = extract_landmarks(image)
    if landmarks:
        preprocessed_landmarks = preprocess_landmarks(
            landmarks, image_width, image_height)
        prediction = predict_hand_gesture(preprocessed_landmarks)
        return prediction
    else:
        return "No hand detected in the image."


# Gradio interface setup
iface = gr.Interface(
    fn=classify_hand_gesture,
    inputs=gr.inputs.Image(),  # Use Gradio's Image input to handle uploaded images
    outputs=gr.outputs.Textbox(),
    live=True,
    capture_session=True
)

# Launch Gradio interface
iface.launch(inline=False)
