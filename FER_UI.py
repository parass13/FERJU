import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import mysql.connector
import hashlib
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

# Database connection functions
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="FER"
    )

def create_usertable():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users_table (
            username VARCHAR(50) PRIMARY KEY,
            password VARCHAR(255)
        )
    """)
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users_table (username, password) VALUES (%s, %s)", (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users_table WHERE username = %s AND password = %s", (username, password))
    data = cursor.fetchall()
    conn.close()
    return data

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Load pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:\\Users\\Paras Sharma\\OneDrive\\Desktop\\ExpressionData\\improved_cnn_emotion_model.h5")

model = load_model()

def predict_expression(image):
    image = image.resize((96, 96))  # Ensure image matches model input size
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)[0]
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return {classes[i]: prediction[i] for i in range(len(classes))}

# Webcam video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        resized_img = cv2.resize(img, (96, 96)) / 255.0
        img_array = np.expand_dims(resized_img, axis=0)
        predictions = model.predict(img_array)[0]
        classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        label = classes[np.argmax(predictions)]
        cv2.putText(img, f"Emotion: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

# Custom CSS for UI Enhancements (Modern aesthetic)
st.markdown("""
    <style>
    body {
        background-color: #f4f4f9;
        font-family: 'Roboto', sans-serif;
        color: #333333;
    }
    .main {
        background-color: #f4f4f9;
        font-family: 'Roboto', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        font-size: 14px;
    }
    .stTextInput>label, .stTextArea>label, .stSelectbox>label {
        color: #2c3e50;
    }
    .stTextInput>div>div>input, .stTextArea>textarea, .stSelectbox>div>div>input {
        color: #333333;
    }
    .stTextInput>div>div>input:focus, .stTextArea>textarea:focus {
        border-color: #3498db;
    }
    .stTextArea>textarea {
        border-radius: 5px;
    }
    .stFileUploader>label {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation logic
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Sidebar content
if not st.session_state["authenticated"]:
    st.sidebar.title("Facial Expression Recognition")
    menu = st.sidebar.radio("Navigate", ["Home", "Login", "Sign Up"])

    if menu == "Home":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(' ')

        with col2:
            st.image("C:\\Users\\Paras Sharma\\OneDrive\\Pictures\\Saved Pictures\\fer.jpg" , width = 225)

        with col3:
            st.write(' ')
        #st.image("C:\\Users\\Paras Sharma\\OneDrive\\Pictures\\Saved Pictures\\fer.jpg" , width = 225)


        st.title("Welcome to the Facial Expression Recognition App")
        st.markdown("""
        This application uses state-of-the-art deep learning techniques to recognize facial expressions.
        **Features:**
        - User authentication for personalized experience.
        - Upload images or use real-time webcam input.
        - User feedback collection for continuous improvement.
        """)
       # st.image("C:\\Users\\Paras Sharma\\OneDrive\\Pictures\\Saved Pictures\\fer.jpg")

    elif menu == "Sign Up":
        st.title("Create an Account")
        username = st.text_input("Username", key="signup_username")
        password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            create_usertable()
            hashed_password = make_hashes(password)
            add_userdata(username, hashed_password)
            st.success("Account created successfully!")

    elif menu == "Login":
        st.title("Login to Your Account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            create_usertable()
            hashed_password = make_hashes(password)
            result = login_user(username, hashed_password)
            if result:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success(f"Welcome {username}!")
            else:
                st.error("Invalid username or password")
else:
    st.sidebar.title("Facial Expression Recognition")
    menu = st.sidebar.radio("Navigate", ["Predict", "User Feedback", "About"])

    if menu == "Predict":
        st.title("Predict Facial Expressions")
        st.markdown("Upload an image or capture one via webcam for real-time emotion recognition.")
        col1, col2 = st.columns([1, 2])
        with col1:
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        with col2:
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                with st.spinner('Analyzing the image...'):
                    predictions = predict_expression(image)
                st.subheader("Prediction Results")
                for emotion, score in predictions.items():
                    st.progress(int(score * 100))
                    st.write(f"{emotion}: {score * 100:.2f}%")
            else:
                st.write("No image uploaded.")
                webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)

    elif menu == "User Feedback":
        st.title("We Value Your Feedback")
        feedback = st.text_area("Share your thoughts about this application")
        if st.button("Submit Feedback"):
            conn = create_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO feedback_table (username, feedback) VALUES (%s, %s)",
                           (st.session_state["username"], feedback))
            conn.commit()
            conn.close()
            st.success("Thank you for your feedback!")

    elif menu == "About":
        st.title("About the Project")
        st.markdown("""
        ### Project Overview
        This project aims to leverage advanced machine learning techniques to identify emotions from facial expressions.
        The model has been trained on a diverse dataset to ensure robustness.

        **Technologies Used:**
        - TensorFlow/Keras for model training.
        - Streamlit for frontend development.
        - MySQL for user data management.

        **Future Work:**
        - Real-time video emotion detection.
        - Deployment as a web service.

        Developed by [Your Team Name].
        """)

    # Sign Out button
    if st.sidebar.button("Sign Out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.experimental_rerun()
