import streamlit as st
import tensorflow as tf
import numpy as np

# Set page configuration
st.set_page_config(page_title="Plant Disease Recognition", page_icon="🌿", layout="wide")

# Custom CSS to style the web app
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
        color: #333333;
    }
    .stButton>button {
        background-color: #2eb82e;
        color: white;
        border-radius: 8px;
        width: 150px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #1d6d1d;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar with new style
st.markdown("""
    <style>
        /* Sidebar background gradient styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom right, #a7ffeb, #1de9b6);
            color: #004d40;
            padding: 20px;
        }
        /* Sidebar title styling */
        [data-testid="stSidebar"] .css-1aumxhk {
            font-size: 24px;
            color: #004d40;
            font-weight: bold;
            text-align: center;
        }
        /* Radio buttons styling */
        [data-testid="stSidebar"] .css-1fcdlh7, [data-testid="stSidebar"] .css-16huue1 {
            color: #00695c;
            font-size: 18px;
            font-weight: bold;
        }
        /* Hover effect on radio buttons */
        [data-testid="stSidebar"] .css-1fcdlh7:hover, [data-testid="stSidebar"] .css-16huue1:hover {
            color: #004d40;
            background-color: #a7ffeb;
            border-radius: 8px;
            cursor: pointer;
        }
        /* Sidebar icons for options */
        [data-testid="stSidebar"] .css-1fcdlh7:nth-of-type(1):before { 
            content: "🏠 "; 
        }
        [data-testid="stSidebar"] .css-1fcdlh7:nth-of-type(2):before { 
            content: "🔍 "; 
        }
        [data-testid="stSidebar"] .css-1fcdlh7:nth-of-type(3):before { 
            content: "ℹ️ "; 
        }
    </style>
    """, unsafe_allow_html=True)

# Updated Sidebar with stylish design
st.sidebar.title("🌿 Dashboard")
app_mode = st.sidebar.radio("Menu", ["Home", "Disease Recognition", "About"])

# Main Page Content
if app_mode == "Home":
    st.markdown("""
        <div style="background-color: #e0f7fa; border-radius: 15px; padding: 25px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); margin-bottom: 20px;">
            <h1 style='text-align: center; color: #2e8b57; font-size: 48px;'>🌿 Plant Disease Recognition System 🌿</h1>
        </div>
    """, unsafe_allow_html=True)

    image_paths = ["home_page4.jpg", "home_page2.jpg"]
    cols = st.columns(len(image_paths))

    for i, image_path in enumerate(image_paths):
        with cols[i]:
            st.image(image_path, use_column_width=True)

    st.markdown("""
        <div style="background-color: #f0f4c3; border-radius: 15px; padding: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin: 40px 0; text-align: center;">
            <h2 style="color: #33691e; font-size: 36px;">Welcome to the Plant Disease Recognition System! 🌿🔍</h2>
            <p style="font-size: 20px; color: #4e342e; max-width: 800px; margin: 0 auto;">
                Our mission is to help in identifying plant diseases efficiently using state-of-the-art AI technology. Simply upload an image of a plant, 
                and our system will analyze it to detect any signs of diseases.
                Together, let's protect our crops and ensure a healthier harvest!
            </p>
            <div style="margin-top: 30px;">
                <h3 style="color: #558b2f;">How It Works</h3>
                <ul style="list-style-type: none; padding: 0;">
                    <li>🌱 <b>Step 1:</b> Upload an image of the plant.</li>
                    <li>📷 <b>Step 2:</b> Our system analyzes the image using AI algorithms.</li>
                    <li>✅ <b>Step 3:</b> Receive instant results and recommendations.</li>
                </ul>
            </div>
           </div>
    """, unsafe_allow_html=True)

elif app_mode == "Disease Recognition":

    st.markdown("""
        <div style="background-color: #e0f2f1; border-radius: 12px; padding: 25px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); margin-bottom: 20px;">
            <h1 style='text-align: center; color: #00695c; font-size: 36px;'>🔍 Disease Recognition</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color: #ffffff; border: 2px dashed #81c784; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
            <h3 style='color: #388e3c;'>Upload an Image of the Plant</h3>
            <p style='color: #4e342e; font-size: 16px;'>Please upload a clear image of the plant leaf to analyze its health status.</p>
        </div>
    """, unsafe_allow_html=True)

    # File uploader with improved UI
    test_image = st.file_uploader("Choose an Image:", type=['jpg', 'jpeg', 'png'])

    if test_image is not None:
        st.image(test_image, caption='Uploaded Image', use_column_width=750)

        # ปุ่ม Predict 
        if st.button("🚀 Predict"):
            with st.spinner('🔄 Analyzing the image...'):
                result_index = model_prediction(test_image)

            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']

            # แสดงผลลัพธ์การวิเคราะห์
            st.markdown(f"""
                <div style="background-color: #a5d6a7; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); margin-top: 20px;">
                    <h2 style='color: #1b5e20;'>🌱 The plant is likely affected by:</h2>
                    <h3 style='color: #388e3c; font-size: 24px;'><b>{class_name[result_index]}</b></h3>
                </div>
            """, unsafe_allow_html=True)

            st.snow()


elif app_mode == "About":
    st.markdown("""
    <div style="background-color: #a5d6a7; border-radius: 12px; padding: 25px; box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <h1 style='text-align: center; color: #2e7d32; font-size: 36px;'>🌿 About This Project 🌿</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #e8f5e9; border-radius: 12px; padding: 25px; box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <div style="text-align: center; margin: 15px 0;">
            <h3 style='color: #1b5e20;'>Developer: <span style='font-weight: bold;'>6510110116 Silmee Panan</span></h3>
            <p style='font-size: 18px; color: #388e3c; max-width: 800px; margin: 0 auto;'>
            This project is dedicated to helping farmers and researchers identify plant diseases quickly and accurately using AI technology.
                Our goal is to contribute to sustainable agriculture by providing tools that enhance crop protection and yield.
            </p>
        </div>
        <h2 style='text-align: center; color: #1b5e20; margin-top: 40px; font-size: 30px;'>About the Dataset</h2>
        <div style="background-color: #c8e6c9; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);">
            <p style='font-size: 16px; color: #2e7d32;'>
            The dataset utilized in this project consists of approximately 87K RGB images of both healthy and diseased crop leaves.
                These images have been carefully labeled and categorized into 38 distinct classes to ensure accurate disease detection.
            </p>
            <h4 style='text-align: center; color: #2e7d32;'>Dataset Structure:</h4>
            <ul style='list-style-type: none; padding: 0; color: #388e3c;'>
                <li>🌿 <b>Train set:</b> 70,295 images</li>
                <li>🌿 <b>Validation set:</b> 17,572 images</li>
                <li>🌿 <b>Test set:</b> 33 images</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
