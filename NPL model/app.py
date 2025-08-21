import streamlit as st
import joblib
import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 📥 Downloads (first time only)
nltk.download('punkt')
nltk.download('stopwords')

# 🚀 Load the model & tools
model = joblib.load("nb_model_2.pkl")
vectorizer = joblib.load("bow_vectorizer_2.pkl")
label_mapping = joblib.load("label_mapping.pkl")



# 🧹 Text preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([char for char in text if not char.isdigit()])
    text = ''.join([char for char in text if char.isascii()])
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

import base64

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image filename
set_background("image_3.png")


# 🧠 App UI
st.markdown(
    "<h1 style='text-align: left; color: #006400; font-size: 50px;'>Emotion Prediction App 🔮</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: left; color: #228B22; font-size: 20px;'>Enter a sentence and find out the emotion behind it!</h4>",
    unsafe_allow_html=True
)


# 🎯 Try Example button
if st.button("Try Example: I'm feeling great!"):
    st.session_state['text'] = "I'm feeling great!"

# 🔀 Two column layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div padding:10px'>
            <h5 style='color:#228B22;'>Your Text</h5>
        </div>
    """, unsafe_allow_html=True)
    user_input = st.text_area("", value=st.session_state.get('text', ''), key="input")

with col2:
    pass

# 🧪 Predict Button
if st.button("Predict Emotion"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction_number = model.predict(vectorized)[0]
        prediction_label = label_mapping[prediction_number]

        st.success(f"Predicted Emotion: {prediction_label}")

        # 😍 Show matching emoji
        emotion_image_map = {
            "happy": "happy.png",
            "sad": "sad.png",
            "angry": "angry.png",
            "surprise": "surprise.png",
            "fear": "fear.png"
        }
        # 📘 Emotion descriptions
        emotion_descriptions = {
            "happy": "Joy, excitement, or positivity 😊",
            "joy": "delight,celebration 😄",
            "sadness": "Feeling down, low, or disappointed 😢",
            "angry": "Frustration, annoyance, or rage 😠",
            "fear": "Anxiety, nervousness, or being scared 😨",
            "surprise": "Shock, amazement, or unexpectedness 😲"
        }

        st.markdown(
                """
                <style>
                /* Stronger selector for success (Predicted Emotion) */
                div[role="alert"].stAlert-success {
                    background-color: rgba(0, 60, 0, 0.95);  /* dark forest green */
                    color: #ADFF2F;  /* green-yellow */
                    border-radius: 10px;
                    border: 1px solid #006400;
                    font-weight: bold;
                }

                /* Stronger selector for info (Description) */
                div[role="alert"].stAlert-info {
                    background-color: rgba(0, 40, 0, 0.95);  /* even darker */
                    color: #7CFC00;  /* lawn green */
                    border-radius: 10px;
                    border: 1px solid #228B22;
                    font-weight: bold;
                }
                </style>
                """,
                unsafe_allow_html=True
            )




        # 📘 Emotion Description
        description = emotion_descriptions.get(prediction_label, "No description available.")
        st.info(f"**{prediction_label.capitalize()}** = {description}")

        # 📊 Probability Chart
        probs = model.predict_proba(vectorized)[0]
        prob_df = pd.DataFrame({
            "Emotion": list(label_mapping.values()),
            "Probability": probs
        })

        st.markdown(
                "<h3 style='color: #228B22; font-weight: bold;'>Prediction Probabilities</h3>",
                unsafe_allow_html=True
            )

        st.bar_chart(prob_df.set_index("Emotion"))


st.markdown("""

    <div style="text-align: left; padding: 10px;">
        <p style="color: #228B22; font-weight: bold; font-size: 16px;">
        </p>
    </div>
""", unsafe_allow_html=True)
