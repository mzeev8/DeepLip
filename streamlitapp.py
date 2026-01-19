import os
import tensorflow as tf
import imageio
import streamlit as st
from utils import load_data, num_to_char
from modelutils import load_model
import numpy as np

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>DeepLip</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image('images/deeplip.png')
    st.markdown("## Welcome to DeepLip!")

options = os.listdir(os.path.join('data', 's1'))

st.markdown("### 🎥 Select a Video")
selected_video = st.selectbox(
    'Pick a video from the dataset below:',
    options
)

if selected_video:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📺 Original Video Preview")
        file_path = os.path.join('data', 's1', selected_video)

        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.markdown("### 🔬 Model Input Visualization")

        video, annotations = load_data(tf.convert_to_tensor(file_path))
        frames = [np.squeeze(frame.numpy(), axis=-1) for frame in video]  
        frames = [np.uint8(frame * 255) for frame in frames]
        imageio.mimsave('output.gif', frames, fps=10)
        
        st.image('output.gif', width=400, caption="Model's View (Preprocessed Frames)")

        model = load_model()
        if model is None:
            st.error("🚨 Failed to load the model. Please check your configuration.")
        else:
            st.success("✅ Model loaded successfully!")
            st.info('**Next Step**: Let’s run a prediction on the selected video.')

            st.markdown("### 🔡 Raw Model Predictions")
            yhat = model.predict(tf.expand_dims(video, axis=0))
            st.write("**Raw argmax tokens:**", tf.argmax(yhat, axis=1).numpy())

            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.write("**CTC Decoded Tokens:**", decoder)

            st.markdown("### 🗣️ Decoded Speech")
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
            st.markdown(f"**Predicted Transcript:** `{converted_prediction}`")

else:
    st.warning("⚠️ No videos found. Please check your 'data/s1' directory or refresh the page.")

