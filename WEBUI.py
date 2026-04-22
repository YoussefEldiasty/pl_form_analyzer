import streamlit as st
import tempfile
import os
from Logic import process_video

st.title("Powerlifting Form Analyzer")

video = st.file_uploader("Upload a squat video", type = ["mp4", "mov", "avi"])

if video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video.read())
    tfile.close()

    input_path = tfile.name
    base, ext = os.path.splitext(input_path)
    output_path = base + "_out.mp4"

    with st.spinner("Processing Video..."):
        rep_data, output_path = process_video(input_path, output_path)
    
    st.success("Analyzing Complete!")

    st.video(output_path)

    st.subheader("Rep Analysis")
    st.dataframe(rep_data)