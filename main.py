import tempfile
import streamlit as st
import cv2
import os
import zipfile
import moviepy.editor as moviepy
from fall_detectors import VideoFallDetector, LiveFallDetector
from streamlit_webrtc import WebRtcMode, webrtc_streamer


def showVideo(video_name):
    try:
        st_video = open(video_name, "rb")
        video_bytes = st_video.read()
        st.video(
            video_bytes,
        )

    except OSError:
        """Error loading video file"""


def runFallDetector(video_name, fps):
    fall_detector = VideoFallDetector()
    st.subheader("Fall Detection")
    with st.spinner("Processing video..."):
        fall_detector.begin(video_name, fps)
    st.success("Done!")

    clip = moviepy.VideoFileClip("fall_detected_video.mp4")
    clip.write_videofile("converted_video_out.mp4", fps=fps)
    showVideo("converted_video_out.mp4")

    os.remove("fall_detected_video.mp4")
    os.remove("converted_video_out.mp4")


def handleImageSequence():
    uploaded_zip = st.file_uploader(
        "Upload a zip file containing .jpg or .png images", type=["zip"]
    )
    fps = st.slider("Frames Per Second (FPS)", 1, 60, 21)
    if uploaded_zip is not None:
        st.subheader("Input Video")

        tmp_dir = tempfile.TemporaryDirectory()
        with st.spinner("Converting to video..."):
            sequence_video_name = "sequence_video.mp4"
            with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                zip_ref.extractall(tmp_dir.name)

            dir_outer = tmp_dir.name
            dir_inner = os.path.join(tmp_dir.name, uploaded_zip.name.split(".")[0])

            if len(os.listdir(dir_outer)) > 1:
                imgs_list = os.listdir(dir_outer)
                dir = dir_outer
            elif len(os.listdir(dir_inner)) > 1:
                imgs_list = os.listdir(dir_inner)
                dir = dir_inner
            else:
                st.error("No images found in zip file")
                return

            images = [
                img for img in imgs_list if img.endswith(".jpg") or img.endswith(".png")
            ]
            frame = cv2.imread(os.path.join(dir, images[0]))
            height, width, layers = frame.shape

            video = cv2.VideoWriter(
                sequence_video_name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            for image in images:
                video.write(cv2.imread(os.path.join(dir, image)))

            cv2.destroyAllWindows()
            video.release()
            tmp_dir.cleanup()

        st.success("Done!")
        showVideo(sequence_video_name)
        runFallDetector(sequence_video_name, fps)


def handleVideoFile():
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=[
            "mp4",
            "avi",
            "mpeg",
            "mov",
            "wmv",
            "mkv",
            "flv",
            "webm",
            "3gp",
        ],
    )
    fps = st.slider("Frames Per Second (FPS)", 1, 60, 21)
    if uploaded_video is not None:
        video_name = uploaded_video.name

        st.subheader("Input Video")
        with st.spinner("Converting video..."):
            with open(video_name, mode="wb") as f:
                f.write(uploaded_video.read())  # save video to disk
            clip = moviepy.VideoFileClip(video_name)
            clip.write_videofile("input_video.mp4", fps=fps)
            os.remove(video_name)

        st.success("Done!")
        showVideo("input_video.mp4")
        runFallDetector("input_video.mp4", fps)


def handleCamera():
    camera_detector = LiveFallDetector()
    webrtc_ctx = webrtc_streamer(
        key="fall-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=camera_detector.process_frame,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def handleDemoVideo():
    st.markdown(
        """
        The Input Video is the input to the app, the Fall Detection video shows the algorithm in action. 
        
        The skeleton and bounding boxes are drawn using pose information from Openpifpaf. 
        The header over the Fall Detection Video shows the result of LSTM classification model for each frame.
        """
    )
    st.subheader("Input Video")
    showVideo("demo_video_in.mp4")
    st.subheader("Fall Detection")
    showVideo("demo_video_fd.mp4")


def main():
    st.markdown(
        "<h1 style='text-align: center;'>Fall Detection App</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("By: [Supun Gamlath](https://github.com/supungamlath)")
    desc_title = st.subheader("Description")
    desc_body = st.markdown(
        """
        Every year, approximately 37 million falls occur worldwide, often necessitating immediate medical attention. 
        For elderly individuals living alone, falls can be particularly dangerous, as they may be unable to call for help. 
            
        For my semester project, I'm building a fall detection robot that can be deployed in any household, able to navigate autonomously and detect falls in real-time.

        This app was built as part of that project to test the fall detection method using Streamlit, Openpifpaf and PyTorch. 
        OpenPifPaf is a library that allows for real-time pose estimation. PyTorch is a deep learning library used to run a LSTM model that was pre-trained to detect falls.

        The fall detection method is based on the research [Multi-camera, multi-person, and real-time fall detection using long short term memory](https://doi.org/10.1117/12.2580700) by Mohammad Taufeeque, Samad Koita, Nicolai Spicher and Thomas M. Deserno.
        
        The method involves analyzing video frames using OpenPifPaf to estimate the pose of a person. 
        Five features are then extracted and fed into an LSTM model, classifying them into one of five classes: `None`, `Normal`, `Falling`, `Warning`, `Fallen`
        
        The five features are:

            1. Aspect ratio of the bounding box around the person. Captures body configuration.  
            2. Logarithm of the angle between the vertical axis and the torso vector. Provides non-linearity to the angle feature.
            3. Rotation energy calculated using torso and head vectors between current and previous frame. Captures motion. 
            4. Derivative of the ratio_bbox. Captures rate of change of body configuration.  
            5. Generalized force calculated using joint angles and velocities over 3 frames. Captures acceleration of motion.
        """
    )

    datasets_title = st.subheader("Datasets Used")
    datasets_body = st.markdown(
        """
        [UR Fall Detection Dataset](http://fenix.ur.edu.pl/mkepski/ds/uf.html)

        [FALL-UP Dataset](https://sites.google.com/up.edu.mx/har-up)
        """
    )

    ref_title = st.subheader("References")
    ref_body = st.markdown(
        """
        1. Mohammad Taufeeque, Samad Koita, Nicolai Spicher, and Thomas M. Deserno "Multi-camera, multi-person, and real-time fall detection using long short term memory", Proc. SPIE 11601, Medical Imaging 2021: Imaging Informatics for Healthcare, Research, and Applications, 1160109 (15 February 2021); https://doi.org/10.1117/12.2580700
        2. Bogdan Kwolek, Michal Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, Computer Methods and Programs in Biomedicine, Volume 117, Issue 3, December 2014, Pages 489-501, ISSN 0169-2607; http://fenix.ur.edu.pl/mkepski/ds/uf.html
        3. S. Kreiss, L. Bertoni, and A. Alahi, OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association. 2021; https://doi.org/10.48550/arXiv.2103.02440
        """
    )

    st.sidebar.title("Select Mode")
    choice = st.sidebar.selectbox(
        "Mode", ("Home", "Demo Video", "Device Camera", "Image Sequence", "Video File")
    )
    st.sidebar.markdown("Select a mode to begin")

    # st.sidebar.text("Demo Video - Watch a video of Fall Detection in action")
    # st.sidebar.text(
    #     "Image Sequence - Upload a zip file containing a sequence of images"
    # )
    # st.sidebar.text("Video File - Upload a video file")

    if choice != "Home":
        desc_title.empty()
        desc_body.empty()
        datasets_title.empty()
        datasets_body.empty()
        ref_title.empty()
        ref_body.empty()

    if choice == "Device Camera":
        handleCamera()

    if choice == "Image Sequence":
        handleImageSequence()

    elif choice == "Video File":
        handleVideoFile()

    elif choice == "Demo Video":
        handleDemoVideo()


if __name__ == "__main__":
    main()
