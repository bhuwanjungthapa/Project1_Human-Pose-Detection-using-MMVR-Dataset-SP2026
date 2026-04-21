import io
import numpy as np
import streamlit as st
from PIL import Image, ImageSequence
import mediapipe as mp

st.set_page_config(
    page_title="RGB Pose Detection — MMVR Project",
    page_icon="🧍",
    layout="wide",
)

st.title("RGB Pose Detection")
st.caption(
    "Upload one or more images and this app will run MediaPipe Pose on them — "
    "equivalent to Shell 1.2 / 2.2 of the MMVR notebook."
)

with st.sidebar:
    st.header("Detection settings")
    min_conf = st.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
    model_complexity = st.selectbox(
        "Model complexity", options=[0, 1, 2], index=1,
        help="0 = lite, 1 = full, 2 = heavy (slower, more accurate)",
    )
    landmark_thickness = st.slider("Landmark thickness", 1, 15, 9)
    connection_thickness = st.slider("Connection thickness", 1, 15, 9)
    landmark_radius = st.slider("Landmark radius", 1, 15, 7)

def load_rgb_image_any_format(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes))
    fmt = getattr(img, "format", "UNKNOWN")
    if fmt == "MPO":
        try:
            img = next(ImageSequence.Iterator(img))
        except Exception:
            pass
    return np.array(img.convert("RGB")), fmt

def run_pose_detection(image_rgb: np.ndarray):
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    landmark_style = mp.solutions.drawing_styles.DrawingSpec(
        color=(0, 255, 0),
        thickness=landmark_thickness,
        circle_radius=landmark_radius,
    )
    connection_style = mp.solutions.drawing_styles.DrawingSpec(
        color=(255, 0, 0),
        thickness=connection_thickness,
        circle_radius=4,
    )

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        min_detection_confidence=min_conf,
    ) as pose:
        results = pose.process(image_rgb)
        annotated = image_rgb.copy()

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style,
            )
            return annotated, results.pose_landmarks, True

        return annotated, None, False

uploaded_files = st.file_uploader(
    "Upload image(s) — JPG / JPEG / PNG (iPhone MPO supported)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least one image to get started.")
    st.stop()

st.subheader(f"Processing {len(uploaded_files)} image(s)")

for idx, f in enumerate(uploaded_files):
    st.markdown("---")
    st.markdown(f"### {f.name}")

    try:
        image_rgb, fmt = load_rgb_image_any_format(f.read())
    except Exception as e:
        st.error(f"Could not read `{f.name}`: {e}")
        continue

    st.caption(f"Format: `{fmt}` | Size: `{image_rgb.shape[1]}×{image_rgb.shape[0]}`")

    with st.spinner("Running MediaPipe Pose..."):
        annotated, landmarks, detected = run_pose_detection(image_rgb)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption="Pose overlay", use_container_width=True)

    if detected:
        st.success(f"Pose detected in {f.name}")
        with st.expander("Show normalized landmark coordinates"):
            rows = []
            for i, lm in enumerate(landmarks.landmark):
                rows.append({
                    "id": i,
                    "x": round(lm.x, 4),
                    "y": round(lm.y, 4),
                    "z": round(lm.z, 4),
                    "visibility": round(lm.visibility, 4),
                })
            st.dataframe(rows, use_container_width=True, key=f"df_{idx}_{f.name}")

        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        st.download_button(
            "Download annotated image",
            data=buf.getvalue(),
            file_name=f"annotated_{f.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
            key=f"download_{idx}_{f.name}",
        )
    else:
        st.warning(f"No pose detected in {f.name}")
