import streamlit as st
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from tqdm import tqdm
import torch
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import umap
from sklearn.cluster import KMeans
import os
import tempfile

# ---- Configuration ----
ROBOFLOW_API_KEY = "oetSR0O32rWFikQEeiNw"
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/10"
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

PLAYER_ID = 2
STRIDE = 30
BATCH_SIZE = 32

st.title("Football Team Clustering & Tracking Video Processor")

uploaded_file = st.file_uploader("Upload a football video", type=["mp4", "avi"])
if uploaded_file:
    st.info("Extracting player crops and processing video. This might take a while...")

    # Save the uploaded video to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
        tmp_input.write(uploaded_file.read())
        SOURCE_VIDEO_PATH = tmp_input.name

    # ---- STEP 1: Collect crops for classification ----
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH, stride=STRIDE)
    crops = []
    frames_for_team = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops
        frames_for_team.append(frame)
        if len(crops) > 100:
            break
    st.success(f"Collected {len(crops)} player crops.")

    # ---- STEP 2: Extract features with SigLIP ----
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
    EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
    crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
    batches = chunked(crops_pil, BATCH_SIZE)
    data = []
    with torch.no_grad():
        for batch in tqdm(batches, desc='embedding extraction'):
            inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
            outputs = EMBEDDINGS_MODEL(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            data.append(embeddings)
    data = np.concatenate(data)
    st.success(f"Extracted {len(data)} embeddings.")

    # ---- STEP 3: UMAP reduction + KMeans clustering ----
    REDUCER = umap.UMAP(n_components=3)
    CLUSTERING_MODEL = KMeans(n_clusters=2)
    projections = REDUCER.fit_transform(data)
    clusters = CLUSTERING_MODEL.fit_predict(projections)
    st.success("UMAP+KMeans team clustering complete.")

    # ---- STEP 4: Per-frame detection/tracking/annotation + output video ----
    BALL_ID = 0
    GOALKEEPER_ID = 1
    REFEREE_ID = 3

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )

    tracker = sv.ByteTrack()
    tracker.reset()

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = output_file.name
    output_file.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cluster_idx = 0  # For assigning team to each player as we go

    stframe = st.empty()
    progress = st.progress(0)
    for idx in tqdm(range(total_frames), desc='Processing video'):
        ret, frame = cap.read()
        if not ret:
            break
        if STRIDE > 1 and idx % STRIDE != 0:
            continue

        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        # Separate players, assign class/team by cluster (your way)
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        # Assign class_id to clusters (team) for first 100 crops
        if len(players_detections) > 0 and cluster_idx + len(players_detections) <= len(clusters):
            players_detections.class_id = clusters[cluster_idx:cluster_idx+len(players_detections)]
            cluster_idx += len(players_detections)
        # Goalkeepers and referees unchanged
        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        # Merge all for annotation
        merged = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])
        labels = [f"#{tracker_id}" if tracker_id is not None else "" for tracker_id in merged.tracker_id]
        merged.class_id = merged.class_id.astype(int)

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=merged)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=merged, labels=labels)
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)

        out.write(annotated_frame)

        if idx % 50 == 0:
            stframe.image(annotated_frame, channels="BGR", caption=f"Frame {idx + 1}")
        progress.progress(int((idx + 1) / total_frames * 100))

    cap.release()
    out.release()
    progress.empty()
    st.success("Processing complete!")

    # Download button for user
    with open(output_path, "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="football_processed.mp4",
            mime="video/mp4"
        )
else:
    st.warning("Please upload a football video to start.")
