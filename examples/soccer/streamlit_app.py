"""
Soccer AI — Streamlit app.

Analyse "temps réel" (image par image, affichée en direct) d'une vidéo de football
uploadée par l'utilisateur : détection des joueurs/ballon, tracking, classification
d'équipes, radar tactique et étude joueur par joueur (numéros, passes, cartographie).

Ce fichier réutilise au maximum les pipelines déjà définis dans `main.py` (le CLI du
projet) et les briques du package `sports/` (annotateurs, TeamClassifier, BallTracker,
ViewTransformer...) plutôt que de les redéfinir.

Lancer avec (depuis la racine du repo) :
    streamlit run examples/soccer/streamlit_app.py
"""
import os
import sys
import tempfile
import time
from typing import Iterator, Optional

import numpy as np
import streamlit as st
import supervision as sv
import torch

# Make both this app's own directory (for `import main` / `player_analysis`)
# and the repo root (for `import sports...`) resolvable regardless of how the
# process was launched — Streamlit Cloud doesn't pip-install the `sports`
# package, it only runs `pip install -r requirements.txt`, so the local
# `sports/` checkout has to be found via sys.path instead.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_APP_DIR))
for _path in (_APP_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from main import (  # noqa: E402
    BOX_ANNOTATOR,
    BOX_LABEL_ANNOTATOR,
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    STRIDE,
    get_crops,
    resolve_goalkeepers_team_id,
    run_ball_detection,
    run_pitch_detection,
    run_player_detection,
    run_player_tracking,
    run_radar,
    run_team_classification,
)
from sports.annotators.soccer import draw_pitch_heatmap  # noqa: E402
from sports.common.team import TeamClassifier  # noqa: E402
from player_analysis import (  # noqa: E402
    CONFIG as PITCH_CONFIG,
    PlayerMatchAnalyzer,
    ocr_available,
)

PARENT_DIR = _APP_DIR
DATA_DIR = os.path.join(PARENT_DIR, 'data')

LOCAL_MODEL_PATHS = {
    'player': os.path.join(DATA_DIR, 'football-player-detection.pt'),
    'ball': os.path.join(DATA_DIR, 'football-ball-detection.pt'),
    'pitch': os.path.join(DATA_DIR, 'football-pitch-detection.pt'),
}

# Google Drive file ids backing the same weights setup.sh downloads.
MODEL_DOWNLOAD_INFO = {
    'player': ('football-player-detection.pt', '17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q'),
    'ball': ('football-ball-detection.pt', '1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V'),
    'pitch': ('football-pitch-detection.pt', '1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf'),
}

# Public Roboflow Universe model backing the same dataset referenced in the README.
HOSTED_PLAYER_MODEL_ID_DEFAULT = "football-players-detection-3zvbc/10"

PLAYER_ANALYSIS_MODE = "Analyse par joueur (numéros, passes, carte)"

BASE_MODES = [
    "Détection joueurs",
    "Détection ballon",
    "Suivi des joueurs (tracking)",
    "Classification d'équipes",
    "Détection du terrain",
    "Radar (terrain + équipes)",
]

MODE_ICONS = {
    "Détection joueurs": "🎯",
    "Détection ballon": "⚪",
    "Suivi des joueurs (tracking)": "🔎",
    "Classification d'équipes": "👕",
    "Détection du terrain": "🏟️",
    "Radar (terrain + équipes)": "🗺️",
    PLAYER_ANALYSIS_MODE: "🔢",
}

# Modes that can run against the hosted Roboflow API (player-detection model only).
HOSTED_COMPATIBLE_MODES = {
    "Détection joueurs",
    "Suivi des joueurs (tracking)",
    "Classification d'équipes",
}


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def heavy_mode_available(device: str) -> bool:
    """
    The per-player analysis mode runs 3 models + tracking + OCR per frame,
    which is only practical with a GPU. It's forced on with
    FORCE_ENABLE_PLAYER_ANALYSIS=1 for advanced/patient CPU users, and off by
    default anywhere a CUDA device isn't available (e.g. free cloud hosting).
    """
    if os.environ.get("FORCE_ENABLE_PLAYER_ANALYSIS") == "1":
        return True
    return device == "cuda"


def hosted_api_available() -> bool:
    try:
        import inference  # noqa: F401
        return True
    except ImportError:
        return False


def get_hosted_model(model_id: str, api_key: str):
    from inference import get_model
    return get_model(model_id=model_id, api_key=api_key)


def run_player_detection_hosted(
    source_video_path: str, model, confidence: float, stride: int = 1
) -> Iterator[np.ndarray]:
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)
    for frame in frame_generator:
        result = model.infer(frame, confidence=confidence)[0]
        detections = sv.Detections.from_inference(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking_hosted(
    source_video_path: str, model, confidence: float, stride: int = 1
) -> Iterator[np.ndarray]:
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = model.infer(frame, confidence=confidence)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification_hosted(
    source_video_path: str, model, confidence: float, device: str, stride: int = 1
) -> Iterator[np.ndarray]:
    crop_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in crop_generator:
        result = model.infer(frame, confidence=confidence)[0]
        detections = sv.Detections.from_inference(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = model.infer(frame, confidence=confidence)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        players_crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(players_crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers) if len(goalkeepers) else np.array([])

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def required_local_models(mode: str) -> list:
    needed = []
    if mode in ("Détection joueurs", "Suivi des joueurs (tracking)",
                "Classification d'équipes", "Radar (terrain + équipes)",
                PLAYER_ANALYSIS_MODE):
        needed.append('player')
    if mode in ("Détection ballon", PLAYER_ANALYSIS_MODE):
        needed.append('ball')
    if mode in ("Détection du terrain", "Radar (terrain + équipes)", PLAYER_ANALYSIS_MODE):
        needed.append('pitch')
    return needed


@st.cache_resource(show_spinner=False)
def download_local_model(name: str) -> str:
    """Downloads a model weight file once per running app instance (cached
    across all sessions), so a fresh cloud deployment is self-sufficient
    without shell access to run setup.sh."""
    filename, file_id = MODEL_DOWNLOAD_INFO[name]
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        import gdown
        gdown.download(id=file_id, output=path)
    return path


def ensure_local_models(names: list) -> list:
    """Downloads any of `names` not already present locally. Returns the
    subset that still failed to download."""
    failed = []
    for name in names:
        try:
            path = download_local_model(name)
            if not os.path.isfile(path):
                failed.append(name)
        except Exception:
            failed.append(name)
    return failed


def build_local_generator(mode: str, video_path: str, device: str, stride: int):
    if mode == "Détection joueurs":
        return run_player_detection(video_path, device, stride=stride)
    if mode == "Détection ballon":
        return run_ball_detection(video_path, device, stride=stride)
    if mode == "Suivi des joueurs (tracking)":
        return run_player_tracking(video_path, device, stride=stride)
    if mode == "Classification d'équipes":
        return run_team_classification(video_path, device, stride=stride)
    if mode == "Détection du terrain":
        return run_pitch_detection(video_path, device, stride=stride)
    if mode == "Radar (terrain + équipes)":
        return run_radar(video_path, device, stride=stride)
    raise ValueError(mode)


def build_hosted_generator(
    mode: str, video_path: str, model, confidence: float, device: str, stride: int
):
    if mode == "Détection joueurs":
        return run_player_detection_hosted(video_path, model, confidence, stride=stride)
    if mode == "Suivi des joueurs (tracking)":
        return run_player_tracking_hosted(video_path, model, confidence, stride=stride)
    if mode == "Classification d'équipes":
        return run_team_classification_hosted(
            video_path, model, confidence, device, stride=stride)
    raise ValueError(mode)


def save_uploaded_video(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Soccer AI — Analyse vidéo",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DETECTED_DEVICE = detect_device()
HEAVY_MODE_ENABLED = heavy_mode_available(DETECTED_DEVICE)

MODES = BASE_MODES + ([PLAYER_ANALYSIS_MODE] if HEAVY_MODE_ENABLED else [])

DEVICE_LABELS = {"cuda": "GPU (CUDA)", "mps": "GPU (Apple MPS)", "cpu": "CPU"}

header_left, header_right = st.columns([4, 1.3])
with header_left:
    st.title("⚽ Soccer AI")
    st.caption(
        "Détection, suivi et analyse tactique de matchs de football par IA — "
        "upload une vidéo, choisis un mode, regarde l'analyse se construire en direct."
    )
with header_right:
    st.markdown(
        f"<div style='text-align:right; padding-top: 0.6rem;'>"
        f"<span style='background:rgba(34,197,94,0.15); color:#22C55E; "
        f"padding:4px 10px; border-radius:999px; font-size:0.85rem; font-weight:600;'>"
        f"● {DEVICE_LABELS.get(DETECTED_DEVICE, DETECTED_DEVICE)}</span></div>",
        unsafe_allow_html=True,
    )

capability_cols = st.columns(len(BASE_MODES) + 1)
for col, name in zip(capability_cols, BASE_MODES + [PLAYER_ANALYSIS_MODE]):
    with col:
        enabled = name != PLAYER_ANALYSIS_MODE or HEAVY_MODE_ENABLED
        icon = MODE_ICONS[name]
        style = "opacity:1;" if enabled else "opacity:0.35;"
        st.markdown(
            f"<div style='{style} text-align:center; font-size:0.78rem; line-height:1.3;'>"
            f"<div style='font-size:1.4rem'>{icon}</div>{name.split('(')[0].strip()}</div>",
            unsafe_allow_html=True,
        )

if not HEAVY_MODE_ENABLED:
    st.caption(
        "🔢 L'analyse par joueur (numéros de maillot, passes, cartographie) est "
        "désactivée ici — elle nécessite un GPU pour rester praticable. "
        "Lance l'app en local avec un GPU CUDA pour l'activer."
    )

st.divider()

# --------------------------------------------------------------------------
# Sidebar — configuration
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("🎯 Mode d'analyse", expanded=True):
        mode = st.selectbox(
            "Mode", MODES,
            format_func=lambda m: f"{MODE_ICONS.get(m, '')} {m}",
        )

        model_source = st.radio(
            "Source du modèle",
            ["Local (recommandé)", "API Roboflow hébergée"],
            help=(
                "Local : télécharge et utilise les poids YOLO (une seule fois), "
                "tourne hors-ligne et est plus rapide. API hébergée : pas de "
                "téléchargement mais dépend du réseau, et ne couvre que les "
                "modes basés sur la détection de joueurs."
            ),
        )
        use_hosted = model_source == "API Roboflow hébergée"

        if use_hosted and not hosted_api_available():
            st.warning(
                "Le paquet `inference` n'est pas installé : l'API hébergée "
                "n'est pas disponible ici (`pip install inference`). "
                "Utilisation des modèles locaux."
            )
            use_hosted = False

        if use_hosted and mode not in HOSTED_COMPATIBLE_MODES:
            st.warning(
                f"« {mode} » nécessite les modèles locaux (ballon/terrain non "
                "disponibles via l'API hébergée dans cette app)."
            )

        if mode == PLAYER_ANALYSIS_MODE:
            st.caption(
                "⚠️ Mode le plus lourd : 3 modèles + tracking + OCR par frame. "
                "Même sur GPU, compte plusieurs minutes pour un court extrait."
            )
            if not ocr_available():
                st.info(
                    "`easyocr` n'est pas installé : les numéros de maillot ne "
                    "seront pas lus (joueurs identifiés par ID de suivi "
                    "uniquement)."
                )

        api_key = ""
        hosted_model_id = HOSTED_PLAYER_MODEL_ID_DEFAULT
        confidence = 0.3
        if use_hosted:
            default_key = os.environ.get("ROBOFLOW_API_KEY", "")
            try:
                default_key = st.secrets.get("ROBOFLOW_API_KEY", default_key)
            except Exception:
                pass
            api_key = st.text_input(
                "Clé API Roboflow", value=default_key, type="password",
                help="Définie via la variable d'environnement ROBOFLOW_API_KEY "
                     "ou st.secrets['ROBOFLOW_API_KEY'], modifiable ici.",
            )
            hosted_model_id = st.text_input("ID du modèle hébergé", value=hosted_model_id)
            confidence = st.slider("Seuil de confiance", 0.1, 0.9, 0.3, 0.05)

    with st.expander("🚀 Performance", expanded=False):
        device = st.selectbox(
            "Device", ["cpu", "cuda", "mps"],
            index=["cpu", "cuda", "mps"].index(DETECTED_DEVICE),
            format_func=lambda d: DEVICE_LABELS.get(d, d),
            help="Détecté automatiquement, modifiable si besoin.",
        )
        stride = st.slider(
            "Traiter 1 frame sur N", 1, 10, 2 if device == "cpu" else 1,
            help="Un stride plus élevé accélère le traitement (moins d'images "
                 "analysées) au prix d'un rendu moins fluide et d'un tracking "
                 "un peu moins stable.",
        )
        display_every = st.slider(
            "Rafraîchir l'aperçu toutes les N frames", 1, 20, 1,
            help="Limite la fréquence de mise à jour de l'image affichée pour "
                 "une UI plus fluide (le traitement, lui, ne saute aucune "
                 "frame sélectionnée par le stride).",
        )

    with st.expander("💾 Export", expanded=False):
        save_output = st.checkbox("Exporter la vidéo annotée", value=True)

    st.divider()
    st.caption(
        "Les modèles locaux manquants sont téléchargés automatiquement au "
        "premier lancement d'un mode (~130 Mo chacun, une seule fois)."
    )

# --------------------------------------------------------------------------
# Main area — upload & run
# --------------------------------------------------------------------------

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "📤 Vidéo de match", type=["mp4", "avi", "mov", "mkv"],
        help="Formats supportés : MP4, AVI, MOV, MKV.",
    )
    start = st.button(
        "▶️ Lancer l'analyse", disabled=uploaded_file is None, type="primary",
        use_container_width=True,
    )

if start and uploaded_file is not None:
    if use_hosted and mode not in HOSTED_COMPATIBLE_MODES:
        st.info(
            f"Le mode « {mode} » n'est pas disponible via l'API hébergée : "
            "utilisation des modèles locaux."
        )
        use_hosted = False

    if use_hosted and not api_key:
        st.error("Renseigne une clé API Roboflow pour utiliser le modèle hébergé.")
        st.stop()

    if not use_hosted:
        needed = required_local_models(mode)
        missing = [n for n in needed if not os.path.isfile(LOCAL_MODEL_PATHS[n])]
        if missing:
            with st.spinner(
                f"Téléchargement des modèles nécessaires ({', '.join(missing)})… "
                "~130 Mo chacun, une seule fois."
            ):
                failed = ensure_local_models(missing)
            if failed:
                st.error(
                    f"Échec du téléchargement automatique des modèles : "
                    f"{', '.join(failed)}. Vérifie la connexion réseau, ou "
                    "télécharge-les manuellement via `./setup.sh` (voir README) "
                    "puis relance l'app."
                )
                st.stop()

    video_path = save_uploaded_video(uploaded_file)
    video_info = sv.VideoInfo.from_video_path(video_path)
    total_to_process = max(1, -(-video_info.total_frames // stride))

    analyzer: Optional[PlayerMatchAnalyzer] = None
    if mode == PLAYER_ANALYSIS_MODE:
        analyzer = PlayerMatchAnalyzer(
            player_model_path=LOCAL_MODEL_PATHS['player'],
            pitch_model_path=LOCAL_MODEL_PATHS['pitch'],
            ball_model_path=LOCAL_MODEL_PATHS['ball'],
            device=device,
        )
        frame_generator = analyzer.process(video_path, stride=stride)
    elif use_hosted:
        model = get_hosted_model(hosted_model_id, api_key)
        frame_generator = build_hosted_generator(
            mode, video_path, model, confidence, device, stride)
    else:
        frame_generator = build_local_generator(mode, video_path, device, stride)

    st.subheader("📽️ Traitement en direct")
    with st.container(border=True):
        image_placeholder = st.empty()
        progress_bar = st.progress(0.0)
        metric_cols = st.columns(3)
        frame_metric = metric_cols[0].empty()
        fps_metric = metric_cols[1].empty()
        elapsed_metric = metric_cols[2].empty()

    sink: Optional[sv.VideoSink] = None
    output_path = None
    if save_output:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        sink = sv.VideoSink(output_path, video_info)
        sink.__enter__()

    needs_warm_up = mode in (
        "Classification d'équipes", "Radar (terrain + équipes)", PLAYER_ANALYSIS_MODE)
    start_time = time.time()
    frame_count = 0
    try:
        if needs_warm_up:
            with st.spinner(
                "Analyse des équipes en cours (échantillonnage des joueurs)…"
            ):
                first_frame = next(frame_generator)
            frame_count = 1
            if sink is not None:
                sink.write_frame(first_frame)
            image_placeholder.image(
                first_frame, channels="BGR", use_container_width=True)

        for frame in frame_generator:
            frame_count += 1
            if sink is not None:
                sink.write_frame(frame)
            if frame_count % display_every == 0 or frame_count >= total_to_process:
                image_placeholder.image(frame, channels="BGR", use_container_width=True)
                elapsed = time.time() - start_time
                fps_proc = frame_count / elapsed if elapsed > 0 else 0.0
                frame_metric.metric("Frames", f"{frame_count}/{total_to_process}")
                fps_metric.metric("Vitesse", f"{fps_proc:.1f} fps")
                elapsed_metric.metric("Temps écoulé", f"{elapsed:.0f}s")
            progress_bar.progress(min(frame_count / total_to_process, 1.0))
    finally:
        if sink is not None:
            sink.__exit__(None, None, None)

    st.success(f"✅ Traitement terminé : {frame_count} frames analysées en "
               f"{time.time() - start_time:.0f}s.")

    if save_output and output_path:
        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Télécharger la vidéo annotée",
                data=f,
                file_name=f"soccer_ai_{mode.replace(' ', '_')}.mp4",
                mime="video/mp4",
            )

    if analyzer is not None:
        st.session_state['player_report'] = analyzer.report()
    else:
        st.session_state.pop('player_report', None)
elif uploaded_file is None:
    st.info("👆 Upload une vidéo de match pour commencer.")

# --------------------------------------------------------------------------
# Player-by-player dashboard (persists across reruns, e.g. changing the
# player selector below, without re-running the whole video pipeline)
# --------------------------------------------------------------------------

if st.session_state.get('player_report') is not None:
    st.divider()
    st.subheader("📊 Étude joueur par joueur")
    report = st.session_state['player_report']

    if not report:
        st.info("Aucun joueur suffisamment suivi pour établir des statistiques.")
    else:
        team_names = {0: "Équipe A", 1: "Équipe B"}
        rows = []
        row_labels = []
        for p in report:
            label = f"#{p['jersey_number']}" if p['jersey_number'] else f"ID {p['tracker_ids'][0]}"
            row_labels.append(label)
            rows.append({
                "Joueur": label,
                "Équipe": team_names.get(p['team_id'], "?"),
                "Touches de balle": p['touches'],
                "Passes faites": p['passes_made'],
                "Passes reçues": p['passes_received'],
                "Distance parcourue (m)": round(p['distance_m'], 1),
            })

        with st.container(border=True):
            st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.container(border=True):
            selected_label = st.selectbox("Cartographie du joueur", row_labels)
            selected = report[row_labels.index(selected_label)]

            col1, col2, col3 = st.columns(3)
            col1.metric("Touches de balle", selected['touches'])
            col2.metric(
                "Passes faites / reçues",
                f"{selected['passes_made']} / {selected['passes_received']}")
            col3.metric("Distance parcourue", f"{selected['distance_m']:.1f} m")

            if len(selected['trajectory']):
                heatmap = draw_pitch_heatmap(PITCH_CONFIG, xy=selected['trajectory'])
                st.image(
                    heatmap, channels="BGR",
                    caption=f"Zones d'activité sur le terrain — {selected_label}",
                    use_container_width=True,
                )
            else:
                st.info("Pas assez de données de position pour ce joueur.")

st.divider()
st.caption(
    "Soccer AI — construit sur "
    "[Ultralytics YOLO](https://docs.ultralytics.com/), "
    "[Supervision](https://github.com/roboflow/supervision) et "
    "[SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip)."
)
