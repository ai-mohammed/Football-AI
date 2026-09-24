"""Inference services, imported only when a user opens the upload workflow."""
import json
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import streamlit as st
import supervision as sv

from main import PLAYER_DETECTION_MODEL_PATH, PITCH_DETECTION_MODEL_PATH, BALL_DETECTION_MODEL_PATH
from player_analysis import PlayerMatchAnalyzer
from sports.common.runtime import resolve_device
from sports.common.replay import render_replay

MODEL_PATHS = {'player': PLAYER_DETECTION_MODEL_PATH, 'pitch': PITCH_DETECTION_MODEL_PATH,
               'ball': BALL_DETECTION_MODEL_PATH}
MODEL_IDS = {'player': '17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q',
             'pitch': '1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf', 'ball': '1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V'}


@st.cache_resource(show_spinner=False)
def ensure_models(profile='broadcast'):
    import os
    import gdown
    for name, value in MODEL_PATHS.items():
        if profile == 'aerial' and name != 'player':
            continue
        path = Path(value)
        if path.is_file():
            continue
        if os.environ.get(f'FOOTBALL_{name.upper()}_MODEL'):
            raise FileNotFoundError(f'Modèle personnalisé introuvable : {path.name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        # A failed download must never leave a partial model at the final path.
        temporary = path.with_suffix('.download')
        try:
            if not gdown.download(id=MODEL_IDS[name], output=str(temporary), quiet=True):
                raise RuntimeError(f'Téléchargement du modèle {name} impossible.')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return MODEL_PATHS


def probe_video(video_bytes, suffix):
    with tempfile.TemporaryDirectory(prefix='football-probe-') as directory:
        source = Path(directory) / f'source{suffix}'
        source.write_bytes(video_bytes)
        info = sv.VideoInfo.from_video_path(str(source))
        if info.fps <= 0 or info.total_frames <= 0:
            raise ValueError('Cette vidéo ne contient aucune image lisible.')
        return {'duration': info.total_frames / info.fps, 'fps': info.fps,
                'width': info.width, 'height': info.height}


def analyze_upload(video_bytes, filename, start, seconds, stride, tracker, enable_ocr, progress,
                   profile='broadcast', pitch_length_m=105., pitch_width_m=68.):
    device = resolve_device('auto')
    limit = 30 if device.startswith('cuda') else 8
    if start < 0 or not 0 < seconds <= limit or stride < 1:
        raise ValueError(f'Choisissez un extrait de 0 à {limit} secondes et un échantillonnage positif.')
    weights = ensure_models(profile)
    with tempfile.TemporaryDirectory(prefix='football-segment-') as directory:
        root = Path(directory)
        source = root / ('source' + Path(filename).suffix.lower())
        source.write_bytes(video_bytes)
        clip, final = root / 'clip.mp4', root / 'annotated.mp4'
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', str(start),
            '-i', str(source), '-t', str(seconds), '-an', '-c:v', 'libx264', '-preset', 'fast',
            '-pix_fmt', 'yuv420p', str(clip)], check=True, capture_output=True)
        info = sv.VideoInfo.from_video_path(str(clip))
        if not info.total_frames or not info.fps:
            raise ValueError('Aucune image lisible à cet instant. Choisissez un autre passage.')
        analyzer = PlayerMatchAnalyzer(weights['player'], weights['pitch'], weights['ball'],
            device=device, tracker_backend=tracker, enable_ocr=enable_ocr,
            imgsz=1280 if device.startswith('cuda') else 960, profile=profile,
            pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m)
        count = 0
        total = (info.total_frames + stride - 1) // stride
        for count, _ in enumerate(analyzer.process(str(clip), stride=stride), start=1):
            progress(min(count / total, 1.)*.95)
        if not count:
            raise ValueError('Aucune image n’a pu être analysée.')
        data = analyzer.export(filename, info.fps, stride)
        data['duration_s'] = min(data['duration_s'], info.total_frames / info.fps)
        render_replay(clip, data, final)
        progress(1.)
        data['source_start_s'] = start
        data['diagnostics']['models'] = {k: Path(v).name for k, v in analyzer.model_paths.items()}
        # Detach NumPy values and model state before storing a lightweight session result.
        data = json.loads(json.dumps(data, default=lambda value: value.tolist()))
        return {'data': data, 'video': final.read_bytes()}
