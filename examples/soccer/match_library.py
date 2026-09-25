"""Browse bounded segments of a long video without loading inference models."""
import hashlib
import gzip
import io
import json
import os
from pathlib import Path
import zipfile

import pandas as pd
import requests
import streamlit as st

from sports.common.matches import video_time, segment_playback
from tactical_dashboard import render_dashboard, WIDTH

APP = Path(__file__).resolve().parent
REMOTE_PREFIX = 'https://github.com/ai-mohammed/Football-AI/releases/download/match-'
STATUS = {'pending': 'À analyser', 'running': 'Calcul en cours', 'ready': 'Analysé',
          'failed': 'À reprendre', 'awaiting_publication': 'En cours de publication'}


def catalog():
    paths = {p.parent.name: p for p in (APP/'match_data').glob('*/manifest.json')}
    local = Path(os.environ.get('FOOTBALL_MATCH_ROOT', str(APP.parents[1]/'runs/football/matches')))
    paths.update({p.parent.name: p for p in local.glob('*/manifest.json')})
    return paths


def download(url, max_bytes):
    if not url.startswith(REMOTE_PREFIX):
        raise ValueError('Adresse de résultat non reconnue.')
    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        buffer = io.BytesIO()
        for chunk in response.iter_content(65536):
            buffer.write(chunk)
            if buffer.tell() > max_bytes:
                raise ValueError('Résultat trop volumineux pour ce lecteur.')
        return buffer.getvalue()


@st.cache_data(ttl=30, max_entries=6, show_spinner=False)
def remote_manifest(url):
    data = json.loads(download(url, 2*1024*1024))
    if data.get('match_schema_version') != 1 or not data.get('segments'):
        raise ValueError('Index de match incompatible.')
    if any(not 0 < s['duration_s'] <= 12 for s in data['segments']):
        raise ValueError('L’index contient un segment dépassant douze secondes.')
    return data


@st.cache_data(max_entries=3, show_spinner=False)
def remote_segment(url, sha256):
    content = download(url, 64*1024*1024)
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError('Le fichier reçu est incomplet. Réessayez son chargement.')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if sorted(archive.namelist()) != ['analysis.json', 'annotated.mp4']:
            raise ValueError('Le contenu du segment est incompatible.')
        if sum(p.file_size for p in archive.infolist()) > 80*1024*1024:
            raise ValueError('Ce segment dépasse la taille autorisée.')
        return json.loads(archive.read('analysis.json')), archive.read('annotated.mp4')


@st.cache_data(max_entries=2, show_spinner=False)
def load_overview(location, sha256=None, modified=None):
    content = download(location, 32*1024*1024) if sha256 else Path(location).read_bytes()
    if sha256 and hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError('Vue globale incomplète.')
    with gzip.GzipFile(fileobj=io.BytesIO(content)) as archive:
        raw = archive.read(96*1024*1024+1)
    if len(raw) > 96*1024*1024:
        raise ValueError('Vue globale trop volumineuse.')
    data = json.loads(raw)
    if data.get('schema_version') != 3 or not data.get('aggregate'):
        raise ValueError('Vue globale incompatible.')
    return data


def render_match_library():
    st.title('Analyse du match')
    paths = catalog()
    if not paths:
        st.info('Aucun match complet n’est encore préparé. Les cinq extraits restent disponibles dans le menu.')
        return
    selected_match = st.selectbox('Match', list(paths),
                                  format_func=lambda k: json.loads(paths[k].read_text(encoding='utf-8'))['title'])
    path = paths[selected_match]
    data = json.loads(path.read_text(encoding='utf-8'))
    local = (path.parent/'segments').is_dir()
    if not local and data.get('remote_manifest_url'):
        try:
            data = remote_manifest(data['remote_manifest_url'])
            st.session_state[f'match_index_{selected_match}'] = data
        except (requests.RequestException, ValueError, KeyError):
            data = st.session_state.get(f'match_index_{selected_match}', data)
            st.info('La progression en ligne est momentanément indisponible. Le dernier index connu reste consultable ; réessayez avec Actualiser.')
    segments = data['segments']
    ready = sum(s['status'] == 'ready' for s in segments)
    window = data.get('analysis_window')
    if window:
        st.caption(f'Période analysée : {video_time(window["start_s"])} → {video_time(window["end_s"])} · '
                   f'{len(segments)} segments de 12 secondes maximum · fichier source : {video_time(data["source"]["duration_s"])}')
    else:
        st.caption(f'{video_time(data["source"]["duration_s"])} de vidéo · {len(segments)} segments · 12 secondes maximum par segment')
    st.progress(ready/len(segments), text=f'{ready} / {len(segments)} segments analysés')
    if st.button('Actualiser la progression'):
        remote_manifest.clear()
        st.rerun()
    if data.get('status') == 'running':
        st.caption('Le GPU du PC traite la file. Les résultats sont enregistrés et publiés au fur et à mesure. Le PC doit rester allumé.')
    with st.expander('Découpage, horaires et limites'):
        st.write('La période analysée est découpée sans trou ni recouvrement : 00:00–00:12, 00:12–00:24, puis la suite. '
                 f'Le dernier segment dure {segments[-1]["duration_s"]:.2f} s. Deux secondes en amont peuvent être analysées '
                 'pour retrouver le début d’une action ; elles ne rallongent pas le segment affiché et ne sont pas comptées deux fois.')
        st.write('Les horaires indiquent la position dans le fichier vidéo. Ils ne correspondent pas nécessairement au chronomètre du match. '
                 'La vidéo contient aussi des ralentis, gros plans, pauses et célébrations. Ces séquences ne sont pas exclues automatiquement : '
                 'additionner les événements de tous les segments ne donne pas les statistiques officielles du match.')
        st.write('Les couleurs A/B utilisent une référence commune au fichier. Les IDs sont propres à chaque segment : '
                 'ID7 dans deux segments ne prouve pas qu’il s’agit du même joueur. Les associations équipe–maillot restent à vérifier.')
    overview = None
    if data.get('overview'):
        try:
            descriptor = data['overview']
            with st.spinner('Chargement des graphiques de toute la période…'):
                if local:
                    overview_path = path.parent/'overview.json.gz'
                    overview = load_overview(str(overview_path), modified=overview_path.stat().st_mtime_ns)
                else:
                    overview = load_overview(descriptor['url'], descriptor['sha256'])
            st.info(f'Une seule analyse sur {video_time(overview["duration_s"])}. Les graphiques réunissent les '
                    f'{len(overview["segments"])} segments ; le choix ci-dessous change seulement la vidéo affichée.')
        except (OSError, ValueError, KeyError, requests.RequestException):
            st.error('Les graphiques globaux n’ont pas pu être chargés. Actualisez la progression pour réessayer. '
                     'Les résultats de chaque segment restent accessibles.')
    mode = st.radio('Afficher', ['Analysés', 'Tous les segments', 'À reprendre'], horizontal=True,
                    index=0 if ready else 1, key=f'match_mode_{selected_match}')
    options = [s for s in segments if mode == 'Tous les segments' or
               (s['status'] == 'ready' if mode == 'Analysés' else s['status'] == 'failed')]
    if not options:
        st.info('Aucun segment dans cette catégorie pour le moment.')
    else:
        by_id = {s['id']: s for s in options}
        key = f'match_segment_{selected_match}_{mode}'
        chosen = st.selectbox('Segment de la vidéo', list(by_id),
            format_func=lambda i: f'{by_id[i]["index"]:03d} · {video_time(by_id[i]["start_s"])} → {video_time(by_id[i]["end_s"])} · {STATUS[by_id[i]["status"]]}', key=key)
        position = list(by_id).index(chosen)
        def move(step):
            st.session_state[key] = list(by_id)[position+step]
        a, b = st.columns(2)
        a.button('Segment précédent', disabled=position == 0, on_click=move, args=(-1,), key=f'previous_{key}')
        b.button('Segment suivant', disabled=position == len(options)-1, on_click=move, args=(1,), key=f'next_{key}')
        segment = by_id[chosen]
        st.subheader(f'Segment {segment["index"]:03d} · {video_time(segment["start_s"])} → {video_time(segment["end_s"])}')
        st.caption('Horaires du fichier source · les graphiques couvrent toute la période sélectionnée.' if overview else
                   'Horaires du fichier source · les mesures ci-dessous portent uniquement sur ce segment.')
        if segment['status'] == 'ready':
            try:
                folder = path.parent/'segments'/segment['id']
                with st.spinner('Chargement de ce segment…'):
                    if local and (folder/'analysis.json').is_file():
                        analysis = json.loads((folder/'analysis.json').read_text(encoding='utf-8'))
                        video = folder/'annotated.mp4'
                    else:
                        analysis, video = remote_segment(segment['bundle_url'], segment['bundle_sha256'])
                if overview:
                    revision = data['overview'].get('sha256', '')[:12]
                    render_dashboard(overview, video, f'{selected_match}_overview_{revision}',
                                     playback=segment_playback(overview, chosen))
                else:
                    revision = segment.get('bundle_sha256', '')[:12]
                    render_dashboard(analysis, video, f'{selected_match}_{chosen}_{revision}')
            except (OSError, ValueError, KeyError, requests.RequestException, zipfile.BadZipFile):
                st.error('Le segment n’a pas pu être chargé. Actualisez la progression puis réessayez ; les autres segments restent disponibles.')
        elif segment['status'] == 'failed':
            st.warning('Ce segment doit être repris. Il reste dans la file et aucune statistique n’est présentée comme calculée.')
        elif segment['status'] == 'running':
            st.info('L’analyse de ce segment est en cours. Actualisez la progression pour consulter son résultat une fois le calcul terminé.')
        elif segment['status'] == 'awaiting_publication':
            st.info('Ce segment est analysé ; ses fichiers sont en cours de publication. Actualisez la progression pour les ouvrir dès qu’ils seront disponibles.')
        else:
            st.info('Ce segment est repéré dans la vidéo et attend son traitement. Choisissez un segment analysé pour consulter des résultats.')
    with st.expander('Plan complet des segments et export'):
        table = pd.DataFrame([{'Segment': s['index'], 'Début vidéo': video_time(s['start_s']),
            'Fin vidéo': video_time(s['end_s']), 'Durée (s)': s['duration_s'], 'État': STATUS[s['status']],
            'Terrain calibré (%)': s.get('summary', {}).get('calibration_pct'),
            'Passes probables': s.get('summary', {}).get('passes')} for s in segments])
        st.dataframe(table, hide_index=True, **WIDTH)
        st.download_button('Plan des segments · CSV', table.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'{selected_match}_segments.csv', mime='text/csv')
