"""Football AI: a fast tactical workspace for video segments."""
import hashlib
from pathlib import Path
import sys
from uuid import uuid4

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
for path in (APP_DIR, APP_DIR.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tactical_dashboard import WIDTH, load_analysis, render_dashboard

st.set_page_config(page_title='Football AI · Analyse tactique', page_icon='⚽',
                   layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
[data-testid="stAppViewContainer"]{font-variant-numeric:tabular-nums}
.block-container{padding:1.7rem 2rem 2rem;max-width:1560px}
h1{font-size:1.85rem!important;letter-spacing:-.035em;padding-top:0!important}
h2,h3{letter-spacing:-.02em}h3{font-size:1.08rem!important;margin-top:.2rem}
[data-testid="stSidebar"]{border-right:1px solid #293441}
[data-testid="stSidebar"] .block-container{padding-top:1.6rem}
[data-testid="stSidebar"] h1{font-size:1.35rem!important}
[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:600}
[data-testid="stMetricLabel"]{color:#A7B4C4}
[data-testid="stCaptionContainer"]{color:#A7B4C4}
[data-baseweb="tab-list"]{gap:1.75rem;border-bottom:1px solid #293441}
[data-baseweb="tab"]{padding:10px 0;font-size:14px}
[data-testid="stMetric"]{padding:8px 0}
[data-testid="stSidebar"] [data-testid="stImage"] img{border-radius:6px}
@media(max-width:720px){.block-container{padding:4rem 1rem 1rem}[data-baseweb="tab-list"]{gap:1rem;overflow-x:auto}h1{font-size:1.5rem!important}}
</style>''', unsafe_allow_html=True)

DEMO_DIR = APP_DIR / 'demo_data'
CLIPS = {'08fd33_0': 'Extrait 01', '0bfacc_0': 'Extrait 02', '121364_0': 'Extrait 03',
         '2e57b9_0': 'Extrait 04', '573e61_0': 'Extrait 05'}
demos = [p for p in sorted(DEMO_DIR.iterdir()) if p.is_dir() and (p / 'analysis.json').is_file()] if DEMO_DIR.exists() else []

with st.sidebar:
    st.title('Football AI')
    st.caption('L’espace d’analyse de vos extraits')
    page = st.radio('Espace de travail', ['Extraits analysés', 'Importer une vidéo'], label_visibility='collapsed')
    st.divider()
    if page == 'Extraits analysés' and demos:
        selected = st.selectbox('Bibliothèque', demos, format_func=lambda p: CLIPS.get(p.name, p.name), key='demo')
        data_file = selected / 'analysis.json'
        data = load_analysis(str(data_file), data_file.stat().st_mtime_ns)
        if (selected / 'preview.jpg').is_file():
            st.image(str(selected / 'preview.jpg'), **WIDTH)
        st.markdown(f'**{data["duration_s"]:g} secondes** · {len(data["frames"])} observations')
        st.caption('Calculé sur GPU, prêt à explorer. Aucun calcul IA n’est relancé pendant la lecture.')
        with st.expander('À propos de cet extrait'):
            st.write(f'Source : {data["source_video"]}')
            st.write('Suivi BoT-SORT, lecture des maillots et calibration du terrain. Les mesures restent des estimations.')
    st.divider()
    st.caption('Développé par Mohammed ADDI')
    st.markdown('[Code & méthode](https://github.com/ai-mohammed/Football-AI)')
    st.caption('Version · Atelier tactique 3')

if page == 'Extraits analysés':
    st.title('Analyse tactique')
    if not demos:
        st.info('La bibliothèque est en cours de préparation. Vous pouvez importer un extrait.')
    else:
        st.caption(f'{CLIPS.get(selected.name, selected.name)} · {data["duration_s"]:g} s · Vidéo, positions et actions au même instant')
        render_dashboard(data, selected / 'annotated.mp4', selected.name)
else:
    st.title('Analyser un extrait')
    st.write('Importez votre vidéo, choisissez un court passage, puis explorez le même tableau de bord.')
    uploaded = st.file_uploader('Vidéo de football', type=['mp4', 'mov', 'avi', 'mkv'],
                                help='300 Mo maximum. Une vue large du terrain donne des positions plus exploitables.')
    if uploaded is None:
        st.info('Pour une démonstration immédiate, ouvrez « Extraits analysés » dans le menu.')
    else:
        try:
            from analysis_service import analyze_upload, probe_video, resolve_device
            from player_analysis import ocr_available
            contents = uploaded.getvalue()
            fingerprint = hashlib.sha256(contents).hexdigest()[:16]
            if st.session_state.get('upload_fingerprint') != fingerprint:
                info = probe_video(contents, Path(uploaded.name).suffix.lower())
                st.session_state.update(upload_fingerprint=fingerprint, upload_info=info)
                st.session_state.pop('upload_result', None)
            info = st.session_state['upload_info']
            gpu = resolve_device('auto').startswith('cuda')
            maximum = min(30. if gpu else 8., info['duration'])
            st.caption(f'Vidéo : {info["duration"]:.1f} s · {info["width"]} × {info["height"]} · {"GPU disponible" if gpu else "Calcul sur CPU : extrait limité à 8 s, traitement plus lent"}')
            a, b = st.columns(2)
            start = a.number_input('Début du passage (s)', min_value=0., max_value=max(0., info['duration']-.1),
                                   value=0., step=.5, key=f'start_{fingerprint}')
            duration = b.number_input('Durée à analyser (s)', min_value=.1,
                max_value=max(.1, min(maximum, info['duration']-start)),
                value=min(5., max(.1, min(maximum, info['duration']-start))), step=.5, key=f'duration_{fingerprint}_{start}')
            with st.expander('Réglages de l’analyse'):
                tracker = st.selectbox('Suivi', ['bytetrack', 'botsort'],
                    format_func=lambda t: 'ByteTrack · rapide' if t == 'bytetrack' else 'BoT-SORT · suivi avec apparence')
                stride = st.select_slider('Échantillonnage : une image sur', [1, 2, 3, 4, 5], value=2 if gpu else 4)
                enable_ocr = st.checkbox('Lire les numéros de maillot', value=ocr_available(), disabled=not ocr_available())
                if not ocr_available():
                    st.caption('La lecture de maillots n’est pas installée sur cet hébergement. Les pistes conservent des IDs.')
            if st.button('Analyser ce passage', type='primary'):
                with st.status('Préparation des modèles et des équipes…', expanded=True) as status:
                    progress = st.progress(0., text='Première analyse : le téléchargement des modèles peut prendre quelques minutes.')
                    result = analyze_upload(contents, uploaded.name, start, duration, stride, tracker, enable_ocr,
                        lambda value: progress.progress(value, text=f'Analyse des images · {value:.0%}'))
                    result['id'] = f'{fingerprint}_{uuid4().hex[:8]}'
                    st.session_state['upload_result'] = result
                    status.update(label='Extrait analysé', state='complete', expanded=False)
            if st.session_state.get('upload_result'):
                result = st.session_state['upload_result']
                st.divider()
                st.subheader('Votre analyse')
                source_start = result['data'].get('source_start_s', 0.)
                st.caption(f'Passage source {source_start:.1f}–{source_start+result["data"]["duration_s"]:.1f} s. Le lecteur ci-dessous commence à 0 s.')
                render_dashboard(result['data'], result['video'], result['id'])
        except Exception as exc:
            st.error(f'Analyse indisponible : {exc}')
            st.caption('Essayez un extrait MP4 plus court ou consultez un exemple déjà analysé.')
