"""Interactive exploration of precomputed, timestamped football observations."""
import base64
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st
import streamlit.components.v1 as components

from sports.common.segments import (EVENT_NAMES, TEAM_NAMES, label, occupancy,
                                    pass_network, structure, summarize)
from sports.common.roster import apply_numbers, registry, review_version
from jersey_review import render_jersey_review

WIDTH = ({'width': 'stretch'} if tuple(map(int, st.__version__.split('.')[:2])) >= (1, 49)
         else {'use_container_width': True})
COLORS = {0: '#FF1493', 1: '#00BFFF', None: '#A7B4C4'}
PLAYER_COLUMNS = {'label': 'Piste / maillot', 'team': 'Équipe', 'visible_s': 'Visible (s)',
                  'distance_m': 'Distance observée (m)', 'speed_kmh': 'Vitesse mesurée (km/h)',
                  'measured_s': 'Mouvement mesuré (s)', 'control_s': 'Contrôle estimé (s)',
                  'control_episodes': 'Épisodes de contrôle', 'passes_made': 'Passes probables',
                  'passes_received': 'Réceptions probables'}
PLAYER_COMPONENT = components.declare_component('football_tactical_replay',
    path=str(Path(__file__).parent / 'components/tactical_player'))


@st.cache_data(show_spinner=False, max_entries=8)
def load_analysis(path, modified):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@st.cache_data(show_spinner=False, max_entries=8)
def video_base64(path, modified):
    return base64.b64encode(Path(path).read_bytes()).decode('ascii')


def chart(figure, key):
    st.plotly_chart(figure, key=key, config={'displayModeBar': False, 'responsive': True}, **WIDTH)


def style(figure, height=265):
    figure.update_layout(height=height, margin=dict(l=16, r=16, t=12, b=35),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#12201F',
        font=dict(family='Arial, sans-serif', color='#B6C1CF', size=12),
        legend=dict(orientation='h', y=1.18, x=0),
        xaxis=dict(gridcolor='#2D3B43', zeroline=False),
        yaxis=dict(gridcolor='#2D3B43', zeroline=False), hoverlabel=dict(bgcolor='#161B22'))
    return figure


def pitch(data):
    length, width = data['pitch']['length'], data['pitch']['width']
    fig = style(go.Figure(), 320)
    line = dict(color='#64847A', width=1)
    fig.add_shape(type='rect', x0=0, y0=0, x1=length, y1=width, line=line)
    fig.add_shape(type='line', x0=length / 2, y0=0, x1=length / 2, y1=width, line=line)
    fig.add_shape(type='circle', x0=length / 2 - 9.15, x1=length / 2 + 9.15,
                  y0=width / 2 - 9.15, y1=width / 2 + 9.15, line=line)
    for a, b in ((0, 16.5), (length - 16.5, length)):
        fig.add_shape(type='rect', x0=a, x1=b, y0=width / 2 - 20.16, y1=width / 2 + 20.16, line=line)
    fig.update_xaxes(range=[-3, length + 3], showgrid=False, visible=False, constrain='domain')
    fig.update_yaxes(range=[width + 3, -3], showgrid=False, visible=False, scaleanchor='x', scaleratio=1)
    fig.update_layout(margin=dict(l=0, r=0, t=8, b=8), showlegend=False)
    return fig


def heatmap(data, summary, team, identity=None):
    length, width = data['pitch']['length'], data['pitch']['width']
    grid = occupancy(summary, team, length, width, identity=identity)
    fig = pitch(data)
    scale = [[0, '#122A25'], [.5, '#267653'], [1, '#86E6A9']]
    maximum = max(max(map(max, grid)), 1e-9)
    xs, ys = [(i+.5)*length/6 for i in range(6)], [(i+.5)*width/4 for i in range(4)]
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=grid, zmin=0, zmax=maximum,
        colorscale=scale, showscale=False,
        hovertemplate='Zone · %{z:.1f}% du temps de présence cumulé<extra></extra>'))
    for row, y in zip(grid, ys):
        for value, x in zip(row, xs):
            if value < .5:
                continue
            rgb = sample_colorscale(scale, [value/maximum])[0]
            channels = [float(c)/255 for c in rgb[4:-1].split(',')]
            linear = [c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in channels]
            luminance = sum(a*b for a,b in zip(linear, (.2126,.7152,.0722)))
            # Black or white guarantees >=4.5:1 throughout this continuous scale.
            color = '#000000' if luminance > .179 else '#FFFFFF'
            fig.add_annotation(x=x, y=y, text=f'{value:.0f}%', showarrow=False,
                               font=dict(color=color, size=13))
    return fig


def network(data, summary, team):
    fig = pitch(data)
    nodes, edges = pass_network(summary, team)
    names = {p['identity_id']: label(p) for p in data['players']}
    for (a, b), count in edges.items():
        start, end = nodes[a], nodes[b]
        fig.add_trace(go.Scatter(x=[start[0], end[0]], y=[start[1], end[1]], mode='lines',
            line=dict(color=COLORS[team], width=min(7, 1.5 + count)),
            text=f'{names[a]} → {names[b]} · {count} passe(s) probable(s)',
            hovertemplate='%{text}<extra></extra>'))
        fig.add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1], xref='x', yref='y',
            axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=1.5, arrowcolor=COLORS[team], standoff=12, text='')
    if nodes:
        fig.add_trace(go.Scatter(x=[p[0] for p in nodes.values()], y=[p[1] for p in nodes.values()],
            mode='markers+text' if len(nodes) <= 18 else 'markers',
            text=[names[i] for i in nodes], textposition='top center',
            marker=dict(size=14, color=COLORS[team], line=dict(color='#E6EDF3', width=1)),
            hovertemplate='%{text}<extra></extra>'))
    else:
        fig.add_annotation(x=data['pitch']['length'] / 2, y=data['pitch']['width'] / 2,
                           text='Aucune passe probable localisable', showarrow=False, font=dict(color='#E6EDF3'))
    return fig


def possession_chart(summary, time_label='Temps dans l’extrait (s)'):
    fig = style(go.Figure(), 128)
    runs = []
    for frame in summary['frames']:
        person = next((p for p in frame['players'] if p['identity_id'] == frame['possessor']), None)
        team = person.get('team_id') if person and frame['calibrated'] and frame['ball'] is not None else None
        if team not in (0, 1):
            team = None
        start = max(summary['start_s'], frame['time_s'])
        end = start + frame['weight_s']
        if runs and runs[-1][0] == team and abs(runs[-1][2] - start) < .001:
            runs[-1][2] = end
        else:
            runs.append([team, start, end])
    for team in (0, 1, None):
        selected = [r for r in runs if r[0] == team]
        fig.add_trace(go.Bar(x=[r[2] - r[1] for r in selected], base=[r[1] for r in selected],
            y=['Contrôle'] * len(selected), orientation='h', name=TEAM_NAMES.get(team, 'Inconnu') if team is not None else 'Indéterminé',
            marker_color=COLORS[team], customdata=[[r[1], r[2]] for r in selected],
            hovertemplate='%{customdata[0]:.2f} – %{customdata[1]:.2f} s<extra>%{fullData.name}</extra>'))
    fig.update_layout(barmode='overlay', bargap=.4, margin=dict(l=0, r=8, t=24, b=30))
    fig.update_xaxes(range=[summary['start_s'], summary['end_s']], title=time_label)
    fig.update_yaxes(visible=False)
    return fig


def player_table(summary):
    rows = [{**p, 'team': TEAM_NAMES.get(p['team_id'], 'Non attribuée')} for p in summary['players']]
    return pd.DataFrame(rows, columns=list(PLAYER_COLUMNS)).rename(columns=PLAYER_COLUMNS)


def event_table(events, data):
    names = {p['identity_id']: label(p) for p in data['players']}
    return pd.DataFrame([{'Temps (s)': round(e['time_s'], 2), 'Événement': EVENT_NAMES[e['type']],
        'Origine': names.get(e.get('from'), '—'), 'Destination': names.get(e.get('to'), '—'),
        'Équipe à réception': TEAM_NAMES.get(e.get('team_id'), '—')} for e in events],
        columns=['Temps (s)', 'Événement', 'Origine', 'Destination', 'Équipe à réception'])


def render_dashboard(data, video_path, clip_id, playback=None):
    if data.get('schema_version') != 3 or not data.get('frames'):
        st.info('Cet extrait ne contient pas encore de positions horodatées. Régénérez sa démo avec scripts/build_demos.py.')
        return
    original = data
    review_key = f'jersey_reviews_{clip_id}_{review_version(data)}'
    data = apply_numbers(data, st.session_state.get(review_key, {}))
    duration = float(data['duration_s'])
    ball_available = data.get('diagnostics', {}).get('ball_events_available', True)
    if data.get('diagnostics', {}).get('analysis_profile') == 'aerial':
        st.info('Vue drone expérimentale : positions et déplacements des pistes visibles. '
                f'Les distances utilisent un terrain de référence de {data["pitch"]["length"]:g} × {data["pitch"]["width"]:g} m ; ses dimensions réelles ne sont pas vérifiées. '
                'Maillots, ballon et passes non analysés dans ce mode.')
    if data.get('diagnostics', {}).get('calibration_coverage_pct', 100) < 60:
        st.info('Le terrain est partiellement calibré sur cet extrait. Les cartes et distances ne couvrent que les instants exploitables ; la vidéo reste consultable en entier.')
    left, right = st.columns([2.1, 1])
    with left:
        start, end = st.slider('Période étudiée', 0., duration, (0., duration), step=.1,
                              format='%.1f s', key=f'window_{clip_id}')
    with right:
        players = {p['identity_id']: p for p in data['players']}
        focus = st.selectbox('Suivre une piste', [None, *sorted(players)],
            format_func=lambda p: 'Tous les joueurs visibles' if p is None else f'{label(players[p])} · {TEAM_NAMES.get(players[p]["team_id"], "Non attribuée")}',
            key=f'focus_{clip_id}')
    if end - start < .05:
        st.info('Élargissez la période pour afficher les observations.')
        return
    summary = summarize(data, start, end)
    path = None if isinstance(video_path, bytes) else Path(video_path)
    video = video_path if path is None else path.read_bytes() if path.is_file() else None
    if video:
        encoded = base64.b64encode(video).decode('ascii') if path is None else video_base64(str(path), path.stat().st_mtime_ns)
        shown = playback if playback is not None else data
        media_id = f'{clip_id}:{shown.get("segment_id", "")}'
        if playback is not None:
            offset = shown['source_start_s']
            st.caption(f'Vidéo : {offset:.0f}–{offset+shown["duration_s"]:.0f} s du fichier. '
                       f'Graphiques : {start:.0f}–{end:.0f} s, tous les segments de cette période réunis.')
        PLAYER_COMPONENT(video_base64=encoded,
            media_id=f'{media_id}:{path.stat().st_mtime_ns}' if path is not None else media_id,
            frames=shown['frames'], events=shown['events'], source_fps=shown.get('source_fps', 25),
            source_start_s=shown.get('source_start_s', 0.), show_source_clock='source_start_s' in shown,
            pitch=data['pitch'], ball_available=ball_available,
            labels={str(k): label(v) for k, v in players.items()},
            short_labels={str(k): f'ID{v.get("local_identity_id", k)}' for k, v in players.items()},
            numbers={str(k): v.get('jersey_number') for k, v in players.items()},
            start=0. if playback is not None else start,
            end=shown['duration_s'] if playback is not None else end,
            focus=focus, key=f'replay_{clip_id}', default=None)
    else:
        st.warning('La vidéo annotée est indisponible. Les mesures restent consultables ci-dessous.')
    cols = st.columns(4)
    known = summary['possession_pct'][0]
    cols[0].metric('Contrôle A / B', f'{known:.0f} / {100-known:.0f} %' if known is not None else 'Indéterminé',
                   help='Part du temps de contrôle attribuable. Ce n’est pas la possession officielle du match.')
    cols[1].metric('Temps indéterminé', f'{summary["unknown_pct"]:.0f} %')
    cols[2].metric('Passes probables', summary['passes'] if ball_available else 'Non analysées')
    cols[3].metric('Terrain calibré', f'{summary["calibration_pct"]:.0f} %')
    st.caption(f'Mesures sur {start:.1f}–{end:.1f} s uniquement · ID = piste de suivi ; N° = consensus de lectures ou validation dans Maillots.')
    if data.get('aggregate'):
        st.caption('Vue cumulée : chaque piste garde son segment (S001, S002…). Les mêmes numéros ou IDs '
                   'dans deux segments ne sont pas fusionnés automatiquement. Les épisodes de contrôle restent bornés par segment.')
    tactical, individual, jerseys, events_tab, quality = st.tabs(['Tactique', 'Joueurs', 'Maillots', 'Événements', 'Fiabilité & exports'])
    with tactical:
        st.subheader('Rythme du contrôle', divider=False)
        if ball_available:
            chart(possession_chart(summary, 'Temps dans la vidéo analysée (s)' if data.get('aggregate')
                                   else 'Temps dans l’extrait (s)'), f'control_{clip_id}')
        else:
            st.caption('Contrôle et possession non analysés dans le profil drone. Les cartes ci-dessous décrivent les joueurs visibles.')
        team = st.radio('Équipe étudiée', [0, 1], format_func=lambda v: TEAM_NAMES[v], horizontal=True,
                        key=f'team_{clip_id}')
        a, b = st.columns(2)
        with a:
            st.subheader('Occupation du terrain')
            chart(heatmap(data, summary, team), f'occupancy_{clip_id}')
            st.caption('Part du temps de présence cumulé dans chaque zone. Joueurs visibles et terrain calibré uniquement.')
        with b:
            st.subheader('Réseau de passes')
            if ball_available:
                chart(network(data, summary, team), f'network_{clip_id}')
                st.caption('Positions moyennes des pistes reliées par une passe probable. Les flèches indiquent le sens ; survolez pour le détail.')
            else:
                st.info('Réseau indisponible : le ballon n’est pas analysé dans ce profil. Consultez les déplacements dans l’onglet Joueurs.')
        st.subheader('Structure des joueurs visibles')
        points = structure(summary, team)
        fig = style(go.Figure(), 225)
        for key, name, color in [('width_m', 'Largeur', '#86E6A9'), ('depth_m', 'Profondeur', '#A7B4C4')]:
            fig.add_trace(go.Scatter(x=[p['time_s'] for p in points], y=[p[key] for p in points],
                mode='lines', name=name, line=dict(color=color, width=2), connectgaps=False,
                hovertemplate='%{x:.2f} s · %{y:.1f} m<extra>%{fullData.name}</extra>'))
        fig.update_xaxes(title='Temps (s)', range=[start, end]); fig.update_yaxes(title='Étendue (m)', rangemode='tozero')
        chart(fig, f'structure_{clip_id}')
        st.caption('Écart entre les joueurs extrêmes : largeur sur l’axe transversal, profondeur sur l’axe longitudinal. Au moins trois pistes visibles ; ce n’est pas la formation complète de l’équipe.')
    with individual:
        st.subheader('Comparer les déplacements')
        available = [p['identity_id'] for p in summary['players']]
        default = [focus] if focus in available else available[:1]
        selected = st.multiselect('Pistes à comparer', available, default=default, max_selections=2,
            format_func=lambda p: f'{label(players[p])} · {TEAM_NAMES.get(players[p]["team_id"], "Non attribuée")}',
            key=f'compare_{clip_id}')
        if selected:
            a, b = st.columns(2)
            fig, speed = pitch(data), style(go.Figure(), 320)
            for i, identity in enumerate(selected):
                color = ['#86E6A9', '#F7C873'][i]
                # Only accepted motion intervals connect positions; gaps remain gaps.
                samples = players[identity].get('motion_samples', [])
                accepted = {round(s['time_s'], 4) for s in samples}
                xs, ys = [], []
                for t, x, y, _, _ in summary['positions'].get(identity, []):
                    if round(t, 4) not in accepted:
                        xs.append(None); ys.append(None)
                    xs.append(x); ys.append(y)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name=label(players[identity]),
                    line=dict(color=color, width=2), marker=dict(size=3), connectgaps=False,
                    hovertemplate='%{x:.1f}, %{y:.1f} m<extra>%{fullData.name}</extra>'))
                tx, speeds, last = [], [], None
                for sample in samples:
                    t = sample['time_s']
                    if start <= t < end:
                        if last is not None and t - last > sample['dt'] * 1.5:
                            tx.append(None); speeds.append(None)
                        tx.append(t); speeds.append(sample['speed_kmh']); last = t
                speed.add_trace(go.Scatter(x=tx, y=speeds, name=label(players[identity]), mode='lines',
                    line=dict(color=color, width=2), connectgaps=False,
                    hovertemplate='%{x:.2f} s · %{y:.1f} km/h<extra>%{fullData.name}</extra>'))
            fig.update_layout(showlegend=True)
            speed.update_xaxes(title='Temps (s)', range=[start, end]); speed.update_yaxes(title='Vitesse estimée (km/h)', rangemode='tozero')
            with a: chart(fig, f'trajectories_{clip_id}')
            with b: chart(speed, f'speeds_{clip_id}')
            if len(selected) == 1:
                with st.expander('Heatmap de la piste sélectionnée'):
                    chart(heatmap(data, summary, None, selected[0]), f'player_heat_{clip_id}')
        else:
            st.info('Sélectionnez une ou deux pistes ayant une position sur le terrain pendant cette période.')
        st.dataframe(player_table(summary), hide_index=True, **WIDTH)
        st.caption('La distance exclut les pertes de suivi, coupures et sauts invraisemblables. La vitesse est sensible aux erreurs de calibration. Plusieurs pistes peuvent correspondre à une même personne ; le tableau ne constitue pas un effectif de onze joueurs.')
    with jerseys:
        render_jersey_review(data, original, review_key, WIDTH)
    with events_tab:
        st.subheader('Transitions observées')
        choices = st.multiselect('Types d’événements', list(EVENT_NAMES), default=list(EVENT_NAMES),
                                format_func=lambda t: EVENT_NAMES[t], key=f'event_types_{clip_id}')
        visible_events = [e for e in summary['events'] if e['type'] in choices]
        if visible_events:
            st.dataframe(event_table(visible_events, data), hide_index=True, **WIDTH)
        else:
            st.info('Aucun événement de ce type sur la période sélectionnée.' if ball_available else
                    'Les événements de ballon ne sont pas analysés dans le profil drone.')
        st.caption('Cliquez sur une action sous la vidéo pour la revoir. Les transitions reposent sur la proximité ballon–joueur : elles ne prouvent pas une passe réussie, une interception ou une récupération.')
    with quality:
        st.subheader('Ce que l’extrait permet de mesurer')
        a, b, c = st.columns(3)
        a.metric('Ballon localisé sur le terrain', f'{summary["ball_pct"]:.0f} %' if ball_available else 'Non analysé')
        b.metric('Pistes cumulées par segment' if data.get('aggregate') else 'Pistes cartographiées', len(summary['players']))
        c.metric('Pistes avec lecture de maillot' if data.get('aggregate') else 'Maillots confirmés',
                 sum(bool(players[p['identity_id']].get('jersey_number')) for p in summary['players'])
                 if data.get('diagnostics', {}).get('ocr_enabled', True) or st.session_state.get(review_key) else 'Non analysés')
        st.write('Les calculs portent sur les joueurs dans le champ de la caméra. Les périodes sans calibration ou sans contrôle identifiable restent explicitement inconnues.')
        st.caption(f'Terrain de référence : {data["pitch"]["length"]:g} × {data["pitch"]["width"]:g} m. Dimensions réelles du stade non vérifiées. Tirs, xG, fautes et corners ne sont pas déduits de ces données.')
        with st.expander('Méthode et paramètres du calcul'):
            diagnostics = {k: v for k, v in data.get('diagnostics', {}).items() if k != 'events'}
            st.json(diagnostics)
            st.caption('Les numéros automatiques sont des consensus de lectures. Les validations humaines sont identifiées dans Maillots. Les équipes A et B sont des groupes de couleurs, sans identification du club.')
        st.subheader('Exporter cette période')
        a, b, c = st.columns(3)
        a.download_button('Joueurs · CSV', player_table(summary).to_csv(index=False).encode('utf-8-sig'),
            file_name=f'{clip_id}_{start:g}-{end:g}_joueurs.csv', mime='text/csv', **WIDTH)
        b.download_button('Événements · CSV', event_table(summary['events'], data).to_csv(index=False).encode('utf-8-sig'),
            file_name=f'{clip_id}_{start:g}-{end:g}_evenements.csv', mime='text/csv', **WIDTH)
        export = {'schema_version': 3, 'source_video': data['source_video'],
                  'identity_registry': registry(data),
                  'source_start_s': data.get('source_start_s', 0.), 'pitch': data['pitch'],
                  'diagnostics': {k: v for k, v in data.get('diagnostics', {}).items() if k != 'events'},
                  'window': {'start_s': start, 'end_s': end},
                  'metrics': {k: v for k, v in summary.items() if k not in ('positions', 'frames', 'events', 'players')},
                  'players': summary['players'], 'frames': summary['frames'], 'events': summary['events']}
        c.download_button('Données · JSON', json.dumps(export, ensure_ascii=False, indent=2),
            file_name=f'{clip_id}_{start:g}-{end:g}_analyse.json', mime='application/json', **WIDTH)
        if video:
            st.download_button('Vidéo annotée de l’extrait complet', video,
                file_name=f'{clip_id}.mp4', mime='video/mp4')
