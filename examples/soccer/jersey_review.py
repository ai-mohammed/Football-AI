"""Inspect original crops before assigning a shirt number to a tracked identity."""
import base64
import pandas as pd
import streamlit as st

from sports.common.roster import apply_numbers, registry
from sports.common.segments import TEAM_NAMES, label

STATUS = {'consensus': 'Lectures concordantes', 'candidate': 'À confirmer',
          'conflict': 'Numéro en conflit', 'unreadable': 'Non lu', 'reviewed': 'Validé manuellement'}


def render_jersey_review(data, original, key, width_options):
    st.subheader('Associer les maillots aux pistes')
    st.caption('Un numéro identifie un joueur dans son équipe. Vérifiez les images originales avant de valider une association. Un numéro non visible reste non lu.')
    people = {p['identity_id']: p for p in data['players']}
    if not people:
        st.info('Aucune piste à identifier sur cet extrait.')
        return
    rows = registry(data)
    st.dataframe(pd.DataFrame([{'ID': p['identity_id'], 'Pistes liées': ', '.join(map(str, p['tracker_ids'])),
                               'Équipe': TEAM_NAMES.get(p['team_id'], 'Non attribuée'),
                               'Maillot': p['jersey_number'] or '—', 'Statut': STATUS[p['status']]}
                              for p in rows]), hide_index=True, **width_options)
    selected = st.selectbox('Piste à identifier', sorted(people),
                            format_func=lambda i: f'{label(people[i])} · {TEAM_NAMES.get(people[i]["team_id"], "Non attribuée")}',
                            key=f'identity_{key}')
    person = people[selected]
    previews = person.get('jersey_previews', [])
    if previews:
        columns = st.columns(len(previews))
        for column, sample in zip(columns, previews):
            with column:
                st.image(base64.b64decode(sample['image_base64']), width=120)
                number = sample.get('number')
                st.caption(f'{sample["time_s"]:.2f} s · {sample["native_resolution"][0]} × {sample["native_resolution"][1]} px')
                st.caption(f'Hypothèse : {number or "non lu"} · score du modèle {sample["confidence"]:.0%}')
        st.caption('Le score exprime la confiance du modèle sur cette image ; ce n’est pas une garantie de lecture correcte.')
    else:
        st.info('Aucune image de maillot exploitable enregistrée pour cette piste. Consultez la vidéo avant toute attribution.')
    if person.get('jersey_candidates'):
        st.write('Lectures retenues : ' + ' · '.join(f'N° {c["number"]} : {c["observations"]} image(s)'
                                                   for c in person['jersey_candidates']))
    with st.form(f'jersey_form_{key}_{selected}'):
        number = st.text_input('Numéro du maillot', value=person.get('jersey_number') or '', max_chars=2)
        submitted = st.form_submit_button('Valider ce numéro')
    if submitted:
        proposed = {**st.session_state.get(key, {}), str(selected): number.strip()}
        try:
            apply_numbers(original, proposed)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state[key] = proposed
            st.rerun()
    if str(selected) in st.session_state.get(key, {}) and st.button('Rétablir la lecture automatique', key=f'reset_{key}_{selected}'):
        st.session_state[key] = {k: v for k, v in st.session_state[key].items() if k != str(selected)}
        st.rerun()
    st.caption('Les validations restent dans cette session et apparaissent sur la carte, dans les tableaux et dans les exports. Les incrustations vidéo correspondent au calcul initial.')
    flat = [{'ID': p['identity_id'], 'Pistes': ','.join(map(str, p['tracker_ids'])),
             'Équipe': TEAM_NAMES.get(p['team_id'], 'Non attribuée'), 'Maillot': p['jersey_number'],
             'Statut': STATUS[p['status']]} for p in rows]
    st.download_button('Associations ID–maillot · CSV', pd.DataFrame(flat).to_csv(index=False).encode('utf-8-sig'),
                       file_name='associations_maillots.csv', mime='text/csv', key=f'registry_{key}')
