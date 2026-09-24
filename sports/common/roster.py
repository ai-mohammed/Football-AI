"""Explicit jersey/track registry; human reviews never invent new observations."""
import hashlib
import json


def review_version(data):
    identity = {'source': data.get('source_video'), 'duration': data.get('duration_s'),
                'players': [(p['identity_id'], p.get('tracker_ids'), p.get('team_id'),
                             p.get('jersey_number'), p.get('jersey_evidence')) for p in data.get('players', [])]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def apply_numbers(data, assignments):
    """Return new player metadata only; raw frames, motion and events stay intact."""
    result = {**data, 'players': [dict(p) for p in data['players']]}
    if set(assignments)-{str(p['identity_id']) for p in data['players']}:
        raise ValueError('Cette piste n’existe pas dans cette analyse.')
    for player in result['players']:
        number = assignments.get(str(player['identity_id']))
        if number is None:
            continue
        if not str(number).isdigit() or not 1 <= int(number) <= 99:
            raise ValueError('Le numéro doit être un entier de 1 à 99.')
        player.update(automatic_jersey_number=player.get('jersey_number'), jersey_number=str(int(number)),
                      jersey_status='reviewed', identity_status='human_reviewed')
    for frame in data.get('frames', []):
        visible = {p['identity_id'] for p in (frame.get('image_detections') or frame.get('players', []))}
        occupied = {}
        for player in result['players']:
            team, number = player.get('team_id'), player.get('jersey_number')
            if player['identity_id'] not in visible or team not in (0, 1) or not number:
                continue
            key = team, number
            if key in occupied and (str(player['identity_id']) in assignments or str(occupied[key]) in assignments):
                raise ValueError(f'Le numéro {number} est déjà attribué à ID{occupied[key]} dans la même équipe, visible au même instant.')
            occupied[key] = player['identity_id']
    return result


def registry(data):
    return [{'identity_id': p['identity_id'], 'tracker_ids': p.get('tracker_ids', [p['identity_id']]),
             'team_id': p.get('team_id'), 'jersey_number': p.get('jersey_number'),
             'automatic_jersey_number': p.get('automatic_jersey_number', p.get('jersey_number')),
             'status': p.get('jersey_status', 'consensus' if p.get('jersey_number') else 'unreadable'),
             'candidates': p.get('jersey_candidates', []), 'evidence': p.get('jersey_evidence', [])}
            for p in data['players']]
