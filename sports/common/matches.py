"""Frame-exact long-video partitions and bounded, non-overlapping exports."""
import copy
import math

from sports.common.segments import summarize


def video_time(seconds):
    seconds = max(0., float(seconds))
    whole = int(seconds)
    return f'{whole//3600:02}:{whole//60%60:02}:{whole%60:02}'


def plan_segments(total_frames, fps, max_seconds=12., context_seconds=2.):
    if not math.isfinite(fps) or fps <= 0 or total_frames < 1:
        raise ValueError('La vidéo doit contenir des images et une cadence valide.')
    if not 0 < max_seconds <= 12 or not 0 <= context_seconds <= 5:
        raise ValueError('Segments : 12 secondes maximum ; contexte : de 0 à 5 secondes.')
    step = math.floor(max_seconds*fps + 1e-7)
    if step < 1:
        raise ValueError('Un segment doit contenir au moins une image.')
    context = round(context_seconds*fps)
    return [{'id': f's{i+1:04d}', 'index': i+1, 'start_frame': start,
             'end_frame': min(start+step, total_frames),
             'analysis_start_frame': max(0, start-context),
             'start_s': round(start/fps, 6), 'end_s': round(min(start+step, total_frames)/fps, 6),
             'duration_s': round((min(start+step, total_frames)-start)/fps, 6),
             'status': 'pending'}
            for i, start in enumerate(range(0, int(total_frames), step))]


def crop_analysis(data, start, duration, source_start, segment_id):
    """Discard analysis warm-up; retain crossing passes in their arrival segment.

    Only the preceding context is analysed. It is never duplicated in playback,
    movement, possession, or event totals. Raw IDs remain segment-scoped.
    """
    end = start+duration
    result = copy.deepcopy(data)
    result.update(duration_s=duration, source_start_s=source_start, segment_id=segment_id,
                  identity_scope='segment', clock_basis='source_video')
    result['frames'] = []
    for original in data['frames']:
        if not start-1e-6 <= original['time_s'] < end-1e-6:
            continue
        frame = copy.deepcopy(original)
        frame['time_s'] = round(max(0., frame['time_s']-start), 6)
        frame['dt'] = round(min(frame['dt'], duration-frame['time_s']), 6)
        result['frames'].append(frame)
    result['events'] = []
    for original in data['events']:
        # Untimed identity diagnostics remain in diagnostics, not the timeline.
        if not start-1e-6 <= original.get('time_s', -1) < end-1e-6:
            continue
        event = copy.deepcopy(original)
        event['time_s'] = round(max(0., event['time_s']-start), 6)
        if 'start_s' in event:
            event['start_s'] = round(event['start_s']-start, 6)
            if event['start_s'] < 0:
                event['origin_in_context'] = True
        result['events'].append(event)
    visible = {p['identity_id'] for frame in result['frames']
               for p in frame.get('image_detections', frame['players'])}
    visible.update(p['identity_id'] for frame in result['frames'] for p in frame['players'])
    for event in result['events']:
        visible.update(event[k] for k in ('from', 'to') if k in event)
    result['players'] = [p for p in result['players'] if p['identity_id'] in visible]
    for player in result['players']:
        positions = [(t-start, p) for t, p in zip(player.get('timestamps', []), player.get('trajectory', []))
                     if start <= t < end]
        player['timestamps'] = [round(t, 6) for t, _ in positions]
        player['trajectory'] = [p for _, p in positions]
        motion = []
        for sample in player.get('motion_samples', []):
            a, b = max(start, sample['time_s']-sample['dt']), min(end, sample['time_s'])
            if b > a:
                motion.append({**sample, 'time_s': round(b-start, 6), 'dt': round(b-a, 6),
                               'distance_m': sample['distance_m']*(b-a)/sample['dt']})
        player['motion_samples'] = motion
        for field in ('jersey_previews', 'jersey_evidence'):
            for sample in player.get(field, []):
                if sample.get('time_s') is not None:
                    sample['source_time_s'] = round(source_start+sample['time_s']-start, 6)
                    sample['time_s'] = round(sample['time_s']-start, 6)
                    sample['in_context'] = sample['time_s'] < 0
    summary = summarize(result)
    stats = {p['identity_id']: p for p in summary['players']}
    for player in result['players']:
        row = stats.get(player['identity_id'], {})
        player.update(distance_m=row.get('distance_m', 0.), measured_seconds=row.get('measured_s', 0.),
                      avg_speed_kmh=row.get('speed_kmh'), possession_seconds=row.get('control_s', 0.),
                      touches=row.get('control_episodes', 0),
                      passes_made=sum(e.get('from') == player['identity_id'] and e['type'] == 'probable_pass' for e in result['events']),
                      passes_received=sum(e.get('to') == player['identity_id'] and e['type'] == 'probable_pass' for e in result['events']))
    diag = result['diagnostics']
    diag.update(frames=len(result['frames']), calibrated_frames=sum(f['calibrated'] for f in result['frames']),
                calibration_coverage_pct=round(summary['calibration_pct'], 1),
                unknown_possession_seconds=round(duration*summary['unknown_pct']/100, 3),
                context_seconds=start, event_boundary_policy='arrival_segment',
                identity_scope='segment', match_clock_verified=False,
                replay_detection='not_available', events=result['events'])
    diag['display_interpolated_positions'] = sum(p.get('interpolated', False) for f in result['frames'] for p in f.get('display_players', []))
    diag['optical_flow_calibrated_frames'] = sum(f.get('calibration_method') == 'pitch_optical_flow' for f in result['frames'])
    diag['unresolved_identities'] = sum(not p.get('jersey_number') for p in result['players'])
    return result


def combine_segments(analyses):
    """One continuous timeline; identities remain explicitly scoped to a segment.

    Neither equal tracker IDs nor equal automatic shirt readings prove an identity
    across independent runs. Remap all references instead of silently merging them.
    """
    ordered = sorted(analyses, key=lambda d: d['source_start_s'])
    if not ordered:
        raise ValueError('Aucun segment à réunir.')
    base = ordered[0]['source_start_s']
    result = {k: copy.deepcopy(ordered[0][k]) for k in
              ('schema_version', 'source_video', 'source_resolution', 'source_fps', 'stride', 'pitch')}
    result.update(players=[], frames=[], events=[], segments=[], source_start_s=base,
                  identity_scope='segment', clock_basis='source_video', aggregate=True)
    next_id, previous_end = 1, base
    seen_segments = set()
    for source in ordered:
        sid = source['segment_id']
        start, duration = source['source_start_s'], source['duration_s']
        if (sid in seen_segments or start < previous_end-1e-6 or duration <= 0 or duration > 12
                or source['pitch'] != result['pitch'] or source['source_video'] != result['source_video']):
            raise ValueError('Segments dupliqués, superposés ou incompatibles.')
        seen_segments.add(sid)
        offset = start-base
        ids = {p['identity_id']: next_id+i for i, p in enumerate(source['players'])}
        next_id += len(ids)
        result['segments'].append({'id': sid, 'start_s': offset, 'end_s': offset+duration})
        for original in source['players']:
            player = copy.deepcopy(original)
            player.update(identity_id=ids[original['identity_id']], local_identity_id=original['identity_id'],
                          segment_id=sid)
            player['timestamps'] = [t+offset for t in player.get('timestamps', [])]
            for field in ('motion_samples', 'jersey_previews', 'jersey_evidence'):
                for sample in player.get(field, []):
                    if sample.get('time_s') is not None:
                        sample['time_s'] += offset
            result['players'].append(player)
        for original in source['frames']:
            frame = copy.deepcopy(original)
            frame.update(time_s=round(frame['time_s']+offset, 6), segment_id=sid)
            for field in ('players', 'image_detections', 'display_players'):
                for player in frame.get(field, []):
                    if 'identity_id' in player:
                        # Raw image detections also include referees, who are not
                        # members of the player registry and must not become one.
                        player['identity_id'] = (ids.get(player['identity_id']) if field == 'image_detections'
                                                 else ids[player['identity_id']])
            for field in ('possessor', 'control_candidate'):
                if frame.get(field) is not None:
                    frame[field] = ids.get(frame[field])
            result['frames'].append(frame)
        for original in source['events']:
            event = copy.deepcopy(original)
            event.update(time_s=round(event['time_s']+offset, 6), segment_id=sid)
            if 'start_s' in event:
                event['start_s'] = round(event['start_s']+offset, 6)
            for field in ('from', 'to'):
                if field in event:
                    event[field] = ids[event[field]]
            result['events'].append(event)
        previous_end = start+duration
    result['duration_s'] = previous_end-base
    summary = summarize(result)
    first = ordered[0]['diagnostics']
    result['diagnostics'] = {k: copy.deepcopy(first[k]) for k in
        ('tracker', 'device', 'analysis_profile', 'models', 'ocr_enabled', 'jersey_backend',
         'pitch_dimensions_verified', 'ball_events_available') if k in first}
    result['diagnostics'].update(frames=len(result['frames']),
        calibrated_frames=sum(bool(f['calibrated']) for f in result['frames']),
        calibration_coverage_pct=round(summary['calibration_pct'], 1),
        unknown_possession_seconds=round(result['duration_s']*summary['unknown_pct']/100, 3),
        identity_scope='segment', segment_count=len(ordered),
        identity_merge_policy='no_automatic_cross_segment_merge',
        event_boundary_policy='arrival_segment', replay_detection='not_available',
        match_clock_verified=False, control_episodes_scope='segment')
    return result


def segment_playback(overview, segment_id):
    """Local video clock with the same identity references as the global charts."""
    segment = next(s for s in overview['segments'] if s['id'] == segment_id)
    start, end = segment['start_s'], segment['end_s']
    frames = [{**f, 'time_s': round(f['time_s']-start, 6)} for f in overview['frames']
              if f.get('segment_id') == segment_id]
    events = [{**e, 'time_s': round(e['time_s']-start, 6),
               **({'start_s': round(e['start_s']-start, 6)} if 'start_s' in e else {})}
              for e in overview['events'] if e.get('segment_id') == segment_id]
    return {'frames': frames, 'events': events, 'duration_s': end-start,
            'source_start_s': overview['source_start_s']+start, 'segment_id': segment_id,
            'source_fps': overview['source_fps']}
