"""Analyse a long video in resumable segments of at most twelve seconds."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'examples/soccer')]

from sports.common.matches import plan_segments, crop_analysis


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'),
                                   default=lambda v: v.tolist()), encoding='utf-8')
    temporary.replace(path)


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_plan(source, output, title, stride):
    import supervision as sv
    info = sv.VideoInfo.from_video_path(str(source))
    if info.fps <= 0 or info.total_frames <= 0:
        raise ValueError('La vidéo ne contient aucune image lisible.')
    fingerprint = checksum(source)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['source']['sha256'] != fingerprint or manifest['processing']['stride'] != stride:
            raise ValueError('La source ou le pas d’analyse diffère du traitement existant. Utilisez un nouveau dossier.')
        manifest['title'] = title
        return manifest, info
    manifest = {'match_schema_version': 1, 'id': output.name, 'title': title,
                'source': {'filename': source.name, 'sha256': fingerprint, 'bytes': source.stat().st_size,
                           'fps': info.fps, 'total_frames': info.total_frames,
                           'width': info.width, 'height': info.height, 'duration_s': info.total_frames/info.fps},
                'created_at': now(), 'updated_at': now(), 'status': 'planned',
                'processing': {'stride': stride, 'max_segment_seconds': 12, 'context_seconds': 2,
                               'identity_scope': 'segment', 'clock_basis': 'source_video',
                               'team_scope': 'match_colour_prototypes', 'replays_included': True},
                'segments': plan_segments(info.total_frames, info.fps)}
    atomic_json(manifest_path, manifest)
    return manifest, info


def fit_match_teams(analyzer, source, info, saved):
    import cv2
    import numpy as np
    import supervision as sv
    from main import get_crops
    from sports.common.team import TeamClassifier
    classifier = TeamClassifier(device=analyzer.device)
    if saved:
        # sklearn also needs fit metadata; the two fixed centres initialise it.
        centers = np.asarray(saved['centers'], dtype=np.float32)
        classifier.cluster_model.fit(centers)
        classifier.cluster_model.cluster_centers_ = centers
        classifier.radii = np.asarray(saved['radii'])
        classifier.fitted = bool(saved['fitted'])
        return classifier, saved
    crops = []
    cap = cv2.VideoCapture(str(source))
    try:
        for frame_index in np.linspace(0, info.total_frames-1, 48, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            result = analyzer._detect_people(frame, conf=.4)
            detections = sv.Detections.from_ultralytics(result)
            players = detections[(detections.class_id == 2)
                                 & ((detections.xyxy[:, 3]-detections.xyxy[:, 1]) < info.height*.25)]
            if len(players) < 6:
                continue  # Fit colours on wide views, not bench/face close-ups.
            crops.extend(c for c in get_crops(frame, players) if c.size)
    finally:
        cap.release()
    # Bound the fit while retaining samples across the whole file.
    crops = [crops[i] for i in np.linspace(0, len(crops)-1, min(400, len(crops)), dtype=int)]
    classifier.fit(crops, allow_role_outliers=True)
    if not classifier.fitted:
        raise ValueError('Impossible de séparer les couleurs des deux équipes sur les plans larges.')
    # Stable colour ordering only; this is not automatic club recognition.
    order = np.argsort(classifier.cluster_model.cluster_centers_[:, 0])
    classifier.cluster_model.cluster_centers_ = classifier.cluster_model.cluster_centers_[order]
    classifier.radii = classifier.radii[order]
    record = {'centers': classifier.cluster_model.cluster_centers_.tolist(),
              'radii': classifier.radii.tolist(), 'fitted': True, 'club_names_verified': False,
              'method': 'jersey_lab_rare_role_rejection',
              'excluded_role_crops': classifier.excluded_role_crops}
    return classifier, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--title', required=True)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--segments', nargs='+', type=int, help='Optional one-based segment numbers; omitted = whole video')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--publish', action='store_true', help='Publish finished bundles to this repository’s dedicated release')
    parser.add_argument('--release-tag', default='match-barcelona-real-2025-26')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, info = load_or_plan(args.video, args.output, args.title, args.stride)
    if args.plan_only:
        print(f'{len(manifest["segments"])} segments planned; final duration {manifest["segments"][-1]["duration_s"]:.2f}s')
        return
    if args.segments and any(i < 1 or i > len(manifest['segments']) for i in args.segments):
        parser.error('Segment number is outside the video.')
    # Prevent two resumed workers from writing this output at once.
    import os
    lock = args.output/'worker.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Un verrou de traitement existe. Vérifiez que le processus est arrêté avant de retirer worker.lock.') from None
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    try:
        run(args, manifest, info)
    finally:
        lock.unlink(missing_ok=True)


def run(args, manifest, info):
    import torch
    from main import PLAYER_DETECTION_MODEL_PATH, PITCH_DETECTION_MODEL_PATH, BALL_DETECTION_MODEL_PATH
    from player_analysis import PlayerMatchAnalyzer
    from sports.common.replay import render_replay
    from sports.common.segments import summarize
    publisher = None
    if args.publish:
        from publish_match import MatchPublisher
        publisher = MatchPublisher(args.release_tag, manifest['title'])
        manifest['remote_manifest_url'] = publisher.index_url
        manifest['release_url'] = publisher.release_url
    analyzer = PlayerMatchAnalyzer(PLAYER_DETECTION_MODEL_PATH, PITCH_DETECTION_MODEL_PATH,
                                   BALL_DETECTION_MODEL_PATH, tracker_backend='botsort', device=args.device)
    classifier, colors = fit_match_teams(analyzer, args.video, info, manifest.get('team_prototypes'))
    manifest.update(team_prototypes=colors, status='running', started_at=now())
    manifest['processing']['device'] = analyzer.device
    manifest['processing']['jersey_backend'] = analyzer.jersey_backend
    manifest_path = args.output/'manifest.json'
    atomic_json(manifest_path, manifest)
    if publisher:
        publisher.index(manifest)
    queue = [s for s in manifest['segments'] if args.segments is None or s['index'] in args.segments]
    for segment in queue:
        folder = args.output/'segments'/segment['id']
        folder.mkdir(parents=True, exist_ok=True)
        bundle = folder/'bundle.zip'
        if segment['status'] == 'ready' and bundle.exists():
            if publisher and not segment.get('bundle_url'):
                segment.update(publisher.segment(bundle, segment['id']))
            continue
        begun = time.perf_counter()
        segment.update(status='running', started_at=now())
        manifest['updated_at'] = now()
        atomic_json(manifest_path, manifest)
        try:
            analyzer.reset_segment()
            # Align context with the core's sampling lattice: first core frame is sampled.
            context_frames = (segment['start_frame']-segment['analysis_start_frame'])//args.stride*args.stride
            analysis_start = segment['start_frame']-context_frames
            for index, _ in enumerate(analyzer.process(str(args.video), stride=args.stride,
                    start_frame=analysis_start, end_frame=segment['end_frame'], team_classifier=classifier)):
                if index and index % 100 == 0:
                    print(f'{segment["id"]}: {index} observations', flush=True)
            data = analyzer.export(args.video.name, info.fps, args.stride)
            data = crop_analysis(data, context_frames/info.fps, segment['duration_s'], segment['start_s'], segment['id'])
            data['diagnostics']['models'] = {k: Path(v).name for k, v in analyzer.model_paths.items()}
            video = folder/'annotated.mp4'
            # Only an incomplete result of this segment is replaced on resumption.
            video.unlink(missing_ok=True)
            render_replay(args.video, data, video, folder/'preview.jpg', source_start_frame=segment['start_frame'])
            atomic_json(folder/'analysis.json', data)
            summary = summarize(data)
            temporary = bundle.with_suffix('.tmp')
            with zipfile.ZipFile(temporary, 'w') as archive:
                archive.write(video, 'annotated.mp4', compress_type=zipfile.ZIP_STORED)
                archive.write(folder/'analysis.json', 'analysis.json', compress_type=zipfile.ZIP_DEFLATED)
            temporary.replace(bundle)
            segment.update(status='ready', finished_at=now(), elapsed_s=round(time.perf_counter()-begun, 2),
                           analysis_path=f'segments/{segment["id"]}/analysis.json',
                           video_path=f'segments/{segment["id"]}/annotated.mp4',
                           summary={'calibration_pct': round(summary['calibration_pct'], 1),
                                    'ball_pct': round(summary['ball_pct'], 1), 'passes': summary['passes'],
                                    'jerseys': sum(bool(p['jersey_number']) for p in data['players']),
                                    'identities': len(data['players'])})
            segment.pop('error', None)
            if publisher:
                try:
                    segment.update(publisher.segment(bundle, segment['id']))
                    segment.pop('publication_error', None)
                except Exception as error:
                    segment['publication_error'] = type(error).__name__
                    print(f'{segment["id"]}: publication failed ({type(error).__name__}); local result retained', flush=True)
        except Exception as error:
            segment.update(status='failed', error=type(error).__name__)
            print(f'{segment["id"]}: analysis failed ({type(error).__name__}: {error})', flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        manifest['updated_at'] = now()
        atomic_json(manifest_path, manifest)
        if publisher:
            try:
                publisher.index(manifest)
            except Exception as error:
                print(f'Progress publication delayed: {type(error).__name__}', flush=True)
        ready = sum(s['status'] == 'ready' for s in manifest['segments'])
        print(f'{segment["id"]}: {segment["status"]}; {ready}/{len(manifest["segments"])} ready; {time.perf_counter()-begun:.1f}s', flush=True)
    manifest['status'] = 'complete' if all(s['status'] == 'ready' for s in manifest['segments']) else 'partial'
    manifest['updated_at'] = now()
    atomic_json(manifest_path, manifest)
    if publisher:
        publisher.index(manifest)


if __name__ == '__main__':
    main()
