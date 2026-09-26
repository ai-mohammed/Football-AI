"""Publish a complete continuous replay; leave historic segment bundles available."""
import argparse
import gzip
import json
from pathlib import Path

from process_full_match import atomic_json, now
from publish_match import MatchPublisher


def validate_analysis(data, manifest):
    duration = manifest['analysis_window']['duration_s']
    if (data.get('identity_scope') != 'continuous' or data.get('duration_s') != duration
            or data.get('source_video') != manifest['source']['filename']
            or data.get('diagnostics', {}).get('chapter_resets') != 0
            or not data.get('frames')):
        raise ValueError('Continuous analysis does not match the requested window.')
    frames = data['frames']
    if abs(frames[0]['time_s']) > 1e-6 or abs(frames[-1]['time_s']+frames[-1]['dt']-duration) > 1e-5:
        raise ValueError('Incomplete continuous timeline.')
    if any(abs(a['time_s']+a['dt']-b['time_s']) > 1e-5 for a,b in zip(frames,frames[1:])):
        raise ValueError('A gap or overlap exists in the timeline.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--continuous-directory', type=Path, required=True)
    parser.add_argument('--tag', default='match-barcelona-real-2025-26')
    parser.add_argument('--catalog', type=Path, help='Update the committed fallback index too')
    args = parser.parse_args()
    manifest_path = args.directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    analysis_path = args.continuous_directory/'analysis.json.gz'
    video_path = args.continuous_directory/'annotated.mp4'
    data = json.loads(gzip.decompress(analysis_path.read_bytes()))
    validate_analysis(data, manifest)
    import cv2
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or abs(fps-data['source_fps']) > 1e-5 or frames != round(fps*data['duration_s']):
            raise ValueError('Incomplete continuous video.')
    finally:
        capture.release()
    publisher = MatchPublisher(args.tag, manifest['title'])
    print('Publishing analysis…', flush=True)
    analysis = publisher.overview(analysis_path)
    print('Publishing continuous video…', flush=True)
    media = publisher.continuous_video(video_path)
    # Activate only once both complete, immutable assets are available.
    manifest['continuous'] = {'duration_s': data['duration_s'], 'identity_scope': 'continuous',
                              'analysis': analysis, 'video': media}
    manifest['updated_at'] = now()
    publisher.index(manifest)
    atomic_json(manifest_path, manifest)
    if args.catalog:
        # Use the actual public index so local paths cannot enter the fallback.
        response = publisher.session.get(publisher.index_url, timeout=60)
        response.raise_for_status()
        public = response.json()
        if public.get('continuous', {}).get('video', {}).get('sha256') != media['sha256']:
            raise RuntimeError('Public index has not refreshed; retry publication before updating the catalog.')
        atomic_json(args.catalog, public)
    print(f'Published {frames} frames / {data["duration_s"]:g}s. {publisher.release_url}', flush=True)


if __name__ == '__main__':
    main()
