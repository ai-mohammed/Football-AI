"""Finalize a bounded match window and publish one cumulative analysis."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sports.common.matches import combine_segments
from process_full_match import atomic_json, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--seconds', required=True, type=float)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--tag', default='match-barcelona-real-2025-26')
    args = parser.parse_args()
    if (args.directory/'worker.lock').exists() or (args.directory/'publisher.lock').exists():
        parser.error('Stop the workers and clear their verified stale locks before finalizing.')
    path = args.directory/'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    end = min(args.seconds, manifest['source']['duration_s'])
    segments = [s for s in manifest['segments'] if s['end_s'] <= end+1e-6]
    if not segments or abs(segments[-1]['end_s']-end) > 1e-6:
        parser.error('Choose an existing segment boundary for this overview.')
    if any(s['status'] != 'ready' for s in segments):
        parser.error('Every segment in the requested window must be ready.')
    analyses = [json.loads((args.directory/'segments'/s['id']/'analysis.json').read_text(encoding='utf-8'))
                for s in segments]
    overview = combine_segments(analyses)
    raw = json.dumps(overview, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    overview_path = args.directory/'overview.json.gz'
    overview_path.write_bytes(gzip.compress(raw, mtime=0))
    manifest.update(full_segment_count=manifest.get('full_segment_count', len(manifest['segments'])),
        segments=segments, status='complete', updated_at=now(),
        analysis_window={'start_s': 0., 'end_s': end, 'duration_s': end})
    manifest['processing']['max_source_seconds'] = end
    manifest['overview'] = {'local_path': 'overview.json.gz', 'duration_s': end}
    if args.publish:
        from publish_match import MatchPublisher
        publisher = MatchPublisher(args.tag, manifest['title'])
        saved = args.directory/'publication.json'
        published = json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else {}
        for segment in segments:
            if segment['id'] not in published:
                published[segment['id']] = publisher.segment(
                    args.directory/'segments'/segment['id']/'bundle.zip', segment['id'])
            segment.update(published[segment['id']])
        manifest['overview'].update(publisher.overview(overview_path))
        manifest.update(remote_manifest_url=publisher.index_url, release_url=publisher.release_url)
        publisher.index(manifest)
        atomic_json(saved, published)
    atomic_json(path, manifest)
    print(f'{len(segments)} segments; {end:g} seconds; {len(overview["events"])} events; '
          f'overview {overview_path.stat().st_size/1024/1024:.1f} MiB')


if __name__ == '__main__':
    main()
