"""Publish only this match's processed bundles; never publish the source file or credentials."""
import copy
import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import requests

REPOSITORY = 'ai-mohammed/Football-AI'


class MatchPublisher:
    def __init__(self, tag, title):
        if not re.fullmatch(r'match-[a-z0-9-]+', tag):
            raise ValueError('A dedicated match- release tag is required.')
        result = subprocess.run(['git', 'credential', 'fill'],
            input='protocol=https\nhost=github.com\npath=ai-mohammed/Football-AI.git\n\n',
            text=True, capture_output=True, check=True)
        credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {credential["password"]}',
            'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
        self.api = f'https://api.github.com/repos/{REPOSITORY}'
        response = self.session.get(f'{self.api}/releases/tags/{tag}', timeout=30)
        if response.status_code == 404:
            response = self.session.post(f'{self.api}/releases', json={
                'tag_name': tag, 'target_commitish': 'main', 'name': title,
                'body': 'Analyse vidéo par Mohammed ADDI. Segments de 12 secondes maximum, publiés progressivement. '
                        'Horaires du fichier source, pas l’horloge officielle du match. '
                        'Les mesures incluent les ralentis et restent des estimations. '
                        'Méthode et modèle de maillots externe : https://github.com/ai-mohammed/Football-AI/blob/main/docs/JERSEY_NUMBERS.md',
                'draft': False, 'prerelease': True, 'make_latest': 'false'}, timeout=30)
        response.raise_for_status()
        self.release = response.json()
        self.release_url = self.release['html_url']
        self.upload_url = self.release['upload_url'].split('{')[0]
        self.base_url = f'https://github.com/{REPOSITORY}/releases/download/{tag}'
        self.index_url = self.base_url+'/progress.json'
        self.assets = {}
        page = 1
        while True:
            r = self.session.get(f'{self.api}/releases/{self.release["id"]}/assets',
                                 params={'per_page': 100, 'page': page}, timeout=30)
            r.raise_for_status(); values = r.json()
            self.assets.update({v['name']: v for v in values})
            if len(values) < 100:
                break
            page += 1

    def _upload(self, name, content, content_type):
        old = self.assets.get(name)
        if old:
            r = self.session.delete(f'{self.api}/releases/assets/{old["id"]}', timeout=30)
            r.raise_for_status()
            self.assets.pop(name, None)
        for attempt in range(3):
            r = self.session.post(self.upload_url, params={'name': name}, data=content,
                                  headers={'Content-Type': content_type}, timeout=180)
            if r.status_code < 500:
                break
            time.sleep(2**attempt)
        r.raise_for_status()
        asset = r.json(); self.assets[name] = asset
        return asset

    def segment(self, path, segment_id):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        name = f'{segment_id}-{digest[:12]}.zip'
        asset = self.assets.get(name) or self._upload(name, content, 'application/zip')
        return {'bundle_url': asset['browser_download_url'], 'bundle_sha256': digest,
                'bundle_bytes': len(content)}

    def index(self, manifest):
        public = copy.deepcopy(manifest)
        if public.get('overview'):
            public['overview'].pop('local_path', None)
        for segment in public['segments']:
            segment.pop('analysis_path', None); segment.pop('video_path', None)
            if segment['status'] == 'ready' and not segment.get('bundle_url'):
                segment['status'] = 'awaiting_publication'
        content = json.dumps(public, ensure_ascii=False, separators=(',', ':')).encode()
        self._upload('progress.json', content, 'application/json')

    def overview(self, path):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        name = f'overview-{digest[:12]}.json.gz'
        asset = self.assets.get(name) or self._upload(name, content, 'application/gzip')
        return {'url': asset['browser_download_url'], 'sha256': digest, 'bytes': len(content)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--tag', default='match-barcelona-real-2025-26')
    parser.add_argument('--watch', action='store_true', help='Publish new finished segments until the worker stops')
    args = parser.parse_args()
    import os
    lock = args.directory/'publisher.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    try:
        manifest_path = args.directory/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        publisher = MatchPublisher(args.tag, manifest['title'])
        saved = args.directory/'publication.json'
        published = json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else {}
        previous = None
        while True:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            changed = False
            for segment in manifest['segments']:
                if segment['status'] == 'ready' and segment['id'] not in published:
                    bundle = args.directory/'segments'/segment['id']/'bundle.zip'
                    try:
                        published[segment['id']] = publisher.segment(bundle, segment['id'])
                        temporary = saved.with_suffix('.tmp')
                        temporary.write_text(json.dumps(published, indent=2), encoding='utf-8')
                        temporary.replace(saved)
                        changed = True
                        print(f'{segment["id"]}: published; {len(published)}/{len(manifest["segments"])}', flush=True)
                    except requests.RequestException as error:
                        print(f'Publication will retry: {type(error).__name__}', flush=True)
                        break
            for segment in manifest['segments']:
                if segment['id'] in published:
                    segment.update(published[segment['id']])
            manifest.update(remote_manifest_url=publisher.index_url, release_url=publisher.release_url)
            version = (manifest.get('updated_at'), tuple(sorted(published)))
            if changed or version != previous:
                try:
                    publisher.index(manifest)
                    previous = version
                except requests.RequestException as error:
                    print(f'Progress publication will retry: {type(error).__name__}', flush=True)
            running = (args.directory/'worker.lock').exists()
            missing = any(s['status'] == 'ready' and s['id'] not in published for s in manifest['segments'])
            if not args.watch:
                if missing or previous != version:
                    raise RuntimeError('Publication incomplete; rerun to resume.')
                break
            if not running and not missing and previous == version:
                break
            time.sleep(45)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
