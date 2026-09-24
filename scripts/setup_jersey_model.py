"""Download and convert the published SoccerNet ViT-S checkpoint, once locally.

External weights: Lukasz Grad, CVPRW 2025. Upstream LICENSE: CC-BY-NC-SA-4.0.
https://github.com/lukaszgrad/uncertainty-jnr
No training or optimizer state is retained in the inference-only safetensors file.
"""
import argparse
import hashlib
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Previously downloaded official ViT-S checkpoint')
    parser.add_argument('--output', type=Path, default=Path('examples/soccer/data/jersey-small16.safetensors'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose another path to preserve the model.')
    import gdown
    import torch
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory(prefix='football-jersey-') as folder:
        checkpoint = args.checkpoint or Path(folder)/'small16.pth'
        if args.checkpoint is None:
            gdown.download(id='1oc8VdEHHxXQfhNZTvbHbm6gfLRkwGl2o', output=str(checkpoint), quiet=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        state = {k.replace('._orig_mod.', '.'): v for k, v in state.items()}
        required = ('classifier.position_embeddings', 'classifier.position_biases', 'classifier.digit_classifier.weight')
        if state.get('backbone.pos_embed', torch.empty(0)).shape != (1, 197, 384) or not all(k in state for k in required):
            raise ValueError('This is not the supported ViT-S jersey checkpoint.')
        compact = {k: v.contiguous() for k, v in state.items()
                   if (k.startswith('backbone.') and not k.startswith('backbone.head.')) or k in required}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        checksum = hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                checksum.update(block)
        save_file(compact, str(args.output), metadata={
            'source': 'https://github.com/lukaszgrad/uncertainty-jnr', 'author': 'Lukasz Grad',
            'license': 'CC-BY-NC-SA-4.0', 'license_readme_discrepancy': 'README says CC-BY-SA-4.0; LICENSE contains NC',
            'checkpoint_sha256': checksum.hexdigest(), 'variant': 'small16_reid'})
    print(f'Saved {args.output} ({args.output.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
