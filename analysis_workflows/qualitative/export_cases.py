#!/usr/bin/env python3
"""Rebuild selected qualitative case evidence from existing scored runs and OCR caches.

Read-only inputs. No OCR, model inference, rescoring, or bootstrap. Outputs stay under
runs/analysis/ (ignored by Git); never upload raw case folders to public GitHub.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

from ulm_extractor.dataset import image_for
from ulm_extractor.storage import key_for

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = ROOT / 'runs/batches/cbdd58de-531a-4c9c-9778-5bc8b0b0be48/batch.json'
DEFAULT_SELECTION = Path(__file__).with_name('case_notes.csv')
DEFAULT_OUTPUT = ROOT / 'runs/analysis/layout_case_review_reproduced'
MODELS = ('qwen35_vlm', 'qwen35_ocr', 'qwen35_laocr', 'qwen35_ppstructure', 'qwen25_ocr')
OCR_FILES = {
    'paddleocr': 'plain_ocr.txt',
    'paddleocr_layout': 'layout_ocr.json',
    'ppstructurev3': 'ppstructure.md',
}
PP_EXTRAS = {
    'original_markdown_path': 'original_markdown.md',
    'fallback_ocr_path': 'fallback_ocr.txt',
    'audit_path': 'ppstructure_audit.json',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def as_existing_file(raw):
    path = Path(raw).expanduser()
    if not path.is_absolute() and not path.is_file():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=DEFAULT_BATCH)
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--include-images', action='store_true', help='Copy benchmark images into ignored local case folders')
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    private_root = (ROOT / 'runs/analysis').resolve()
    if output == private_root or not output.is_relative_to(private_root):
        parser.error('Evidence output must be a subdirectory of runs/analysis/ (ignored by Git)')
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output is not empty; refusing to overwrite: {output}')

    batch_path = as_existing_file(args.batch)
    batch_dir = batch_path.parent
    run_root = batch_dir.parents[1]
    batch = read_json(batch_path)
    docs = read_json(batch_dir / 'documents.json')
    if len(docs) != 400:
        raise ValueError(f'Expected 400 benchmark documents; found {len(docs)}')
    document_lookup = {}
    for doc in docs:
        identity = (str(doc['dataset_name']), str(doc['id']))
        if identity in document_lookup:
            raise ValueError(f'Duplicate document identity: {identity}')
        document_lookup[identity] = doc

    with as_existing_file(args.selection).open(encoding='utf-8', newline='') as f:
        selection = list(csv.DictReader(f))
    if len(selection) != 16:
        raise ValueError(f'Expected 16 selected cases; found {len(selection)}')
    identities = [(row['dataset'], row['id']) for row in selection]
    if len(set(identities)) != 16:
        raise ValueError('Duplicate selected case')
    if set(identities) - set(document_lookup):
        raise ValueError(f'Selected cases missing from the batch: {set(identities) - set(document_lookup)}')

    specs = {spec['id']: spec for spec in batch['runs']}
    if not set(MODELS).issubset(specs):
        raise ValueError('Missing model configuration in batch')

    predictions = {}
    for model in MODELS:
        entries = read_json(run_root / specs[model]['run_id'] / 'predictions_scored.json')
        if len(entries) != 400:
            raise ValueError(f'Expected 400 scored predictions for {model}')
        keyed = {key_for(item): item for item in entries}
        if len(keyed) != 400:
            raise ValueError(f'Duplicate predictions for {model}')
        predictions[model] = keyed

    cache = {}
    for engine in OCR_FILES:
        index = read_json(batch_dir / 'ocr' / engine / 'index.json')
        if len(index['records']) != 400:
            raise ValueError(f'Incomplete {engine} OCR cache')
        cache[engine] = index['records']

    copies = []
    payloads = []
    manifest = []
    for selected in selection:
        dataset, doc_id = selected['dataset'], selected['id']
        doc = document_lookup[dataset, doc_id]
        key = key_for(doc)
        dest = output / f'{dataset}_{doc_id}'

        for engine, target_name in OCR_FILES.items():
            copies.append((as_existing_file(cache[engine][key]['text_path']), dest / target_name))
        for field, target_name in PP_EXTRAS.items():
            copies.append((as_existing_file(cache['ppstructurev3'][key][field]), dest / target_name))
        if args.include_images:
            image = as_existing_file(image_for(doc, batch['image_directory']))
            copies.append((image, dest / image.name))

        payloads.append((dest / 'reference.json', {
            'category': selected['selection_group'],
            'question': doc['question'],
            'answers': doc['answers'],
        }))
        row = {'dataset': dataset, 'id': doc_id, 'selection_group': selected['selection_group'],
               'document_key': key, 'image_path': doc['image_path']}
        for model in MODELS:
            scored = predictions[model].get(key)
            if scored is None:
                raise ValueError(f'Missing scored prediction for {model} {dataset} {doc_id}')
            row[model] = scored['score']
            payloads.append((dest / (model + '.json'), {
                'score': scored['score'],
                'prediction': scored.get('predict'),
                'hit_token_limit': scored.get('hit_token_limit'),
            }))
        manifest.append(row)

    output.mkdir(parents=True, exist_ok=True)
    for original, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    for target, obj in payloads:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output / 'case_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (output / 'README.txt').write_text(
        'Private reconstructed evidence for 16 purposively selected OCRBench v2 KIE cases.\n'
        'Inputs: saved predictions_scored.json and three OCR cache indexes.\n'
        'No inference, OCR or scoring was rerun. Do not commit or publicly upload these files.\n'
        f'Original benchmark images copied: {args.include_images}\n', encoding='utf-8')
    print(f'Exported {len(manifest)} cases to {output}')
    print('No inference, OCR, scoring, or existing analyses were changed.')


if __name__ == '__main__':
    main()
