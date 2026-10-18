#!/usr/bin/env python3
"""Offline Chinese -> Cyrillic Khalkha Mongolian merchant draft CLI."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable

TRANSLATABLE_FIELDS = (
    'texts.zh.name', 'texts.zh.summary', 'texts.zh.location',
    'texts.zh.address', 'texts.zh.openingHours',
    'texts.zh.services.*.text', 'texts.zh.products.*.name',
    'texts.zh.products.*.specification', 'texts.zh.products.*.supplyNote',
)
MODEL_ID = 'facebook/nllb-200-3.3B'
DEFAULT_CACHE = '/Volumes/data/NLLB/hf-cache'
MAX_INPUT_TOKENS = 512
MAX_OUTPUT_LENGTH = 256


class DraftError(Exception):
    """An actionable input/model error safe to print to stderr."""


def blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def validate_texts(value, path):
    if not isinstance(value, dict):
        raise DraftError(f'{path}: expected an object')
    for key in ('name', 'summary', 'location', 'address', 'openingHours'):
        if key in value and value[key] is not None and not isinstance(value[key], str):
            raise DraftError(f'{path}.{key}: expected string/null')
    for key, fields in (('products', ('name', 'specification', 'supplyNote')),
                        ('services', ('text',))):
        items = value.get(key, [])
        if not isinstance(items, list):
            raise DraftError(f'{path}.{key}: expected an array')
        ids = set()
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id']:
                raise DraftError(f'{path}.{key}: each item needs a nonempty string id')
            if item['id'] in ids:
                raise DraftError(f'{path}.{key}: duplicate id {item["id"]!r}')
            ids.add(item['id'])
            for field in fields:
                if field in item and item[field] is not None and not isinstance(item[field], str):
                    raise DraftError(f'{path}.{key}[{item["id"]}].{field}: expected string/null')


def prepare(data):
    """Work on a copy; fail on reference conflicts rather than delete data."""
    if not isinstance(data, dict):
        raise DraftError('input must be a merchant JSON object')
    status = data.get('translationStatus')
    if not isinstance(status, dict) or status.get('mn') not in ('missing', 'draft', 'reviewed'):
        raise DraftError('translationStatus.mn must be missing, draft or reviewed')
    if status['mn'] == 'reviewed':
        return copy.deepcopy(data), []
    out = copy.deepcopy(data)
    texts = out.get('texts')
    if not isinstance(texts, dict):
        raise DraftError('texts must be an object')
    zh = texts.get('zh')
    validate_texts(zh, 'texts.zh')
    mn = texts.get('mn')
    if mn is None:
        mn = texts['mn'] = {}
    validate_texts(mn, 'texts.mn')
    # Copy structural references, never translate or silently replace conflicts.
    def reference(source, dest, key, path):
        if key not in source:
            return
        if key in dest and dest[key] != source[key]:
            raise DraftError(f'{path}.{key}: reference conflict; resolve manually')
        dest[key] = copy.deepcopy(source[key])
    reference(zh, mn, 'mapUrl', 'texts.mn')
    for group in ('products', 'services'):
        source = zh.get(group, [])
        current = {item['id']: item for item in mn.get(group, [])}
        extras = set(current) - {item['id'] for item in source}
        if extras:
            raise DraftError(f'texts.mn.{group}: unmatched IDs {sorted(extras)}; no data removed')
        ordered = []
        for item in source:
            dest = current.get(item['id'], {'id': item['id']})
            if group == 'products':
                reference(item, dest, 'imageId', f'texts.mn.{group}[{item["id"]}]')
            ordered.append(dest)
        mn[group] = ordered
    jobs = []
    for path in TRANSLATABLE_FIELDS:
        parts = path.split('.')[2:]
        if '*' in parts:
            group, _, field = parts
            for source, dest in zip(zh.get(group, []), mn[group]):
                text = source.get(field)
                if not blank(text) and blank(dest.get(field)):
                    jobs.append((dest, field, text, f'texts.mn.{group}[{source["id"]}].{field}'))
                elif field in source and field not in dest and blank(text):
                    dest[field] = text
        else:
            field = parts[0]
            text = zh.get(field)
            if not blank(text) and blank(mn.get(field)):
                jobs.append((mn, field, text, f'texts.mn.{field}'))
            elif field in zh and field not in mn and blank(text):
                mn[field] = text
    return out, jobs


def apply_jobs(out, jobs, translate: Callable[[str], str], log):
    failed = 0
    for dest, key, text, path in jobs:
        for attempt in range(2):
            try:
                value = translate(text)
                if not isinstance(value, str) or not value.strip():
                    raise DraftError('model returned empty text')
                dest[key] = value
                break
            except Exception as exc:
                if attempt == 1:
                    failed += 1
                    log(f'ERROR {path}: {exc} (failed after 2 attempts; original preserved)')
    if jobs:
        out['translationStatus']['mn'] = 'draft'
    return failed


def split_input(text, count):
    if count(text) <= MAX_INPUT_TOKENS:
        return [text]
    sentences = re.split(r'(?<=[。？！!?])', text)
    chunks, current = [], ''
    for sentence in sentences:
        if not sentence:
            continue
        if count(sentence) > MAX_INPUT_TOKENS:
            raise DraftError('input exceeds 512 tokens and a sentence cannot be split safely')
        if current and count(current + sentence) > MAX_INPUT_TOKENS:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)
    return chunks


class NllbTranslator:
    def __init__(self):
        # Respect custom HF_HOME for empty-cache failure tests, always stay offline.
        os.environ.setdefault('HF_HOME', DEFAULT_CACHE)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        if not torch.backends.mps.is_available():
            raise DraftError('MPS unavailable; CPU fallback requires user confirmation')
        cache = str(Path(os.environ['HF_HOME']) / 'hub')
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, src_lang='zho_Hans', cache_dir=cache, local_files_only=True)
        self.torch = torch
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID, cache_dir=cache, local_files_only=True, use_safetensors=False,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to('mps').eval()
        self.target = self.tokenizer.convert_tokens_to_ids('khk_Cyrl')
        if self.target == self.tokenizer.unk_token_id:
            raise DraftError('target language token khk_Cyrl missing')

    def __call__(self, text):
        count = lambda s: len(self.tokenizer(s, truncation=False)['input_ids'])
        chunks = split_input(text, count)
        results = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors='pt', truncation=False).to('mps')
            with self.torch.inference_mode():
                tokens = self.model.generate(
                    **inputs, forced_bos_token_id=self.target,
                    num_beams=4, max_length=MAX_OUTPUT_LENGTH, do_sample=False)
            ids = tokens[0].tolist()
            if len(ids) >= MAX_OUTPUT_LENGTH:
                raise DraftError('output reached the 256-token cap; possible truncation')
            value = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
            if not value.strip():
                raise DraftError('model returned empty text')
            results.append(value)
        return '\n'.join(results)


def atomic_write(path, data):
    path = Path(path)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as f:
            temp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.chmod(temp, path.stat().st_mode & 0o777)
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def main(argv=None, factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--in-place', action='store_true')
    args = parser.parse_args(argv)
    log = lambda message: print(message, file=sys.stderr, flush=True)
    try:
        data = json.loads(args.input.read_text(encoding='utf-8'))
        if (isinstance(data, dict) and isinstance(data.get('translationStatus'), dict)
                and data['translationStatus'].get('mn') == 'reviewed'):
            log('SKIP translationStatus.mn=reviewed: merchant unchanged; no output written')
            return 0
        out, jobs = prepare(data)
        if jobs:
            try:
                translator = (factory or NllbTranslator)()
            except Exception as exc:
                log(f'ERROR model load: {exc}; check offline cache HF_HOME; no output written')
                return 2
            failed = apply_jobs(out, jobs, translator, log)
        else:
            failed = 0
        target = args.input if args.in_place else Path(str(args.input) + '.mn-draft.json')
        atomic_write(target, out)
        log(f'WROTE {target}: {len(jobs)-failed} translated, {failed} failed; human review required')
        return 1 if failed else 0
    except (OSError, ValueError, TypeError, DraftError) as exc:
        log(f'ERROR input/output: {exc}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
