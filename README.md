# merchant-ai-draft

An offline CLI that fills missing **Chinese → Cyrillic Khalkha Mongolian** merchant copy using `facebook/nllb-200-3.3B`. Built for a noncommercial Hacktoberfest Week 2 prototype. No upload UI, page builder, QR generation, image model, or automatic publication.

**Every output is an unreviewed machine draft.** Local tests observed wrong proper names and product terms, omitted clauses, an incorrect discount (Chinese “八折” became “8 o’clock”), and a shipment promise changing from dispatch to arrival. Larger model size did not eliminate these problems. Numbers, currency, units, minimum orders, delivery terms and addresses require bilingual human review.

## Requirements and local setup

- Native Apple Silicon Python 3.10+ (tested: Python 3.12.14).
- macOS with working PyTorch MPS. Model weights are loaded in FP16; no automatic CPU fallback.
- Previously downloaded official `facebook/nllb-200-3.3B` checkpoint. The CLI **never downloads** a model.
- The local tested environment already exists at `/Volumes/data/NLLB/.venv`.

```bash
source /Volumes/data/NLLB/activate.sh
cd "/Volumes/data/Chatgpt app/merchant-ai-draft"
export HF_HOME=/Volumes/data/NLLB/hf-cache
export HF_HUB_OFFLINE=1
python mn_draft.py --input examples/before.json
```

On another machine, install `requirements.txt` into an independent environment and prepare the official checkpoint in `$HF_HOME/hub` first. `HF_HOME` defaults to `/Volumes/data/NLLB/hf-cache`; override it to use another existing cache. The CLI passes this cache explicitly, enables offline mode, and uses `local_files_only=True`. A disconnected external drive or incomplete cache is an error, not a reason to download to the internal disk. Weights are approximately 17.58 GB on disk; runtime and loading need additional memory. Do not run other large models concurrently on a 16 GB Mac.

Generation uses `zho_Hans` → `khk_Cyrl`, `mps`, `torch.float16`, `num_beams=4`, `max_length=256`, and no sampling. The existing environment also contains Accelerate for low-memory checkpoint loading; no web server or new model dependency is introduced.

## Input and output

Input is one merchant JSON object matching the existing business content schema. `examples/before.json` is a complete fictional schema-compatible merchant with `texts.mn: null`, one product, one service, and nonpublic placeholder contact/image data. All merchant details are invented; map links use `example.com`. Image references are placeholders, not bundled assets or real publication permissions.

```bash
# Leaves the source file unchanged; writes examples/before.json.mn-draft.json.
python mn_draft.py --input examples/before.json

# Explicit opt-in: atomically replace the input file.
python mn_draft.py --input merchant.json --in-place
```

Default output is the full input path plus `.mn-draft.json`. No file is written for a reviewed merchant: the CLI logs a skip and exits 0 without loading the model. Writes use a temporary sibling file and `os.replace`, including default output writes.

`TRANSLATABLE_FIELDS` at the top of `mn_draft.py` is the explicit allowlist:

- `texts.zh.name`, `summary`, `location`, `address`, `openingHours`
- `texts.zh.services.*.text`
- `texts.zh.products.*.name`, `specification`, `supplyNote`

Only missing, null, empty or whitespace-only target values are filled. Existing nonempty Mongolian text is never replaced. Unknown fields are not translated; existing unknown values are preserved. `mapUrl`, product `imageId`, IDs and all merchant-level fields remain unchanged. Image alt text lies outside this CLI's scope. Source-only structural references are copied verbatim to newly created Mongolian objects. Conflicting references, duplicate IDs and target-only IDs cause a clear input error rather than deletion. Product/service entries match by ID and are ordered like the Chinese entries.

If `translationStatus.mn` is `reviewed`, the entire merchant is untouched. For new translation attempts the status is `draft`, including partial failures; the CLI never marks anything reviewed. When no translation is needed the original status is retained.

Long input is measured with the actual tokenizer, including special tokens, against a 512-token limit. Oversize text is split at Chinese/ASCII sentence-ending punctuation and packed into chunks; an oversize indivisible sentence fails explicitly. Chunks are joined with newlines. Output reaching the 256-token cap is considered a failure even if an EOS token was forced at the limit; text is never silently truncated and accepted.

A field gets one retry. After failure its original value is retained (missing stays missing), an error containing the field path is logged, and later fields continue. Model initialization failure occurs before any field processing and produces no output. Partial output can be valid JSON but fail the upstream content schema; it is not publication-ready.

Exit codes:

- `0`: completed without field failures, or skipped a reviewed merchant.
- `1`: partial translation failures; inspect stderr and the draft file.
- `2`: input, reference conflict, model initialization, or output error.

## Tests

```bash
python -m unittest discover -s tests -v
```

Standard-library tests inject a translation function (no model download) to verify reviewed protection, blank-only filling, ID alignment, structural reference preservation, retries/partial failures, long-text handling and atomic-write failure recovery. The actual offline empty-cache check is:

```bash
empty_cache=$(mktemp -d /Volumes/data/NLLB/tmp/empty-cache.XXXXXX)
HF_HOME="$empty_cache" HF_HUB_OFFLINE=1 \
  python mn_draft.py --input examples/before.json
# Expected: clear model-load error, nonzero exit, no new output.
```

The real-model demo must be run separately; `examples/after.json` should be copied from the actual CLI output, never handwritten. Reviewed fixtures and model-load failures need no paid service or internet access.

## Existing build integration

After filling missing fields, bilingual human review, and successful validation with the original project's business schema and content checks, pass the resulting merchant JSON to that project's existing `src/content/businesses/` workflow. This CLI does not change the upstream project, add publication permission, or certify that a merchant may be published. Schema-compatible JSON alone does not establish valid image assets, category membership, consent, or translation quality.

## Licenses

The repository code is MIT licensed (see `LICENSE`). The separate **`facebook/nllb-200-3.3B` model weights are CC-BY-NC 4.0**, not MIT. This project is a noncommercial evaluation prototype; the repository license does not grant commercial model rights. Follow Meta's attribution and model-license conditions when using or redistributing weights; weights are not included here.

- [Official model card and usage limitations](https://huggingface.co/facebook/nllb-200-3.3B)
- [NLLB language-code and generation documentation](https://huggingface.co/docs/transformers/model_doc/nllb)

DEV post writing and publication are outside this CLI. Demo examples and a terminal recording are the intended supporting artifacts.

## Included demo artifacts

- `examples/before.json`: user-confirmed fictional input, validated against the upstream business schema.
- `examples/after.json`: unedited real 3.3B output from the CLI (9 translated fields, zero runtime failures), also schema-validated. **Not linguistically approved**: the takeaway service was mistranslated as a mining-related service; the noodle name/summary also need correction.
- `docs/demo.cast`: real terminal recording in asciinema v2 format; replay using an existing asciinema-compatible viewer. It shows the command, exit status and output JSON. No viewer is required to run the CLI.
- `scripts/record_demo.py`: standard-library-only recorder to regenerate the demo and copy actual CLI output to `after.json`.

The original business project was accessed read-only for schema validation. Its image/content build was not run or modified. Placeholder references in this repository are for structural compatibility tests, not deployable image assets.

### Second example: fictional scarf wholesaler

`examples/before2.json` / `examples/after2.json` show merchant `demo-scarf-wholesale`: three fictional cashmere scarf/shawl products with different dimensions, colors, minimum order quantities and two RMB price tiers per product, plus one mixed-color packing service. All business details and prices are invented; contacts remain nonpublic, image paths are placeholders, and publication permission is false.

```bash
python mn_draft.py --input examples/before2.json
# Default output: examples/before2.json.mn-draft.json
# examples/after2.json is an exact copy of that actual CLI output.
```

The real offline 3.3B run filled 15 text fields with no runtime failures. Both files passed the original business schema validation; source text, merchant metadata, product/service IDs and order, image references and map URL were preserved. The model output is unedited and still marked `draft`.

**This is a diagnostic example, not a successful translation-quality demonstration.** In the generated wholesale terms, RMB was mistranslated as Mongolian tugrik and freight conditions were omitted. Product names, material and color descriptions also contain substantial errors. Structural/schema checks cannot detect these semantic mistakes. The merchant input is approved as fictional demo data; the generated Mongolian text has not been linguistically approved.
