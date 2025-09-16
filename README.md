# Automatic Transcription Agreement

Command line utilities for running several automatic speech recognition (ASR) models on the same dataset and measuring how closely their transcripts agree. The toolchain can both generate transcripts (weakly supervised setup) and enrich existing transcripts with agreement metrics.

## Features
- Batch inference across multiple ASR backends (OpenAI Whisper, Microsoft Phi-4, SpeechBrain Conformer) with optional GPU placement control.
- Silence detection to avoid wasting inference time on empty utterances.
- Agreement metrics based on word error rate (WER) and simple transcript merging heuristics.
- Progress-aware CSV pipelines that produce enriched datasets ready for downstream review.

## Requirements
- Python 3.9+
- System packages: `ffmpeg` (required by [pydub](https://github.com/jiaaro/pydub)), libsndfile (required by [SoundFile](https://pysoundfile.readthedocs.io/)).
- Python packages: `torch`, `torchvision`, `transformers`, `pandas`, `numpy`, `pydub`, `soundfile`, `speechbrain`, `tqdm`, `pyyaml`.

Install the Python dependencies with your preferred workflow, for example:

```bash
pip install -r requirements.txt
```


## Data Preparation
Create a CSV file that lists the utterances you want to process. At minimum it must include a `wav` column with absolute or repository-relative paths to audio files:

```csv
id,wav,speaker
utt_0001,/data/callcentre/audio/utt_0001.wav,alice
utt_0002,/data/callcentre/audio/utt_0002.wav,bob
```

If you already have transcripts, add one column per ASR system (column names should match the identifiers in the configuration file described below). These columns can then be enriched directly without re-running ASR.

## Configuration File
Both entry points consume the same YAML configuration schema:

```yaml
asr_models:
  - whisper
  - phi4
  - conformer
threshold: 0.25  # maximum acceptable average WER for best_ai_wrd
```

- `asr_models`: ordered list of model identifiers to run/compare. Supported values are `whisper`, `phi4`, and `conformer`. The order determines the column names written to the CSV.
- `threshold`: agreement threshold (0–1 floats). When the best transcript has an average WER above this value, the `best_ai_wrd` column is set to `<WER above threshold>`.

## Usage
Run the scripts from the repository root so that relative paths in the CSV resolve correctly.

### 1. Generate transcripts (`weak_supervised_transcription.py`)

```bash
python weak_supervised_transcription.py data/utterances.csv configs/asr.yaml \
  --output data/transcribed.csv \
  --sequential           # optional: load models one-by-one instead of all at once
  --same-gpu             # optional: force every model onto CUDA:0
  --silence-threshold -45  # optional: tweak silence detection in dBFS
  --skip-agreement       # optional: keep only raw transcripts
  --no-progress          # optional: hide progress bars
```

What happens:
- Detects silence in each `wav` file; silent utterances are tagged with `<Sil>` instead of being transcribed.
- Loads each requested ASR backend on CPU/GPU (parallel by default, sequential when `--sequential` is used).
- Writes one column per model and, unless `--skip-agreement` is set, appends the following agreement columns:
  - `agreement_wrd`: heuristic merge across all ASR transcripts.
  - `best_ai_wrd`: transcript whose average pairwise WER falls below `threshold`; otherwise `<WER above threshold>`.

The output CSV path defaults to `transcribed_<input name>.csv` when `--output` is omitted.

### 2. Enrich existing transcripts (`agreement.py`)

If you used `--skip-agreement` or if you need to re-computes those same columns on an existing CSV, so you can reuse prior transcripts or tweak the threshold without re-running the ASR. If you only need the fresh transcripts plus agreement metrics, step 1 is enough; step 2 is optional tooling for post-processing.

```bash
python agreement.py data/transcribed.csv configs/asr.yaml --output data/enriched.csv
```

- Reads the configured ASR columns from the input CSV.
- Computes the same agreement fields described earlier.
- Overwrites/creates `agreement_wrd` and `best_ai_wrd` columns in the output file.

Use this script when transcripts have already been generated offline, or if you want to adjust the agreement threshold without re-running ASR inference.
