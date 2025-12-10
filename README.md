# wav_diff

A toolkit for converting audio to 16-bit spectrogram images and generating synchronized captions, designed for training image diffusion models on audio data.

## Overview

This project provides tools to:
- Convert audio files to high-fidelity 16-bit PNG spectrograms (and back)
- Generate captions for spectrogram slices using AI models
- Compare original vs reconstructed spectrograms with quality metrics

## Quick Start

For the simplest workflow, use the interactive `main.py` script which combines spectrogram generation and captioning in one step:

```bash
python main.py
```

This will prompt you for:
- Input audio file or folder
- Output folder
- Artist name (optional)
- Song title (optional)

The script automatically generates both spectrogram images and MuQ-based captions.

**Sample data included:** `sample_data/Gex.mp3` - try it out!

## Installation

```bash
pip install librosa numpy soundfile opencv-python torch transformers tinytag muq
```

## Tools

### main.py

Interactive wrapper that combines `audio_converter.py` and `muq_captioner.py` into a single automated pipeline. Ideal for quickly generating training datasets.

```bash
python main.py
```

**Workflow:**
1. Prompts for input/output paths and metadata
2. Generates 1024x1024 spectrogram images (Phase 1)
3. Generates MuQ-based captions with optional artist/title prefix (Phase 2)

### audio_converter.py

Converts audio to spectrogram images and reconstructs audio from images.

**Encode audio to images:**
```bash
python audio_converter.py input.wav -o ./output_images --size 1024
```

**Decode images back to audio:**
```bash
python audio_converter.py ./output_images -o reconstructed.wav --decode
```

| Argument | Description |
|----------|-------------|
| `input` | Input audio file or directory |
| `-o, --output` | Output directory/file (default: `./output`) |
| `--decode` | Decode mode: convert images to audio |
| `--size` | Image resolution (default: 1024) |

**Technical details:**
- Uses STFT with n_fft=2048, hop_length=512
- Encodes magnitude (dB) in red channel, phase (cos/sin) in green/blue
- 16-bit PNG preserves full dynamic range
- Lanczos4 interpolation for resizing
- Slices audio into square chunks (1025x1025 STFT bins) before resizing

### compare_spectrograms.py

Compares original and reconstructed spectrograms, calculating PSNR and generating visual comparisons.

```bash
python compare_spectrograms.py ./original_spectrograms ./reconstructed_spectrograms -o ./comparison_results
```

| Argument | Description |
|----------|-------------|
| `original_dir` | Directory containing original spectrograms |
| `reconstructed_dir` | Directory containing reconstructed spectrograms |
| `-o, --output` | Output directory (default: `./comparison_results`) |

**Output:**
- Side-by-side comparison images (Original | Reconstructed | Diff x50)
- Per-slice PSNR scores (16-bit scale)
- Average PSNR across all slices

### narrative_captioner.py

Generates detailed natural language descriptions using Qwen2-Audio-7B-Instruct.

```bash
python narrative_captioner.py ./audio_files -o ./training_captions
```

| Argument | Description |
|----------|-------------|
| `input` | Input audio file or directory |
| `-o, --output` | Output directory (default: `./training_captions`) |
| `--n_fft` | FFT size (default: 2048) |
| `--hop` | Hop length (default: 512) |

**Output:** `.txt` files with prose descriptions of each audio slice, synchronized with spectrogram chunks.

### muq_captioner.py

Generates terse tag-style captions using MuQ-MuLan similarity matching against AudioSet and MagnaTagATune vocabularies.

```bash
python muq_captioner.py ./audio_files -o ./training_captions
```

| Argument | Description |
|----------|-------------|
| `input` | Input audio file or directory |
| `-o, --output` | Output directory (default: `./training_captions`) |
| `--n_fft` | FFT size (default: 2048) |
| `--hop` | Hop length (default: 512) |

**Features:**
- Prompts for optional artist/title prefix (e.g., "style of Artist Name")
- Matches audio against AudioSet ontology (instruments, genres, moods)
- MagnaTagATune texture descriptors
- BPM detection via librosa
- Output format: `"A Dark Rock track featuring Electric Guitar. Characterized as heavy. 120 BPM."`

## Workflow Example

### Automated (Recommended)

```bash
python main.py
# Follow prompts to generate spectrograms + captions in one step
```

### Manual

1. **Convert audio to spectrograms:**
   ```bash
   python audio_converter.py ./sample_data -o ./spectrograms
   ```

2. **Generate captions (choose one):**
   ```bash
   # Detailed prose descriptions (~16GB VRAM)
   python narrative_captioner.py ./sample_data -o ./captions

   # Terse tag-style captions (~8GB VRAM)
   python muq_captioner.py ./sample_data -o ./captions
   ```

3. **Train your diffusion model** on the image/caption pairs

4. **Reconstruct audio from generated images:**
   ```bash
   python audio_converter.py ./generated_spectrograms -o output.wav --decode
   ```

5. **Compare quality:**
   ```bash
   python compare_spectrograms.py ./spectrograms ./generated_spectrograms
   ```

## Project Structure

```
wav_diff/
├── main.py                 # Interactive pipeline (images + captions)
├── audio_converter.py      # Audio <-> spectrogram conversion
├── compare_spectrograms.py # PSNR comparison tool
├── narrative_captioner.py  # Qwen2-Audio prose captions
├── muq_captioner.py        # MuQ-MuLan tag-style captions
├── sample_data/            # Sample audio files for testing
│   └── Gex.mp3
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch with CUDA (recommended for captioners)
- ~16GB VRAM for Qwen2-Audio-7B (narrative_captioner.py)
- ~8GB VRAM for MuQ-MuLan (muq_captioner.py, main.py)

## License

MIT
