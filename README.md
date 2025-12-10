# wav_diff

A toolkit for converting audio to 16-bit spectrogram images and generating synchronized captions, designed for training image diffusion models on audio data.

## Overview

This project provides tools to:
- Convert audio files to high-fidelity 16-bit PNG spectrograms (and back)
- Generate captions for spectrogram slices using AI models
- Compare original vs reconstructed spectrograms with quality metrics

## Installation

```bash
pip install librosa numpy soundfile opencv-python torch transformers tinytag muq
```

## Tools

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
- Per-slice PSNR scores
- Average PSNR across all slices

### narrative_captioner.py

Generates detailed natural language descriptions using Qwen2-Audio-7B.

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

Generates terse tag-style captions using MuQ-MuLan similarity matching.

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
- Prompts for optional artist/title prefix
- Matches audio against 300+ curated tags across Genre, Instrument, Mood, and Texture categories
- Includes BPM detection
- Output format: `"Rock, Heavy Metal, Distorted Guitar, Drums, Aggressive, Dark, 140 BPM"`

## Workflow Example

1. **Convert audio to spectrograms:**
   ```bash
   python audio_converter.py ./music -o ./spectrograms
   ```

2. **Generate captions (choose one):**
   ```bash
   # Detailed prose descriptions
   python narrative_captioner.py ./music -o ./captions

   # Terse tag-style captions
   python muq_captioner.py ./music -o ./captions
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

## Requirements

- Python 3.8+
- PyTorch with CUDA (recommended for captioners)
- ~16GB VRAM for Qwen2-Audio-7B
- ~8GB VRAM for MuQ-MuLan

## License

MIT
