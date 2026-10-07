**wav_diff product improvement strategy and technical review**

Draft based on the review of commit `6aef73a9d21eab42ee004985be022a46948758ba` on October 7, 2026. This document records recommendations; it does not indicate that the changes have been implemented.

The subsequent [continuous audio and latent generation plan](ARCHITECTURE_PLAN.md) develops the selected direction into implementation milestones. It makes audio-native latents the primary model path while retaining a corrected complex STFT as a diagnostic reference.

The project is a useful research prototype, but audio reconstruction and dataset correctness need to be established before investing in substantial diffusion training. The highest priorities are preserving the numerical audio representation, synchronizing images and captions, satisfying caption model input requirements, and detecting incomplete datasets.

**Product intent and current implementation**

The intended workflow is to convert music into captioned spectrogram images, train an image diffusion model externally, and reconstruct audio from generated images. The repository currently provides:

- `audio_converter.py`: audio loading, trimming, STFT conversion, 16-bit image encoding, and reconstruction.
- `muq_captioner.py`: similarity-based music tags and tempo estimates.
- `narrative_captioner.py`: Qwen2-Audio natural-language descriptions.
- `main.py`: image generation followed by MuQ captioning.
- `compare_spectrograms.py`: image comparisons and PSNR.

Training and generation are not implemented or validated here. A successful round trip through the converter would establish a preprocessing baseline, but would not prove that generated spectrograms produce useful audio.

The recommended near-term product is a reproducible audio dataset builder with verifiable reconstruction and caption alignment. Keep image diffusion as an explicit experiment, and compare representations before committing to a larger training effort.

**Technical findings and corrective actions**

1. **High priority: spectrogram resizing introduces substantial distortion.**

   The encoder converts 1025×1025 spectrogram chunks into 1024×1024 images using Lanczos interpolation. The decoder interpolates them back. This mixes frequency bins, time frames, magnitudes, and phase vectors. Sixteen-bit PNG storage cannot restore information lost during interpolation. See [audio_converter.py](audio_converter.py), particularly encoding around line 67 and decoding around line 95.

   Preserve the native numerical grid. Use padding and valid-region masks when the model requires particular dimensions. Keep human-readable previews separate from numerical training artifacts. Avoid solving the dimension mismatch by silently dropping frequency bins.

2. **High priority: forward and inverse transforms disagree about centering.**

   Encoding uses `center=False`, while decoding leaves the inverse transform at its centered default. A one-slice test converted 526,336 samples into 524,288 samples, with reconstructed content beginning 1,024 samples into the reference. See the forward transform around line 43 and inverse transform around line 107 in [audio_converter.py](audio_converter.py).

   Define and persist one transform contract: FFT size, hop, window, centering, padding, sample rate, and original sample count. Handle window coverage at signal boundaries explicitly. Changing only the inverse centering flag is insufficient to guarantee reconstruction of the first and last samples.

3. **High priority: image and caption segment boundaries differ.**

   Captioning counts chunks using `len(audio) // 524800`. Image generation counts complete groups of 1,025 STFT frames, which require additional window support. A 524,800-sample input produces zero images but qualifies for one caption. A full image covers 1,536 samples beyond its nominal caption interval. The same caption segmentation formula appears in both captioners. See [audio_converter.py](audio_converter.py) around line 57, [muq_captioner.py](muq_captioner.py) around line 127, and [narrative_captioner.py](narrative_captioner.py) around line 90.

   Incomplete image chunks are discarded. Short clips can disappear entirely, and longer tracks lose their final partial chunk.

   Compute segment records once and share them across all stages. Distinguish the content interval from additional transform context. Pad partial segments, record valid sample counts, and caption only the intended content interval. Preserve original timeline offsets if trimming is enabled.

4. **High priority: caption model sample rates are incorrect.**

   MuQ receives 44.1 kHz samples directly, although the selected model expects 24 kHz. The model therefore receives samples with the wrong time and frequency interpretation. Qwen receives an explicit 44.1 kHz sample rate, although its feature extractor expects 16 kHz; the Transformers 4.45 feature extractor rejects this mismatch rather than resampling automatically. See [muq_captioner.py](muq_captioner.py) around line 83 and [narrative_captioner.py](narrative_captioner.py) around line 41.

   Segment the canonical audio first, then explicitly resample inside each model adapter. Assert the adapter input rate and test duration preservation. The model contracts are documented in the [MuQ configuration](https://huggingface.co/OpenMuQ/MuQ-MuLan-large/blob/main/config.json), [Qwen preprocessing configuration](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/blob/main/preprocessor_config.json), and [Transformers feature-extractor validation](https://raw.githubusercontent.com/huggingface/transformers/v4.45.0/src/transformers/models/whisper/feature_extraction_whisper.py).

5. **High priority: caption categories are contaminated and predictions lack abstention.**

   The AudioSet loader searches descriptions for words such as `instrument`, rather than following category relationships. The word `instrumentation` in the Punk rock description causes that genre to enter the instrument vocabulary. The bundled [first caption](step1_originals/Gex_slice_0000.txt) already says “featuring Punk rock.” See [muq_captioner.py](muq_captioner.py) around line 32 and the [AudioSet ontology](https://raw.githubusercontent.com/audioset/ontology/master/ontology.json).

   The captioner always chooses a winning label, supports only one instrument, and silently substitutes a different vocabulary after a network failure. The resulting labels are neither calibrated nor reproducible across vocabulary changes. Tempo is also presented as a definite integer without uncertainty.

   Use a versioned vocabulary with validated category membership, following actual ontology relationships where applicable. Allow multiple instruments, missing attributes, and abstention. Preserve similarity scores without presenting them as probabilities. Validate thresholds and tempo behavior against a human-reviewed set. Record vocabulary version, model revision, prompt, and any fallback used. Keep artist and title as metadata, with an explicit option to include them in training captions.

6. **High priority: incomplete jobs can be reported as successful.**

   Encoding returns empty lists for some failures, captioning catches per-file exceptions, and the wrapper does not validate paired outputs before announcing completion. Filename stems also collide for inputs such as `song.wav` and `song.mp3`. Rerunning into the same directory can leave stale slices. See [main.py](main.py) around line 46 and output naming in both processing stages.

   Introduce structured stage results, stable source and segment identifiers, atomic writes, explicit resume rules, and a final integrity check. Distinguish completed, skipped, and failed records. A failed or incomplete run must not report a fully ready dataset and must return an appropriate nonzero exit code.

7. **Medium priority: preprocessing and magnitude encoding contain undocumented losses.**

   Loading defaults to mono. Silence trimming removes content without retaining its offset. Magnitude encoding clips dynamic range, and a zero STFT coefficient decodes to approximately 0.01 magnitude rather than zero. This is a spectral coefficient value, not a measured waveform noise level. WAV output defaults to PCM16. See [audio_converter.py](audio_converter.py) around lines 33, 113, and 117.

   Make channel conversion, trimming, gain handling, magnitude scaling, and output subtype explicit policies. Encode zero correctly and measure any quantization error. Use floating-point reconstruction output during evaluation so final PCM quantization does not obscure representation errors.

8. **Medium priority: evaluation and decoding permit invalid input relationships.**

   The comparator matches sorted files by position, ignores unmatched extras, and resizes mismatched images. The bundled outputs contain 11 original images and 10 reconstructed images, but the comparator can still report an average. Image PSNR also does not establish waveform or perceptual quality. See [compare_spectrograms.py](compare_spectrograms.py) around line 28.

   Decoding concatenates all PNGs in a directory, skips unreadable files, and accepts 8-bit input while interpreting it using 16-bit scaling. A mixed-track directory can therefore become one waveform, and missing slices can silently remove time. See [audio_converter.py](audio_converter.py) around line 82.

   Match artifacts by manifest identity. Validate track grouping, sequence completeness, dimensions, dtype, channel order, and representation version. Reject incompatible input rather than silently resizing or skipping it. Report missing coverage separately from quality metrics.

**Evidence from the review**

The controlled tests measured waveform SNR after aligning the reference by 1,024 samples to isolate representation loss from the separate centering defect. Higher SNR is better.

| Signal | Native 1025×1025 without resizing | Default 1024×1024 round trip |
|---|---:|---:|
| Two-tone signal | 59.6 dB | 58.5 dB |
| Bundled music excerpt | 50.0 dB | 21.0 dB |
| Seeded white noise | 80.7 dB | 13.3 dB |

Each measurement used one 1,025-frame slice. These results demonstrate numerical distortion; they are not a whole-corpus benchmark or listening evaluation. The much smaller loss for the two-tone signal shows why simple tonal tests alone are insufficient.

Additional checks reproduced the one-slice length loss, the zero-image boundary case, nonzero decoded magnitude for zero coefficients, and acceptance of an 8-bit PNG. All five Python modules parsed successfully. The caption models were not downloaded or executed; their sample-rate findings come from source inspection and official model contracts.

The isolated test environment used Python 3.12, NumPy 2.5.3, librosa 1.0.0, OpenCV 5.0.0, and SoundFile 0.14.0. The repository does not lock dependencies, so these measurements should not be treated as a reproduction of the environment that generated the bundled artifacts.

**Recommended implementation**

Use one versioned dataset contract shared by segmentation, representation encoding, captioning, decoding, and evaluation:

```text
Source audio
    → explicit preprocessing
    → canonical segment records
        → representation encoder
        → caption model adapter
    → validated dataset manifest
    → training export
```

Each segment record should contain:

- Stable source and segment identifiers, source hash, and track identity.
- Original and canonical sample rates, channel policy, and preprocessing configuration.
- Original timeline offset, content boundaries, transform context, padding, and valid sample count.
- Representation name, version, numerical layout, and decoder configuration.
- Artifact locations and hashes.
- Caption text, structured attributes, scores, model revision, vocabulary version, and generation settings.
- Per-stage status and actionable failure information.

A small Python package is sufficient. Separate segmentation, representations, caption adapters, manifests, evaluation, and CLI commands. Load model dependencies only when needed. Provide independent build, caption, validate, decode, and evaluate operations so users can repair or rerun one stage without repeating the entire job.

Cache text embeddings for fixed vocabularies, batch model inference where practical, and process long recordings with bounded memory while preserving transform context. Model initialization and network downloads should not be required for basic help, validation, or audio-only operations.

**Representation decision**

Maintain a numerical complex-STFT reference implementation first. It should preserve sample timing, channel information according to policy, and all valid samples. This gives subsequent lossy representations a trustworthy baseline.

For the image-diffusion experiment, preserve the native grid using padding and masks. Implement a training loader that retains intended numerical precision and channel semantics. Disable ordinary image transformations such as flips, color adjustment, and arbitrary crops. Measure the intended image model's encoder/decoder round trip before training its denoiser: PNG bit depth alone does not establish what precision the training pipeline preserves.

Generated phase channels require separate evaluation. Normalizing phase vectors does not guarantee consistency between overlapping STFT frames. Measure generated outputs and assess whether a consistency projection or a different representation improves audible quality; do not assume source round-trip fidelity transfers to generation.

If the main objective is useful generated audio, also benchmark an audio-native continuous latent representation from a pretrained audio autoencoder. Compare it with the corrected spectrogram approach on reconstruction, conditioning, listening quality, storage, throughput, and training cost. Audio latent diffusion has established architectural precedent in [Fast Timing-Conditioned Latent Audio Diffusion](https://arxiv.org/abs/2402.04825), but the best representation for this project remains an empirical decision.

**Delivery priorities and acceptance criteria**

| Priority | Deliverable | Acceptance criterion |
|---|---|---|
| 1 | Shared segmentation, matched transforms, no spectrogram resizing, correct model sample rates, strict input validation | Exact declared length and timing; no unexplained sample loss; every accepted segment has valid paired artifacts |
| 2 | Versioned caption vocabulary, abstention, structured metadata, and a human-reviewed evaluation set | Measured label precision and coverage; no category contamination; labels refer to the correct interval |
| 3 | Corrected spectrogram versus audio latent benchmark | Representation selected using held-out reconstruction and generated-audio results, listening evaluation, and measured resource costs |
| 4 | Resumable CLI, manifests, embedding cache, bounded-memory processing, packaged dependencies, and CI | Interrupted jobs resume safely; failures remain visible; repeated runs preserve dataset identity |

The first implementation milestone should be a trustworthy encode → decode → evaluate loop and guaranteed caption alignment. Fixing these foundations should precede large training runs or expansion of the user interface.

**Validation and product quality**

Split training and evaluation by source track before generating segments, and detect duplicate recordings. Otherwise, adjacent or duplicated material can appear in both sets. Add artist-disjoint evaluation when the product claims generalization to unfamiliar artists.

Build regression fixtures for silence, tones, noise, impulses at boundaries, stereo signals, short clips, exact chunk boundaries, partial tails, corrupt images, duplicate filenames, and interrupted runs. Test adapters against model input contracts independently of expensive model inference, then add a small optional integration suite for supported model versions.

Measure sample count, alignment, coverage, waveform error, spectral error, clipping, and silence behavior. Use listening tests for perceptual quality and human-reviewed labels for caption accuracy. Set numeric quality thresholds after establishing a representative baseline; do not infer them from image PSNR or these few review probes.

Make completed dataset status reviewable: show accepted and rejected source counts, valid segment counts, paired artifact coverage, and reasons for omissions. Preserve structured metadata alongside exported captions so users can filter uncertain labels without regenerating audio artifacts.

Finish with a tested dependency lock, optional captioning dependencies, a documented Python support range, and CI for the numerical core. Remove tracked bytecode and relocate bulky generated examples to an appropriate artifact workflow while keeping small regression fixtures. Update the README to distinguish demonstrated capabilities from planned training and generation behavior.
