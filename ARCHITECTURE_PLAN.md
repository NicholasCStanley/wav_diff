**wav_diff continuous audio and latent generation implementation plan**

Draft dated October 7, 2026. This is the proposed implementation direction following the [technical review](IMPROVEMENT_STRATEGY.md). It replaces the earlier image-first experiment as the main development path. The work described here is planned, not implemented.

Build a reproducible pipeline that preserves complete audio tracks, encodes them into continuous audio latents, and adapts a compatible pretrained generator. Keep the waveform as the source of truth and a corrected complex STFT as a diagnostic reference. Store time continuously and read training windows without spectrogram resizing.

**Product scope and assumptions**

The proposed first generation milestone is text-conditioned instrumental music clips of 10–30 seconds, followed by continuation, inpainting, and longer music. These durations are initial test targets, subject to the selected model's validated limits. Full-song structure and vocals are later capabilities requiring separate evaluation.

The deployment GPU and available VRAM are unresolved. The current workspace could not expose GPU information through NVML, so it is not a reliable hardware benchmark. Initial engineering must support CPU data preparation and optional GPU model adapters. Select model size and training settings after a local forward/backward memory profile; inference memory figures do not establish training capacity.

The repository's single bundled song is suitable for a smoke test, not a representative training or evaluation corpus. Corpus size, musical coverage, and the user's first generation task remain inputs to the training milestone. They do not block implementing storage, timeline correctness, adapters, and reconstruction evaluation.

**Architecture decisions**

| Decision | Proposed choice | Reason |
|---|---|---|
| Source of truth | Original files plus explicit canonical waveform derivatives | Learned latents are lossy and tied to model versions |
| Audio timeline | Complete track with exact sample coordinates | Preserves timing, tails, and caption alignment |
| Learned representation | Continuous audio latents from a frozen pretrained autoencoder | Avoids image interpolation and reduces sequence length |
| Generator | Matched pretrained latent generator, initially adapted with LoRA | Reuses learned audio structure and reduces training scope |
| Conditioning | Existing model text encoder and supported duration controls | Maintains checkpoint compatibility |
| Training examples | Views into a track, with context and valid-region masks | Bounds memory while retaining source identity |
| Diagnostic representation | Native complex STFT without resizing | Separates numerical pipeline defects from autoencoder loss |
| Persistence | Numerical arrays and versioned manifests | Enables validation, partial reads, and reproducible exports |
| Image and video representations | Optional research baselines | A folded spectrogram volume adds geometry that audio does not naturally possess |

One logical track tensor does not require loading an entire track into GPU memory. Storage chunking, encoder processing blocks, training windows, and generation duration are separate configuration choices.

```text
Original audio ──→ canonical waveform and timeline
                        │
                        ├──→ complex STFT reference ──→ reconstruction checks
                        │
                        ├──→ caption intervals and reviewed metadata
                        │
                        └──→ frozen audio encoder ──→ continuous latent cache
                                                         │
                                         window sampler and conditioning
                                                         │
                                            pretrained generator + LoRA
                                                         │
Prompt + duration ───────────────────────→ generated latents
                                                         │
                                                 matched decoder
                                                         │
                                           waveform + generation record
```

**Model selection and compatibility**

Evaluate the Stable Audio 3 small music model and its matched SAME-S autoencoder as the first candidate. Evaluate a larger matched pair only if measured quality and hardware capacity justify it. The official project exposes audio encoding, generation, and adaptation workflows, making an integration more appropriate than building a diffusion model from scratch. [Official repository](https://github.com/Stability-AI/stable-audio-3).

Stable Audio Open 1.0 is a fallback reference with a waveform autoencoder, text encoder, and latent diffusion transformer. It advertises up to 47 seconds of stereo output. Treat it as a separate backend with its own latent space and model configuration. [Model card](https://huggingface.co/stabilityai/stable-audio-open-1.0).

Before adopting either backend, pin the repository revision, dependency environment, checkpoint revision and hashes, and the permitted local access path. Resolve model access terms when obtaining weights; planning does not download or accept access terms. Record the model's supported platform, duration limits, dtype, attention implementation, sample rate, channel count, and conditioning fields.

The generator, autoencoder, latent normalization, text encoder, and sampler must form one tested configuration. Do not substitute an independently attractive codec without verifying compatibility or retraining the generator. A decoder with excellent reconstruction can still produce latents that are difficult to model.

Use the selected backend's supported base checkpoint and objective for adaptation. Keep its encoder and decoder frozen initially. Limit LoRA to the generator first and leave text and duration conditioning frozen unless an experiment establishes a benefit. Stable Audio 3 documents base-checkpoint adaptation and layer filtering; exact filters must be checked against the pinned implementation. [Adaptation workflow](https://github.com/Stability-AI/stable-audio-3/blob/main/docs/workflows/lora.md).

**Data layout and contracts**

Use these internal tensor conventions:

| Artifact | Stored layout | Required meaning |
|---|---|---|
| Canonical waveform | `[audio_channels, samples]` | Explicit rate and channel policy; floating-point reference |
| Optional STFT reference | `[audio_channels, frequency, frames, real_imag]` | Complex values and complete inverse-transform metadata |
| Learned latents | `[latent_channels, latent_frames]` | Backend identity, scale convention, time mapping, and valid extent |
| Model batch | `[batch, latent_channels, latent_frames]` | Adapter converts to the backend's required layout |

The learned representation is not an RGB volume. The batch dimension makes a latent batch a rank-three tensor; that alone does not make it a video representation.

Start with one memory-mappable `.npy` latent array per track, using `allow_pickle=False` on load, and JSON metadata. Store derived canonical audio as float WAV where a persistent derivative is useful. Keep original files unchanged. Full-track STFT arrays are optional diagnostic artifacts, not a mandatory second copy of every training track.

```text
dataset/
    dataset.json
    tracks.jsonl
    splits.json
    tracks/<track_id>/
        track.json
        canonical.wav
        captions.jsonl
        representations/<representation_id>/
            latents.npy
            representation.json
    runs/<run_id>/
        config.json
        metrics.json
        checkpoints/
        examples/
```

The representation identifier fingerprints the source identity, preprocessing, model revision, encoder settings, latent normalization, posterior policy, dtype, and processing mode. Caption revisions have their own identity so a wording change does not force audio re-encoding.

Track metadata records original and canonical sample rates and sample counts, channel transformations, source location and hash, optional gain changes, and trim offsets. Defaults preserve silence and stereo; mono-to-stereo conversion for a model is explicit. Comparison references are the canonical waveform for model reconstruction and the original waveform for auditing preprocessing.

Representation metadata records tensor shape, valid samples and frames, padding, encoder delay if any, latent time mapping, context policy, and artifact hashes. Record whether latents are posterior means or seeded samples. Follow the generator's expected latent distribution rather than choosing a posterior policy only for convenience.

Do not infer latent geometry from prose examples. Inspect the pinned model and test lengths just below, at, and above its stride and padding boundaries. Record any additional packing or patchification. If streaming is supported, confirm its relationship to full-pass encoding experimentally.

Write temporary artifacts and validate them before atomic publication. Mark a representation complete only when its manifest and array agree. Interrupted writes must remain visibly incomplete. Regeneration must never silently reuse incompatible caches.

**Timeline and window sampling**

Define caption and content intervals as half-open sample ranges in the canonical timeline. Preserve the mapping back to original audio. Use integer sample coordinates for persistence and convert to seconds only at UI and backend boundaries.

Build training windows by selecting a content interval, aligning reads to the encoder's actual time grid, and adding the required left and right context. Keep valid masks and the requested duration separate from padded batch length. Use duration buckets for efficient batching. Padding must be excluded from the training loss and handled by attention or the backend's documented equivalent. If a backend cannot honor masks, use compatible valid-length batches or explicitly reject the configuration.

Whole-track encoding followed by latent cropping is not automatically equivalent to encoding isolated waveform crops. Convolutions, attention, padding, and normalization can change boundary values. The adapter must specify which mode matches its training distribution. Test full-pass, context-crop, and supported chunked encoding before enabling a cache workflow. If global context prevents bounded equivalence, use the backend's validated fixed-window path and retain the track manifest as the authoritative timeline.

Decode with sufficient context, compensate only for documented or measured delay, and crop to the exact requested sample interval. Compare errors both over the complete signal and near processing boundaries. Crossfading can reduce audible seams but must not conceal missing samples or replace conditioning on previous musical context.

Stable Audio 3 provides latent save/load and chunked autoencoder workflows. They are candidate adapter capabilities, not proof of seamless behavior on this corpus. [Autoencoder workflow](https://github.com/Stability-AI/stable-audio-3/blob/main/docs/workflows/autoencoder.md).

**Captioning and conditioning**

Accept manual captions before requiring an automatic model. Store track-level descriptions separately from interval-level observations. A track's global genre description must not imply that every short crop contains every listed instrument.

Move MuQ and Qwen behind optional adapters and fix the sample-rate and vocabulary defects described in the review. Preserve score, model revision, prompt, and review status. Permit abstention. Let interval overlap rules decide which local captions apply to a training window; flag ambiguous combinations for review.

Use a human-reviewed evaluation set independent of the caption generator. Cache text embeddings only while the text encoder and preprocessing are frozen. Record artist, title, and other identifiers as metadata, with explicit control over whether they enter conditioning.

Start with text and duration fields already supported by the checkpoint. Exact beat grids, section timelines, key controls, and new conditioning channels require model support and targeted training; storing those annotations alone does not make the generator obey them.

**Training and generation strategy**

Run the unmodified pretrained generator first to establish a quality and resource baseline. Then run one small, reproducible adaptation experiment using a fixed train/validation split and a fixed set of held-out prompts and seeds. Include both target-style prompts and broader prompts to detect loss of general capability.

Precompute latents only after the frozen encoder passes reconstruction and context tests. Keep the text encoder frozen for the first experiment. Use the backend's objective, loss weighting, conditioning dropout, sampler, and guidance conventions. Verify that only intended parameters receive gradients. Track per-example valid lengths and dataset coverage.

Profile a complete optimization step, including backward computation, optimizer state, and checkpointing. Increase window duration and batch size only after measuring peak memory and throughput. Gradient accumulation changes effective batch size but does not solve an individual window exceeding memory. Mixed precision and gradient checkpointing are measured options, not assumptions about GPU support.

Save inference adapters separately from resumable training state. A complete resume includes optimizer and scheduler states, step, data position or sampler state, random-number states, configuration, and dataset fingerprint. Reloading adapter weights alone is a warm start. If upstream training only supports warm starts, implement full resume or label the limitation accurately.

Every generated file should have a record containing model and adapter hashes, prompt, duration, seed, sampler settings, dependency versions, and output sample count. Describe reproducibility within a pinned environment; do not promise bitwise equality across GPU architectures.

For longer music, first extend within the model's validated context range. Evaluate continuation and inpainting with retained audio context. Section planning and hierarchical generation are later experiments if measured musical structure remains inadequate. Independent clip generation followed by concatenation is not evidence of coherent full-song generation.

**Implementation sequence**

Each milestone is a separately reviewable change. The command names below are proposed interfaces and do not exist yet.

| Milestone | Scope | Acceptance evidence |
|---|---|---|
| M0 — Contracts and fixtures | Package skeleton, schema versioning, configuration, source identities, deterministic splits, synthetic fixtures | CPU-only schema and identity tests; duplicates stay within a split; optional model imports do not affect core CLI |
| M1 — Continuous audio reference | Explicit preprocessing, complete timelines, float waveform storage, native complex STFT round trip | Exact output length; zero unexplained delay; edge impulses survive; non-silent float32 fixtures target at least 100 dB SNR; silence tested by absolute error |
| M2 — Pretrained adapter and benchmark | One matched model family, standalone encode/decode, shape and padding probes, reconstruction report | Finite outputs, exact declared crop length, stereo retained, measured full/chunk boundary behavior, memory and latency recorded |
| M3 — Dataset production | Per-track latent cache, captions, window loader, masks, atomic completion, resume and cache invalidation | No orphan records; no cross-track windows; complete declared coverage; caption intervals align; interrupted jobs recover safely |
| M4 — Pretrained generation | Text/duration interface and generation records using the selected backend | Repeatable smoke runs in a pinned environment; valid output duration; fixed-prompt baseline and resource measurements |
| M5 — Controlled adaptation | Frozen codec and conditioner, generator LoRA, matched objective, validation, full resume | Gradient audit; tiny-set overfit diagnostic; resumed versus uninterrupted comparison; held-out comparison with base model |
| M6 — Longer audio and editing | Duration expansion, continuation/inpainting, local listening interface | Context and seam evaluation; preserved regions checked; long-range rhythm, repetition, and section consistency assessed |

M0 → M1 → M2 is the critical initial path. M3 and M4 follow a validated backend, and both precede M5. M6 depends on demonstrated quality and available hardware. No credible training schedule or quality promise can be made from the bundled song alone.

Suggested package boundaries are `audio`, `representations`, `dataset`, `captioning`, `backends`, `evaluation`, and `cli` under `src/wav_diff/`. Keep backend-specific model and trainer code upstream where practical. Add an adapter only for a tested need; avoid creating a generic multi-backend framework before the first backend works.

The proposed CLI operations are `ingest`, `encode`, `caption`, `validate`, `reconstruct`, `evaluate`, `generate`, and `train`. Each accepts explicit configuration and returns structured status. Core data work must run without model downloads. An optional local listening interface comes after CLI behavior is stable.

**Evaluation and decision rules**

Maintain four separate evaluations: preprocessing fidelity, reference STFT fidelity, learned-autoencoder fidelity, and generated-audio quality. A good reconstruction score cannot substitute for prompt adherence or musical structure.

The initial benchmark should combine synthetic fixtures with approximately 20–30 diverse, independently selected recordings covering quiet passages, transients, bass, dense mixes, stereo spatial information, and long tails. This is a proposed engineering benchmark, not a statistically representative product study. Hold source tracks and detected duplicates in one split; reserve a final test set from tuning.

For reconstruction, measure sample coverage, delay, waveform SNR where meaningful, multi-resolution spectral error, loudness change, peaks, stereo correlation, silence behavior, and boundary-local error. Measure learned-codec output without automatically normalizing away gain differences. Preserve float output for evaluation; apply playback or delivery level handling explicitly.

For generation, use blinded listening comparisons, prompt adherence, audible defects, rhythmic stability, and diversity. Add text/audio similarity as supporting evidence, using an evaluator independent of the automatic captioner where possible. Do not use a small-sample distributional score as the sole decision rule. Check similarity to training recordings when assessing apparent adaptation improvements.

Hard gates are structural: exact declared lengths, valid masks, finite arrays, no silent truncation, no split leakage, and complete manifests. Learned-codec and generation quality thresholds must be chosen from baseline results and the intended listening use case. Publish per-track distributions and failure examples rather than a single average.

Choose the first model pair that meets the declared quality target within measured resource limits. If the autoencoder fails the listening benchmark, change the matched backend or investigate codec adaptation before expanding diffusion training. Changing the encoder invalidates its latent caches and may require corresponding generator adaptation.

**Migration and the first concrete deliverable**

Retain existing scripts and sample outputs as legacy reference artifacts. Mark the PNG representation as legacy in documentation and keep it out of the new default workflow. Rebuild new datasets from source audio; do not treat already resized PNGs as faithful conversion inputs.

Link this plan from the README, add an isolated dependency environment with a tested lockfile, and keep optional model dependencies separate from the numerical core. Store generated arrays, checkpoints, and benchmark outputs outside ordinary source tracking, with small deterministic fixtures retained for tests.

The first deliverable is M0 and M1: a CPU-testable command that ingests a complete track, preserves its timeline and channels, reconstructs it through the native STFT reference, and emits a machine-readable fidelity report. M2 then answers the main research question: how much quality and memory efficiency the selected audio autoencoder gains or loses on representative material. That evidence determines the model integration and adaptation configuration.
