# gguf-quant-determinism

Tests whether llama.cpp (pinned b3821) k-quant quantization is bit-identical across x86_64/gcc (ubuntu-latest) and arm64/clang (macos-14) when FP contraction is disabled (`-ffp-contract=off`, plus defensive no-vectorize flags).
Why: if a single pragma-equivalent flag makes quantization deterministic across build environments, that is the basis for an opt-in deterministic quantization mode upstream.
Both platforms quantize the same sha256-pinned F16 GGUFs to Q4_K_M / Q6_K / Q8_0 (no imatrix, so whole-file sha256 is a valid comparator), in default and strict build modes.
Two models run in the matrix: SmolLM2-135M-Instruct (fast smoke) and Llama-3.2-1B-Instruct (representative: SmolLM2's embedding dim of 576 means most of its tensors fall back to non-k-quant paths — only ~30 tensors hit the true k-quant kernels — so the 1B model is the evidence that the k-quant path itself is deterministic).
Read the `verdict` job: CORE CLAIM CONFIRMED = strict-mode hashes match cross-arch for all three quants on BOTH models; default mode is expected to DIFFER cross-arch for Q4_K_M/Q6_K (the contrast) and MATCH for Q8_0.

Upstream fix this evidence supports: [ggml-org/llama.cpp#25353](https://github.com/ggml-org/llama.cpp/pull/25353).

## Quality effect: none measurable

Determinism changes which bytes come out. This section measures whether it changes model quality. It does not.

Setup: Q4_K_M was quantized from the same Llama-3.2-1B F16 + imatrix by two builds of tag b3821 — a default build and a `-ffp-contract=off` (strict) build. The two Q4_K_M files differ by sha256 (they make different near-tie scale choices). Both were evaluated with the same `llama-perplexity` binary on wikitext-2, `n_ctx` 512, 564 chunks.

| Model | PPL | ± |
|-------|----:|----:|
| F16 baseline | 14.0004 | 0.10410 |
| Q4_K_M, default build | 14.4599 | 0.10772 |
| Q4_K_M, strict (`-ffp-contract=off`) | 14.4592 | 0.10772 |

Reading: quantization itself costs ~0.46 PPL over F16. The contraction flag changes PPL by 0.0007 — about 650x smaller than quantization's own effect, and about 150x below the ±0.108 error estimate. The measured conclusion: reproducibility is quality-free.

Why this is expected: contraction only flips near-tie scale candidates in the k-quant search. When two candidate scales are within rounding of each other, either one encodes the block's weights equally well, so choosing between them deterministically does not cost accuracy.
