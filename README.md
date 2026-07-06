# gguf-quant-determinism

Tests whether llama.cpp (pinned b3821) k-quant quantization is bit-identical across x86_64/gcc (ubuntu-latest) and arm64/clang (macos-14) when FP contraction is disabled (`-ffp-contract=off`, plus defensive no-vectorize flags).
Why: if a single pragma-equivalent flag makes quantization deterministic across build environments, that is the basis for an opt-in deterministic quantization mode upstream.
Both platforms quantize the same sha256-pinned F16 GGUFs to Q4_K_M / Q6_K / Q8_0 (no imatrix, so whole-file sha256 is a valid comparator), in default and strict build modes.
Two models run in the matrix: SmolLM2-135M-Instruct (fast smoke) and Llama-3.2-1B-Instruct (representative: SmolLM2's embedding dim of 576 means most of its tensors fall back to non-k-quant paths — only ~30 tensors hit the true k-quant kernels — so the 1B model is the evidence that the k-quant path itself is deterministic).
Read the `verdict` job: CORE CLAIM CONFIRMED = strict-mode hashes match cross-arch for all three quants on BOTH models; default mode is expected to DIFFER cross-arch for Q4_K_M/Q6_K (the contrast) and MATCH for Q8_0.
