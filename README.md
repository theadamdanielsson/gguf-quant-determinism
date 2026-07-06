# gguf-quant-determinism

[![determinism](https://github.com/theadamdanielsson/gguf-quant-determinism/actions/workflows/determinism.yml/badge.svg)](https://github.com/theadamdanielsson/gguf-quant-determinism/actions/workflows/determinism.yml)

Re-runnable CI evidence for two claims about llama.cpp GGUF quantization:

1. **Default builds are not bit-reproducible across machines.** The same F16, the same quant type (and, where used, the same imatrix) produce different bytes on x86_64/gcc (ubuntu-latest) vs arm64/clang (macos-14). The cause is FP contraction (FMA) in the quant scale-search loops: near-tie scale comparisons flip when the compiler fuses a multiply-add.
2. **Disabling FP contraction fixes it** — and the exact fix proposed upstream in [ggml-org/llama.cpp#25353](https://github.com/ggml-org/llama.cpp/pull/25353) (a per-file `-ffp-contract=off` on `ggml/src/ggml-quants.c`, nothing else) is itself tested here, applied to a pinned llama.cpp master and asserted bit-identical across both platforms.

## The matrix

| Leg | llama.cpp | Builds | Models | Quants |
|-----|-----------|--------|--------|--------|
| `quantize` | tag `b3821` (the tag the published reference quants were built with) | default vs strict (`-ffp-contract=off` + defensive no-vectorize flags, global) | SmolLM2-135M, Llama-3.2-1B | Q4_K_M, Q6_K, Q8_0, IQ4_XS, and Q4_K_M **with imatrix** |
| `master-patch` | pinned master (`20a04b2`) | unpatched default vs **the #25353 patch, default flags otherwise** | Llama-3.2-1B | Q4_K_M, Q6_K, Q8_0, IQ4_XS |
| `msvc` (informational) | pinned master | MSVC default (`/fp:precise`, which does not contract) | Llama-3.2-1B | Q4_K_M, Q6_K, Q8_0, IQ4_XS |

All model and imatrix inputs are sha256-pinned to the published Hugging Face files. The imatrix is passed by an identical relative path on every platform so the `quantize.imatrix.file` KV embedded in the output header is byte-identical too, keeping whole-file sha256 a valid comparator.

SmolLM2-135M is the fast smoke model; its embedding dim of 576 means most of its tensors fall back to non-k-quant paths — only ~30 tensors hit the true k-quant kernels — so Llama-3.2-1B is the evidence that the k-quant path itself is deterministic.

## Results (green run of the full matrix)

- **b3821 default builds:** Q4_K_M, Q6_K, IQ4_XS, and imatrix-Q4_K_M all **differ** cross-arch, on both models. Q8_0 matches (it has no scale-search loop — nothing to fix).
- **b3821 strict builds:** all five quants **bit-identical** cross-arch, on both models. Asserted by the `verdict` job.
- **Master, unpatched default:** Q4_K_M, Q6_K, IQ4_XS **differ** cross-arch; Q8_0 matches — the divergence is alive at master, not a b3821 artifact.
- **Master + the #25353 patch alone:** all four quants **bit-identical** across x86_64/gcc and arm64/clang. Asserted by the `verdict` job. This is the patch as proposed, not a stricter proxy for it.
- **MSVC (informational):** a default MSVC build at the same master commit produces **exactly the patched GNU/Clang hashes** on all four quants. MSVC does not contract at its default `/fp:precise`, so this independently confirms the mechanism — the patch makes GNU/Clang builds produce the bytes MSVC users already get. Net: gcc/Linux/x86_64, clang/macOS/arm64, and MSVC/Windows/x86_64 all emit one hash set per quant.
- **Cost:** single-threaded Q4_K_M quantization of the 1B at master measured 92.5 s default vs 80.5 s patched on ubuntu-latest, and 56.7 s vs 53.3 s on macos-14 — no slowdown observed; treat the differences as run-to-run noise. (An earlier ~33% figure measured at b3821 does not reproduce at master.)

Read the `verdict` job of any run for the full hash tables. CORE CLAIM CONFIRMED means: strict-mode b3821 hashes match cross-arch for all five quants on both models, and the patched master build matches cross-arch for all four quants.

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

## How to re-run

Everything is public and pinned; no secrets, no local state:

1. Fork this repo.
2. Actions tab → `determinism` → Run workflow (or push any commit touching the workflow).
3. Read the `verdict` job output.

The workflow is [.github/workflows/determinism.yml](.github/workflows/determinism.yml); the upstream patch under test is in [patches/](patches/).

## License

MIT.
