# gguf-quant-determinism

[![determinism](https://github.com/theadamdanielsson/gguf-quant-determinism/actions/workflows/determinism.yml/badge.svg)](https://github.com/theadamdanielsson/gguf-quant-determinism/actions/workflows/determinism.yml)
[![conversion](https://github.com/theadamdanielsson/gguf-quant-determinism/actions/workflows/conversion.yml/badge.svg)](https://github.com/theadamdanielsson/gguf-quant-determinism/actions/workflows/conversion.yml)
[![chain](https://github.com/theadamdanielsson/gguf-quant-determinism/actions/workflows/chain.yml/badge.svg)](https://github.com/theadamdanielsson/gguf-quant-determinism/actions/workflows/chain.yml)

Re-runnable CI evidence for three claims about how GGUF files get made:

1. **Default builds are not bit-reproducible across machines.** The same F16, the same quant type (and, where used, the same imatrix) produce different bytes on x86_64/gcc (ubuntu-latest) vs arm64/clang (macos-14). The cause is FP contraction (FMA) in the quant scale-search loops: near-tie scale comparisons flip when the compiler fuses a multiply-add.
2. **Disabling FP contraction fixes it** — and the exact fix proposed upstream in [ggml-org/llama.cpp#25353](https://github.com/ggml-org/llama.cpp/pull/25353) (a per-file `-ffp-contract=off` on `ggml/src/ggml-quants.c`, nothing else) is itself tested here, applied to a pinned llama.cpp master and asserted bit-identical across both platforms.
3. **The conversion step (safetensors → F16) is already bit-reproducible.** `convert_hf_to_gguf.py` at the same pinned master produces byte-identical F16 files across OSes, architectures, and dependency versions — provided the inputs are pinned. Together with 2, the whole safetensors → F16 → quant chain is deterministic.

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

## The conversion leg (safetensors → F16)

[conversion.yml](.github/workflows/conversion.yml) converts revision-pinned Hugging Face snapshots with the converter's own pinned requirements and asserts one sha256 per model — the hash produced on an independent arm64 machine outside CI. ubuntu-latest (x86_64) and macos-14 (arm64) must both reproduce it, byte for byte.

| Model | Why it's in the matrix |
|-------|------------------------|
| SmolLM2-135M-Instruct | dense, BF16, fast smoke model |
| Qwen2.5-0.5B-Instruct | dense, BF16, second family |
| granite-3.0-1b-a400m-instruct | MoE (GraniteMoe), experts stored pre-stacked |
| Mixtral-tiny | MoE with per-expert checkpoint tensors — exercises the expert-stacking path |

A second job (`convert-resharded`) splits each single-file snapshot into small shards plus an index.json, converts that, and asserts the **same** hash as the unsharded leg: shard topology does not reach the output.

Every input is closed over — the exact file set and the sha256 of every snapshot file are asserted before converting (digest files in [conversion/](conversion/)). That closure is load-bearing; three things change the output if you let them drift:

- **The source directory name.** `general.name`, `general.basename`, `general.finetune` and `general.size_label` are derived from it by gguf-py heuristics, and `--model-name` overrides only the first. Convert from a directory named exactly like the source repo.
- **README.md.** If a model card is present the converter reads it, and its fields become KVs (license, languages, organization). It is part of the input; pin it or exclude it consistently.
- **Tensor order.** The converter writes tensors in input iteration order. Order-preserving shards reproduce the file exactly; a permuted weight_map permutes the output while every tensor's bytes stay identical (measured: 0 of 290 content hashes changed). Publisher shards are contiguous in practice, but strictly the index is part of the recipe.

Probed locally beyond the CI matrix, all bit-identical: repeated runs, PYTHONHASHSEED, numpy 1.26.4 → 2.5.1, torch 2.11.0 → 2.12.1, Python 3.12 → 3.13, and an F32-weight model where the F16 cast genuinely rounds. One more datum: bartowski's published SmolLM2-135M F16, converted on different hardware with a roughly two-years-older stack, has bit-identical tensor payloads to the pinned conversion here — the diff is header metadata only.

Why this is expected: BF16 → F16 is exact for every in-range value (a 7-bit mantissa fits in a 10-bit one), and the converter does no reductions and no threading, so the numeric surface where machines could disagree is nearly empty. It still had to be measured. Measured scope: dense and MoE text models, single- and multi-shard, F16 output. Not measured: mmproj/vision paths, converter-side quantized outtypes (q8_0 etc.), big-endian.

## The chain leg (attest on one machine, verify on another)

[chain.yml](.github/workflows/chain.yml) is what the two determinism results unlock, run end to end. An ubuntu/x86_64/gcc machine starts from public sources only — the SmolLM2-135M-Instruct snapshot at a pinned revision, llama.cpp at the pinned master (quantize built with the #25353 patch) — converts, quantizes, and writes two linked proofs with [ggufpacker](https://github.com/theadamdanielsson/ggufpacker): F16 from snapshot, Q4_K_M from F16.

A macos-14/arm64/AppleClang machine then receives **only the two JSON statements** — about 4 KB. The F16 and the quant never leave the first machine. From the same public sources it re-derives both steps with `ggufpacker verify-chain` and byte-compares against the attested digests, with the snapshot itself checked against what Hugging Face publishes at the recorded revision. A conversion statement with a broken link to the quant statement must be refused.

A green run means: a different OS, architecture and compiler reproduced the exact quant bytes from huggingface.co and llama.cpp sources plus two small proof files — the whole path from the published safetensors to the deployed quant, checked rather than trusted.

## How to re-run

Everything is public and pinned; no secrets, no local state:

1. Fork this repo.
2. Actions tab → `determinism`, `conversion` or `chain` → Run workflow (or push any commit touching the workflow).
3. Read the `verdict` job output.

The workflows are [.github/workflows/determinism.yml](.github/workflows/determinism.yml), [.github/workflows/conversion.yml](.github/workflows/conversion.yml) and [.github/workflows/chain.yml](.github/workflows/chain.yml); the upstream patch under test is in [patches/](patches/).

## License

MIT.
