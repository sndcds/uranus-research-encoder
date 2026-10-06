# Jina v5 merged FP32 ONNX experiment

This experiment belongs exclusively to the Encoder repository. It does not change
Phase 2D, judgments, relevance policy, chunking, production configuration, Qdrant,
or database state. It does not replace the native Torch backend or remove v3 code.
No quantization, reduced-precision runtime, deployment, or automatic merge is part
of this experiment.

## Frozen reference and gates

The reference is `TorchBackend` at Encoder commit
`ae0f9a66d6fb6c7da18cfa999da046c77f22dedc`, using
`jinaai/jina-embeddings-v5-text-small` revision
`dd76d535f5447ca3897a9c893fb1e612ead98192` and only `adapters/retrieval`.
The complete source-file hashes, native `model.py` hash, and package versions are
in [native-provenance.json](validation/jina-v5-merged-v1/native-provenance.json).
Original dependency pins and native inference implementation remain unchanged.
Source BF16 safetensors are loaded into FP32 exactly as in the native reference;
the exported graph and all weight initializers are FP32.

Commit `9d09839` froze [plan.json](validation/jina-v5-merged-v1/plan.json) and
[selection.json](validation/jina-v5-merged-v1/selection.json) before any ONNX
inference. Every vector and every batch comparison uses these thresholds:

| Metric | Gate |
| --- | --- |
| Maximum absolute component difference | ≤ 1e-5 |
| Mean absolute component difference | ≤ 1e-6 |
| Cosine similarity (comparison in FP64) | ≥ 0.999999 |
| Absolute error of each embedding's L2 norm | ≤ 1e-5 |

Merging changes `Wx + B(Ax)` into `(W + BA)x`; FP32 reduction order can change.
The bounds therefore permit small rounding differences while remaining strict on
normalized 1024-component vectors. Cosine comparisons use FP64 to avoid a rounded
FP32 dot product disguising small errors. These thresholds are not adjusted after
observing outputs.

The independent ranking gates require identical full event ordering, identical
Top-10 membership and order, identical winning chunks, and score differences at
most 1e-5. Exact numerical ties use the original Research comparator: choose an
event's chunk by `(-score, point_id)`, then order events by `(-score, event_id)`.
No epsilon ties or score rounding are introduced. Scores are recomputed using
the same FP64 dot product for each backend. No v3/v5 score comparison is made.

## Selection and limits

The frozen selection has 179 inputs: all 120 original query texts, 56 real v5
candidate chunks, one supplemental short query, one longer natural query, and
one long multilingual passage. The original proposals and benchmark query texts
were checked for exact equality. Source hashes are stored with the selection;
no judgments are copied or evaluated.

The candidate rule includes the v5 top five chunks for `historical-q29`,
`wheelchair-da`, `dance-en`, `concerts-en` and improvement controls
`historical-q02`, `quiet-da`, `creative-da`, `nordic-en`, `songwriters-en`, plus v5
regression diagnostic candidates and shortest/longest examples of each existing
chunk kind. This covers DE/DA/EN, accessibility, venues, atmosphere, outdoor,
multiple-result concerts, abstract queries and multilingual content. Real chunks
reach the unchanged 480-token chunk limit; the supplemental passage is 2163 tokens.

This is a candidate-pool parity experiment, **not a complete 611-event benchmark**.
Original event eligibility is not reconstructed for the subset. Ranking counts
include two supplemental queries and must not be confused with full-corpus
120/120 parity. The 32768-token model limit remains the API limit but is not
claimed as a practically validated inference length. Eager attention at that
length requires resources beyond this experiment's bounded memory profile.

Each input is embedded individually through the unchanged native `embed` method.
Additional native batched forwards test the same model with mixed-length batches
of three, on both left and right padding; ONNX is tested identically. The batch
ordering deterministically interleaves short and long inputs. The graph supports
dynamic batch and sequence dimensions, including singleton inputs.

The native batch control passes all 179 inputs on both padding sides; maximum
component differences are 1.79e-7 (right) and 4.99e-7 (left). See
[native-batch-parity.json](validation/jina-v5-merged-v1/native-batch-parity.json).
This control alone is not a Native/ONNX parity claim.

## Merge and export

[merge-audit.json](validation/jina-v5-merged-v1/merge-audit.json) records all 196
matrices (seven projection targets in all 28 layers) and 392 adapter tensors.
Before mutation, target names, the sole active adapter, rank, scaling, dtype and
layer convention are checked. For each target the implementation independently
reconstructs `W_base + scaling * (B @ A)`, then compares its digest bit for bit to
PEFT's safe merge. Unsupported/transposed variants fail closed. Each verified
wrapper is replaced by its base layer. Remaining LoRA modules/parameters are
rejected, and obsolete adapter metadata is removed. No adapter selection is used
by the exported inference graph.

Design B is used: inputs `input_ids` and `attention_mask` are INT64 with dynamic
`[batch, sequence]`; output `text_embeds` is FP32 `[batch, 1024]`. Prefixing remains
outside the graph: `Query: ` and `Document: `. Native Qwen3 uses **causal** eager
attention; the export supplies an equivalent additive causal/padding mask to
avoid tracing the library's Python mask factory. Pooling finds the maximum valid
position from `attention_mask`, so it works with either padding side. A single
`torch.nn.functional.normalize(..., p=2, dim=1)` is inside the graph. The runtime
must neither pool again nor normalize a second time.

Export uses opset 18, `dynamo=True`, `optimize=False`, external FP32 data and ORT
`ORT_DISABLE_ALL`. [graph-audit.json](validation/jina-v5-merged-v1/graph-audit.json)
records dimensions, initializers, operator domains and external filenames.
Large model files remain outside Git; their hashes and provenance are versioned.

The local and AI-host exports are separate artifacts. Their base-weight hashes
match, but 56/196 merged-matrix fingerprints differ between CPU implementations of
FP32 GEMM. Every matrix independently matches the explicit formula bit for bit on
its exporting host. This is platform-scoped reproducibility, not a claim of
cross-platform byte identity. Both graph identities are retained; the primary
host graph receives its own complete parity run. CPU descriptions and both merge
audits are recorded with the results.

Reproduction from the Encoder checkout (all model loading is offline):

```sh
PYTHONPATH=src .venv/bin/python scripts/export_v5_merged_onnx.py \
  --model-root /absolute/pinned/hub \
  --output-dir /absolute/new-export-directory
PYTHONPATH=src .venv/bin/python scripts/check_v5_onnx_parity.py native \
  --model-root /absolute/pinned/hub --output-dir /absolute/new-result-directory
PYTHONPATH=src .venv/bin/python scripts/check_v5_onnx_parity.py onnx \
  --graph-dir /absolute/new-export-directory --output-dir /absolute/new-result-directory
PYTHONPATH=src .venv/bin/python scripts/check_v5_onnx_parity.py report \
  --output-dir /absolute/new-result-directory
```

The exporter refuses an existing output directory. Use a new result directory for
each execution; retain failed reports. NPZ files preserve complete vectors, while
JSON reports include per-input metrics and every ranking/score/winner difference.
The runtime artifact manifest requires a separately trusted manifest SHA256;
hashes in a colocated manifest alone do not authenticate altered artifacts.

## Execution and performance policy

The AI host runs only new isolated containers: network disabled, model/runtime
mounts read-only, writable experiment output, eight CPUs, eight intra-op threads
and one inter-op thread. Native and ONNX jobs run sequentially. Initial native
parity uses the historical 6 GiB memory/8 GiB memory-plus-swap limits; export and
ONNX parity have 8/10 GiB to accommodate export intermediates and long padded
batches. These memory profiles must be reported rather than conflated.

Performance is blocked until merge, vector, batch, ranking and contract gates
pass. `scripts/benchmark_v5_onnx.py` additionally verifies the tested graph's
manifest identity before loading. Its default comparison is eight CPUs/eight
threads; optional independent processes accept 1, 2 or 4 threads. Each workload
has five warmups and 120 recorded samples, covering query, passage and batch-three
inference. It records model initialization, first inference, mean/median/p95/p99,
throughput, CPU time, wall time and process peak RSS. Raw latency samples and
exact workload IDs are retained. The p99 estimate from 120 samples is coarse.

The historical v3 ONNX retrieval mean (~115 ms) and historical native v5 retrieval
mean (~504 ms) are context only. New measurements must be reported separately.
Embedding-call measurements include tokenization but exclude HTTP/Qdrant and
therefore are not the same end-to-end retrieval metric. Shared-host measurements
are not a production capacity forecast. Speed cannot rescue a failed parity gate.

## Explicit runtime and dependency check

`v5-onnx-merged` requires a reviewed manifest digest in
`ENCODER_V5_ONNX_MANIFEST_SHA256` and an absolute `ENCODER_MERGED_ONNX_ROOT`.
Graph/external data, tokenizer/config, native provenance, merge audit, schema,
revision, dimensions, prefixes, pooling and normalization are verified before ORT
loads. Startup also checks runtime package versions and graph input/output types
and dynamic shapes. Failure leaves readiness false; no fallback is attempted.
The default remains `torch`. `/health` remains liveness; `/ready` and `/version`
identify the new backend while preserving the logical model and embedding version.

The public backend keeps native per-text serialization to preserve the existing
memory behavior. It returns graph vectors directly after finite/shape/norm checks;
it does not renormalize. The separately benchmarked padded graph batch is an
experimental capability, not a claim about HTTP batch throughput.

An initial HTTP check in the full development environment passed all six vector
checks but failed the extra Torch-isolation assertion: Transformers 5.17 imports
Torch when it is installed and does not honor the older `USE_TORCH=0` convention.
[The failed check is retained](validation/jina-v5-merged-v1/http-contract-with-torch-installed.json).
A fresh locked `runtime-onnx` environment with neither Torch nor PEFT installed
passes the same real HTTP checks, including both roles, token counts, readiness,
version, invalid-kind rejection and all vector thresholds. No tolerance was changed.
