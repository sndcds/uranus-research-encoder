# Jina v5 merged FP32 ONNX experiment

This experiment belongs exclusively to the Encoder repository. It does not change
Phase 2D, judgments, relevance policy, chunking, production configuration, Qdrant,
or database state. It does not replace the native Torch backend or remove v3 code.
No quantization, reduced-precision runtime, deployment, or automatic merge is part
of this experiment.

## Executed FP32 parity result — 2026-10-06

The primary AI-host export passes every preregistered parity gate. The complete
[per-input/batch/ranking report](validation/jina-v5-merged-v1/parity.json) preserves
all metrics, rankings, chunk winners and score differences.

| Gate | Result |
| --- | --- |
| Merge | 196/196 exact matches; 392 adapter tensors; max difference 0 |
| Vector | 179/179 PASS |
| Batch | 1074/1074 PASS (native, ONNX and cross-runtime; left/right padding) |
| Ranking | 122/122 full order and Top-10 identical; no changed chunk winners |
| Original queries | 120/120 identical **within the frozen subset** |
| Max component difference | 2.384185791015625e-7 |
| Worst per-input mean difference | 3.409305104895566e-8 |
| Minimum cosine | 0.9999999999990994 |
| Worst final-vector norm error | 6.298541845861649e-7 |
| Maximum per-query score difference | 6.133497080229589e-7 |
| Rank correlation / maximum shift | 1 / 0 for every query |
| Real HTTP contract | PASS, without installed/imported Torch or PEFT |

The rankings cover 56 real chunks from 50 real events plus one synthetic long
passage, hence 51 candidate events per query. The 122 queries include two
supplemental queries. This does **not** establish full 611-event Top-10 parity.
The full snapshot contains 2080 chunks (1094 unique texts). The complete corpus
run was not performed because the mandatory long/padded
checks and fresh performance profiles already require sustained CPU execution;
no smaller-pool result is presented as that optional full-corpus result.

The independently exported local graph also passed all 179/1074/122 checks against
the host native reference; its separate manifest and summary remain recorded.
The final host graph has a distinct fingerprint and was evaluated independently.
The normal test suite passes 264 tests (two opt-in tests deselected), with Ruff
checks, formatting and whitespace validation passing. Actual model/HTTP runs are
separate evidence from these unit tests.

## Frozen reference and gates

The reference is `TorchBackend` at Encoder commit
`ae0f9a66d6fb6c7da18cfa999da046c77f22dedc`, using
`jinaai/jina-embeddings-v5-text-small` revision
`dd76d535f5447ca3897a9c893fb1e612ead98192` and only `adapters/retrieval`.
The complete source-file hashes, native `model.py` hash, and package versions are
in [native-provenance.json](validation/jina-v5-merged-v1/native-provenance.json).
Original dependency pins and native inference implementation remain unchanged.
Source BF16 safetensors are loaded into FP32 exactly as in the native reference;
the exported graph and all weight initializers are FP32. The copied upstream
configuration intentionally retains its original source dtype metadata for hash
identity; graph tensor types, not that tokenizer-side metadata, define inference.

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

[host-merge-audit.json](validation/jina-v5-merged-v1/host-merge-audit.json) records all 196
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
`ORT_DISABLE_ALL`. The [primary host graph audit](validation/jina-v5-merged-v1/host-graph-audit.json)
and [initial local graph audit](validation/jina-v5-merged-v1/graph-audit.json)
record dimensions, initializers, operator domains and external filenames.
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

The [performance execution plan](validation/jina-v5-merged-v1/performance-plan.json)
was committed before timing. The actual
[container launcher](validation/jina-v5-merged-v1/host-performance.sh) records the
read-only source/runtime mounts and identical 6/8 GiB performance memory limits
for both backends. Each profile runs once in the registered sequential order;
there is no randomized repetition or claim of statistical significance.

Performance uses the existing common benchmark Python environment for both backends;
Transformers may import its installed Torch package during ONNX tokenization. The
separate real HTTP check proves that the serving backend also works without it.
Peak RSS therefore describes the measured common environment, not the smallest
possible ONNX-only process.

The historical v3 ONNX retrieval mean (~115 ms) and historical native v5 retrieval
mean (~504 ms) are context only. New measurements must be reported separately.
Embedding-call measurements include tokenization but exclude HTTP/Qdrant and
therefore are not the same end-to-end retrieval metric. Shared-host measurements
are not a production capacity forecast. Speed cannot rescue a failed parity gate.

## Measured performance — 2026-10-06

The new eight-CPU/eight-thread experiment establishes **no general ONNX speedup**.
ONNX query mean is 6.3% higher, passage mean 9.4% lower, and experimental batch-three
mean 8.9% higher in these sequential shared-host runs. All parity gates passed
before these measurements; performance does not change their thresholds or result.

Each row below has 120 measured calls after five workload warmups. The batch row
contains three texts per call; native uses three serial forwards and ONNX one
padded graph forward. These batch results are not HTTP throughput measurements.
The query and passage rows contain one text per call.

| Workload | Runtime | Mean ms | Median ms | p95 ms | p99 ms | Texts/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| query | native | 419.2 | 350.9 | 897.6 | 1113.1 | 2.385 |
| query | onnx | 445.6 | 394.1 | 702.1 | 903.8 | 2.243 |
| passage | native | 1944.4 | 2001.8 | 3381.1 | 4255.7 | 0.514 |
| passage | onnx | 1761.4 | 1772.8 | 3477.0 | 4136.7 | 0.568 |
| batch3 | native | 5673.7 | 6093.9 | 8725.0 | 9290.1 | 0.529 |
| batch3 | onnx | 6176.4 | 6691.6 | 8396.0 | 8998.1 | 0.486 |

| Workload | Runtime | Wall s | CPU s | Peak RSS GiB |
| --- | --- | ---: | ---: | ---: |
| query | native | 50.31 | 383.19 | 3.775 |
| query | onnx | 53.50 | 406.77 | 4.176 |
| passage | native | 233.34 | 1786.29 | 3.775 |
| passage | onnx | 211.38 | 1618.24 | 4.300 |
| batch3 | native | 680.85 | 5222.28 | 3.775 |
| batch3 | onnx | 741.19 | 5673.02 | 4.300 |

| Runtime (8 threads) | Model initialization s | First inference ms |
| --- | ---: | ---: |
| native | 4.216 | 466.2 |
| onnx | 1.226 | 1025.1 |

| Query threads | Native mean ms | ONNX mean ms |
| ---: | ---: | ---: |
| 1 | 646.2 | 1718.1 |
| 2 | 646.3 | 977.0 |
| 4 | 389.4 | 589.8 |
| 8 | 419.2 | 445.6 |
Wall and CPU times cover the measured workload only, excluding initialization,
first inference and warmup. CPU time sums work across process threads. Peak RSS
is the cumulative process high-water mark (raw reports use KiB), not incremental
workload allocation. Model initialization excludes manifest/file hashing and
the shared imports before its timer; it includes tokenizer/backend construction.
First inference is recorded separately before workload warmup.

The 1/2/4-thread rows are secondary query-only profiles. Eight CPUs remain
available in all profiles and inter-op threads remain one. Different profiles
must not be substituted for the primary eight-thread comparison.

Raw samples, workload IDs and complete metrics are in
[the native eight-thread report](validation/jina-v5-merged-v1/performance-native-8.json)
and [the ONNX eight-thread report](validation/jina-v5-merged-v1/performance-onnx-8.json).
The [verification index](validation/jina-v5-merged-v1/performance-verification.json)
references all eight reports: 1440 measured samples, matching ordered inputs,
pinned package versions, graph/plan/parity/contract identities and independently
recomputed mean/median/p95/p99/throughput. The substantial tails and single run per
profile limit interpretation; these are not end-to-end retrieval or production
capacity results. The historical v3 ~115 ms and native-v5 ~504 ms retrieval values
are not included as new measurements in these tables.

## Explicit runtime and dependency check

`v5-onnx-merged` requires a reviewed manifest digest in
`ENCODER_V5_ONNX_MANIFEST_SHA256` and an absolute `ENCODER_MERGED_ONNX_ROOT`.
Graph/external data, tokenizer/config, native provenance, merge audit, schema,
revision, dimensions, prefixes, pooling and normalization are verified before ORT
loads. Startup also checks runtime package versions and graph input/output types
and dynamic shapes. Failure leaves readiness false; no fallback is attempted.
The default remains `torch`. `/health` remains liveness; `/ready` and `/version`
identify the new backend while preserving the logical model and embedding version.

The benchmark native `batch3` workload calls the native backend three times
sequentially; ONNX `batch3` calls the padded graph once. Query/passage workloads
use single inputs. Native timing includes the existing backend validation and
list conversion; ONNX timing covers tokenization and the graph call. These are
embedding-kernel measurements, not measured HTTP endpoint latencies. Raw JSON
`host` fields are ephemeral container hostnames; all runs use the same AI host.

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

## Primary experimental artifact

The primary artifact remains on the AI host at
`/home/awendelk/v5-merged-onnx-20261006/export-v2/graph`. Its independently trusted
manifest SHA256 is:

```text
ae61e7c440f358691bace2f080f5ebe08dcf4f44a778f9066e1b335e03200f6a
```

The initial local artifact at `models/jina-v5-merged-v1` has a different manifest
and weight fingerprint and its own recorded parity result. Do not mix files or
manifest digests from the two exports. Complete native and ONNX vectors are kept
as NPZ files under the host experiment's `output/` and locally under
`models/jina-v5-host-parity-v1/`; the versioned parity report pins their hashes.
No production configuration is changed by recording these artifact locations.
