# Experimental locally merged ONNX

Torch remains the production default. These graphs are local derived artifacts,
not the upstream Jina ONNX release. No deployment or live-cache changes are part
of this experiment. The public HTTP schemas and embedding version are unchanged.

The eight-vCPU server validation qualifies **`onnx-merged` with ORT `basic`, eight
intra-op threads and one inter-op thread** as the recommended configuration for a
future production change on this tested host: semantic/ranking parity passes,
representative p50 falls 36.6%, throughput rises 68.8%, and peak RSS falls 3.9%
against Torch using the same eight threads. The straightforward merge without
`basic` does not meet the performance target. Keep request processing sequential;
padded batching has a separate memory/throughput tradeoff. This PR does not switch
the production default or deploy anything. See [server results](#server-results).


## Export contract

`jina-v3-merged-f32-v1`, exporter `1`, opset **18**. Source:
`jinaai/jina-embeddings-v3-hf` at
`d18862d9a48706220815554fac3ebb4dfa46fc28`. Export dependencies are separate from
runtime dependencies: Torch 2.11.0 CPU, Transformers 5.17.0, PEFT 0.21.1,
ONNX 1.23.1, ONNXScript 0.6.2.

Each task loads the pinned native safetensors and only its own adapter in a
fresh process, with `local_files_only=True`, `trust_remote_code=False`, float32,
and eager attention. No downloads, remote model Python, upstream graph rewriting,
quantization, FP16, or pooling changes are involved.

The native Transformers adapter mixin does not expose `merge_and_unload`.
Instead, all ordinary linear LoRA targets are independently reconstructed as
`base + scaling * (B @ A)` (with transpose if required), then merged using
PEFT `layer.merge(safe_merge=True, adapter_names=[task])`. Expected and merged
float32 matrices must have identical SHA-256 digests. Embedding LoRA matrices use
`base + scaling * (B @ A).T` in blocks of 1024 vocabulary rows. This avoids the
multiple full vocabulary-table copies of PEFT's safe merge. Tiny-model tests
compare the embedding result directly with PEFT's `get_delta_weight` formula.
Full and blocked GEMM are [not guaranteed bit-identical across CPU kernels](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html). The
cross-block fixture checks exact equality for small dyadic weights, and checks
both random float32 results against an independent float64 reference using the
standard dot-product/scaling/addition rounding bound. This does not change any
embedding parity threshold. Embedding audit hashes certify the specified blocked
formula; linear audit hashes independently certify PEFT's same-shape merge.
Every target is audited: 145 linear matrices and two embedding tables per task,
including the unused native pooler. Each adapter wrapper is then replaced by its
plain base layer. Parameters are frozen.

The first full-table merge attempt was killed with exit 137 on this host;
row-block embedding merge completed. This is an export-memory observation, not
an inference benchmark.

Before export, the entire parity corpus is evaluated through both unmerged and
merged native models. The frozen component/cosine/norm thresholds apply here
as well. Failure stops export. `TokenEmbeddings` exposes only native token states;
it supplies the equivalent additive 4D attention mask to avoid tracing the
Transformers Python mask factory. Exact native/wrapper equality is checked using
a padded two-item input before exporting.

`torch.onnx.export(dynamo=True, optimize=False, external_data=True)` preserves
dynamic batch and sequence dimensions (capture bounds batch 1–16, sequence
2–8192). ONNX checker and structural checks run after export. No optional ORT
passes are enabled in the initial backend. CPU execution is sequential, inter-op
is fixed at 1, intra-op is explicitly configurable from 1 to 8. Spinning,
CPU memory arenas, and memory patterns are disabled for the baseline.

After completing the one-thread baseline benchmark, structural inspection found
144 transposes of constant weight initializers in each export. A separate
`--merged-optimization basic` experiment enables `ORT_ENABLE_BASIC` at session
initialization. [ORT documents this level](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
as constant folding and other basic semantics-preserving rewrites. It does not
enable the extended transformer/GELU/attention fusion levels. The stored graphs
and manifest remain unchanged. The runtime setting is recorded in each new
parity/benchmark report, and must pass the same real-model parity gate before
benchmarking. The baseline default is still `disabled`; the completed baseline thread sweep
used that setting.

## Layout and provenance

```text
export-directory/
  manifest.json
  tokenizer/{config.json,tokenizer.json,tokenizer_config.json,...}
  retrieval-query/{model.onnx,model.onnx.data,merge-audit.json}
  retrieval-passage/{model.onnx,model.onnx.data,merge-audit.json}
```

Both graphs accept only `input_ids` and `attention_mask`, int64
`[batch, sequence]`; output `text_embeds` is float32 `[batch, sequence, 1024]`.
There is no task input or LoRA task-bank selection. Python/NumPy performs the
same masked mean pooling and float32 L2 normalization as upstream OnnxBackend.
Requests still execute one text at a time.

The operator-owned manifest identifies the source repository/revision, source
file hashes, tool versions, contract/exporter version, opset, dtype, task and
source adapter, plus SHA-256 and byte size for every generated file. Runtime
checks the contract/source identity, task-to-path mapping, required graphs and
external data, tokenizer files, paths, and every listed digest before creating
sessions. The manifest is the operator's trust root, not a signed upstream
release; replacing both manifest and files requires operator review. Local
artifact digests are never compiled into the application.
The deterministic contract specifies the export method and acceptance rules;
it does not promise byte-identical serialization or float32 results across CPU
platforms. Each independently generated artifact set has its own manifest and
must pass the same audits and parity gate. The server evidence therefore uses
the server manifest rather than the development-host artifact hashes.

Export uses a sibling staging directory, publishes the manifest only after both
tasks complete, and verifies source hashes again before publication. Failure
leaves a diagnostic staging directory. An existing output requires `--force`,
and must identify itself as an earlier export of this contract. Source/output
overlap, including resolved source blob targets, is rejected.
Completed directories/files use modes 0755/0644 so the unprivileged runtime can
read a read-only mount. Stdout is the sorted manifest JSON; progress goes to stderr.

## Reproduction (offline)

Install development tools separately with `uv sync --locked --group export`.
The model cache must already contain the pinned native weights and both adapters.
Use an isolated output outside that cache:

```sh
JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/export_merged_onnx.py \
  --model-root /absolute/audit-cache \
  --output-dir /absolute/derived/merged-v1

JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/compare_backends.py \
  --model-root /absolute/audit-cache \
  --merged-root /absolute/derived/merged-v1 \
  --output validation/merged-onnx-parity.json
```

Stop if either three-way parity comparison fails. Do not relax the thresholds:
component error ≤ 1e-6, cosine ≥ 0.99999999999, norm error ≤ 1e-6, equal token
counts, identical full ranking order, task distinction, exact repeats and
request-size stability. The corpus includes five German/English queries, seven
Kulturbytes-style passages and the existing long multi-section text, all under
both tasks. Tiny generated tests are normal CI; real-cache tests require the
`integration` marker and explicit cache environment variables.

Only after parity passes, benchmark in fresh processes:

```sh
JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/benchmark_backends.py \
  --model-root /absolute/audit-cache --merged-root /absolute/derived/merged-v1 \
  --backends torch onnx onnx-merged --thread-sweep 1 2 4 8 \
  --lengths 32 128 480 1024 --batch-sizes 1 4 --warmups 2 --samples 5 \
  --output validation/merged-onnx-benchmark.json
```

Torch's production intra-op setting is 1. Compare one-thread rows directly;
ONNX 2/4/8-thread rows use more intra-op parallelism and are labeled separately.
The tool records actual token lengths, every raw timing sample, p50, p95,
texts/sec, load seconds, runtime/CPU/thread metadata and process peak RSS. The OS
file cache is not flushed. Request size 4 is sequential, not true graph batching.

The separate benchmark-only `--padded-batching` experiment groups up to four
texts with right padding. It is not enabled by the HTTP backend. After the
sequential baseline, validate it with `compare_backends.py --padded-batching`
using the same model/merged roots and chosen thread count. This checks all corpus
texts in groups, alone, and as a short/long padded pair; the extra pair is also
compared directly with the corresponding Torch vectors. Repeat output remains
exact; request-size comparisons use the unchanged component/cosine gate.
This mode also emits an independent sequential-merged comparison at the selected
thread count. Its parity-process RSS includes both modes; performance and memory
comparisons use the separately isolated benchmark workers.
Only if it passes, run `benchmark_backends.py --padded-batching --backends
onnx-merged` with the same chosen thread count and the same lengths/request sizes.
Use a separate JSON output. The timing inputs remain homogeneous within each
request to preserve the existing benchmark contract; mixed-length masking is
validated separately. Generated tests exercise real ORT batches and nonzero PAD
hidden states, so accidentally including padding in pooling fails CI.

Experimental service selection is explicit:
`ENCODER_BACKEND=onnx-merged`, `ENCODER_MERGED_ONNX_ROOT=/absolute/derived/merged-v1`,
`ENCODER_ONNX_INTRA_OP_THREADS=1`, `ENCODER_ONNX_INTER_OP_THREADS=1` and the existing
license acknowledgement. The original `onnx` backend still loads the upstream
scalar-task graph from the HF cache.
`ENCODER_MERGED_ONNX_OPTIMIZATION=basic` explicitly selects the separately gated
basic-pass experiment; leaving it unset preserves the unoptimized baseline.

The dedicated `runtime-onnx-merged` image inherits the existing minimal ONNX
runtime, with `ENCODER_BACKEND=onnx-merged` and `/merged` as its artifact root:

```sh
docker build --target runtime-onnx-merged -f deploy/Dockerfile \
  -t uranus-research-encoder:onnx-merged-experimental .
```

Mount the complete export directory read-only at `/merged` for isolated
validation. No weights are baked into the image. Torch, PEFT, ONNX and ONNXScript
belong only to export/development dependencies. The default final Docker target
remains `runtime-torch`; no deployment or registry publication is part of this PR.

The locally built dedicated image is **436,338,984 bytes (416.13 MiB)**,
excluding mounted model artifacts. An offline, read-only container import check
confirmed that Torch, PEFT, ONNX and ONNXScript are absent, while the merged
backend imports successfully with ORT 1.30.0 and Transformers 5.17.0. Its runtime
user is `10001:10001`. See [image evidence](validation/merged-runtime-image.json).

## Development-host results

Both native adapter merges passed before ONNX export:

| Task | Audited matrices | Max component difference | Minimum cosine | Max norm error |
|---|---:|---:|---:|---:|
| query | 147 | 1.396984e-7 | 0.9999999999995166 | 7.505195e-8 |
| passage | 147 | 1.201406e-7 | 0.9999999999996017 | 9.099906e-8 |

All 147 base matrix hashes match across the independent task loads. 146 merged
matrices differ between tasks; the unused `pooler.dense` is the sole unchanged
matrix. See [query audit](validation/merged-query-weight-audit.json),
[passage audit](validation/merged-passage-weight-audit.json),
[export manifest](validation/merged-export-manifest.json), and
[graph audit](validation/merged-graph-audit.json).

| Task | Graph bytes | External tensor bytes | Total MiB |
|---|---:|---:|---:|
| query | 6,840,520 | 2,233,278,464 | 2,136.34 |
| passage | 6,738,739 | 2,233,278,464 | 2,136.25 |

The host is an Intel Core i5-8265U with 8 logical CPUs and about 14 GiB RAM.
These local measurements must not be presented as new measurements on the
user's deployment server. Both task graphs remain resident in the baseline,
so duplicated float32 backbone weights are a likely memory cost to measure.
This is a shared development laptop, with other applications active; raw timing
variation and p95 matter. Fresh workers isolate each backend's process memory,
but do not isolate CPU scheduling, thermal behavior or the operating-system
file cache. These local data alone cannot justify a production decision; the
separate server repeat is reported below.
The corpus touches only part of the vocabulary: measured RSS is not an upper
bound for every future workload or for all externally stored tensor pages.

The full three-backend semantic comparison passed; raw per-text metrics and
rankings are in [merged-onnx-parity.json](validation/merged-onnx-parity.json).
Its historical `onnx_norm`/`onnx` field names describe the candidate identified by
`candidate_backend` in each comparison.

| Candidate vs Torch | Max component difference | Minimum cosine | Max norm error | Rankings/tokens/tasks/stability |
|---|---:|---:|---:|---|
| upstream ONNX | 4.777685e-7 | 0.9999999999944622 | 7.833922e-8 | pass |
| locally merged ONNX | 4.237518e-7 | 0.9999999999954730 | 7.673905e-8 | pass |

All 26 task/text combinations are finite 1024-dimensional vectors. Full ordering
of all seven passages is identical for every query. Query and passage embeddings
differ for all 13 texts, and each separately matches its Torch adapter. Repeats
and request-size checks are exact. The largest corpus item has 912 tokens.
After completing the baseline thread sweep, the full comparison was repeated at
the selected eight-thread setting. It passed with the same recorded maximum
component error and minimum cosine for both ONNX candidates; see
[eight-thread parity](validation/merged-onnx-threads8-parity.json).

| Backend | Process-cold load seconds | Parity-process peak RSS MiB |
|---|---:|---:|
| Torch | 9.674 | 3867.35 |
| upstream ONNX | 17.303 | 4833.05 |
| locally merged ONNX | 29.388 | 3826.36 |

These RSS figures are from parity, not the warmed latency benchmark. Graph file
size is not resident-memory size. The manifest is immutable export-time
provenance; its release-parity reminder points to the independent report.

### Sequential baseline benchmark

All 96 cases completed successfully in six fresh processes. Each case used two warmups and five measured requests. All merged rows below have optional ORT optimization **disabled**. Column numbers are intra-op threads; inter-op is 1. The Torch reference uses its production one-thread setting.

The [raw benchmark JSON](validation/merged-onnx-benchmark.json) retains every duration, p95, throughput, process load time, peak RSS, CPU/affinity and runtime metadata. The target lengths 480/1024 produced 478/1023 actual tokens. Request size 4 still means four sequential forwards.

| Backend | Intra-op | Load seconds | Peak RSS MiB |
|---|---:|---:|---:|
| torch | 1 | 7.694 | 3863.86 |
| onnx | 1 | 17.829 | 4944.04 |
| onnx-merged | 1 | 28.051 | 4110.49 |
| onnx-merged | 2 | 26.323 | 4167.95 |
| onnx-merged | 4 | 23.294 | 4175.33 |
| onnx-merged | 8 | 23.183 | 4173.43 |

#### p50 request latency (milliseconds)

| Tokens | Kind | Request size | Torch 1 | Upstream 1 | Merged 1 | Merged 2 | Merged 4 | Merged 8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | query | 1 | 373.15 | 4412.48 | 1101.55 | 794.56 | 536.90 | 461.35 |
| 32 | query | 4 | 1496.83 | 9807.66 | 4614.90 | 3186.52 | 1908.73 | 1675.24 |
| 32 | passage | 1 | 383.85 | 2381.69 | 1178.72 | 784.26 | 465.45 | 414.25 |
| 32 | passage | 4 | 1561.08 | 9210.86 | 4664.18 | 3091.46 | 1939.07 | 1712.99 |
| 128 | query | 1 | 1183.86 | 3053.12 | 1817.53 | 1171.66 | 808.18 | 788.90 |
| 128 | query | 4 | 4668.56 | 14009.32 | 7293.89 | 4799.47 | 3410.72 | 3134.81 |
| 128 | passage | 1 | 1182.76 | 3057.44 | 1793.16 | 1154.27 | 829.05 | 785.11 |
| 128 | passage | 4 | 5219.16 | 23742.83 | 7622.71 | 4788.17 | 3220.52 | 3122.43 |
| 478 | query | 1 | 5070.12 | 7337.22 | 5079.99 | 3742.58 | 2611.52 | 2474.44 |
| 478 | query | 4 | 22251.78 | 29428.23 | 21957.69 | 13855.61 | 10301.22 | 9984.74 |
| 478 | passage | 1 | 5108.08 | 6666.24 | 5154.98 | 3626.83 | 2525.66 | 2503.18 |
| 478 | passage | 4 | 20940.79 | 28363.53 | 21497.66 | 13737.47 | 10187.54 | 10111.79 |
| 1023 | query | 1 | 14060.41 | 17617.30 | 11750.44 | 8278.81 | 6155.91 | 6068.69 |
| 1023 | query | 4 | 59264.25 | 67183.42 | 51040.67 | 32223.46 | 24120.81 | 24061.35 |
| 1023 | passage | 1 | 15575.62 | 16604.45 | 13841.03 | 8079.01 | 5884.62 | 5961.08 |
| 1023 | passage | 4 | 62609.54 | 78381.90 | 84001.33 | 32304.70 | 23751.07 | 23917.82 |

#### Throughput (texts/second), request size 4

| Tokens | Kind | Request size | Torch 1 | Upstream 1 | Merged 1 | Merged 2 | Merged 4 | Merged 8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | query | 4 | 2.67 | 0.39 | 0.87 | 1.25 | 2.05 | 2.33 |
| 32 | passage | 4 | 2.53 | 0.44 | 0.85 | 1.29 | 2.02 | 2.36 |
| 128 | query | 4 | 0.86 | 0.27 | 0.54 | 0.84 | 1.16 | 1.26 |
| 128 | passage | 4 | 0.76 | 0.19 | 0.53 | 0.83 | 1.17 | 1.27 |
| 478 | query | 4 | 0.18 | 0.14 | 0.18 | 0.29 | 0.39 | 0.40 |
| 478 | passage | 4 | 0.19 | 0.14 | 0.19 | 0.29 | 0.39 | 0.37 |
| 1023 | query | 4 | 0.07 | 0.06 | 0.08 | 0.12 | 0.17 | 0.17 |
| 1023 | passage | 4 | 0.06 | 0.05 | 0.04 | 0.12 | 0.17 | 0.17 |

Eight threads was the fastest tested baseline setting by the equally weighted geometric mean of the 12 matched 32–480-token p50 ratios (both tasks, request sizes 1/4). Against Torch, the ratio was 0.7051 (29.5% lower); the corresponding throughput-ratio geometric mean was 1.4121 (41.2% higher). Four threads produced ratios 0.7548 and 1.3081. One-thread merged ONNX was slower overall: ratios 1.6651 and 0.6018. These aggregates are not weighted by production traffic and do not erase the short-text regressions shown above.

The eight-thread peak RSS was 8.0% above Torch and 15.6% below upstream ONNX. The one-thread long passage request-size-four samples ranged from 58.62 to 212.32 seconds (p95 189.45 seconds). No outliers were removed. This shared-host variability limits causal performance conclusions.

### Basic optimization

The separate [basic-pass parity report](validation/merged-onnx-basic-parity.json)
passes at eight intra-op threads: maximum component difference
3.911554813e-07, minimum cosine
0.9999999999954965, maximum norm error 7.457872120e-08.
Dimensions, finiteness, tokens, full ranking order, task separation, repeats and
sequential request-size stability all pass the original gate. The stored graph
files and their hashes are unchanged; optimization occurs during ORT session
initialization. No extended optimizer, quantization, FP16 or approximation is
used.

The [local basic-pass benchmark](validation/merged-onnx-basic-benchmark.json)
is **incomplete**. The requested run order was 1, 2, 4, 8 threads. The first three
workers exited with SIGKILL (-9); the kernel recorded global out-of-memory kills
during the run, with host swap exhausted. One thread preserved 11 of 12 cases;
two and four threads produced no case file (their historical fallback records
lack the thread field). Eight threads completed all 12 representative cases.
That completed run had 3865.97 MiB peak RSS and a 46.41-second load, with aggregate
p50/throughput ratios of 0.5710/1.7411 versus the local one-thread Torch reference.
Those results do not establish a reliable optimum across the failed sweep.
No samples were fabricated or removed. The runner now stops immediately after a
failed or incomplete worker and preserves its requested thread count.

Further local model runs were stopped. Real padded batching and the optimized
1024-token workload were not validated locally. The generated padded-batch tests
pass, but that is not a substitute for real-model validation.

## Server results

At the operator's request, validation runs on the actual eight-vCPU server
using an existing separate native/ONNX test cache mounted read-only. A new isolated
workspace holds code, dependencies, exports and results. Benchmark workers use
all eight intra-op threads, including a benchmark-only Torch override; production
Torch remains unchanged. Containers have all eight CPUs available and a 7 GiB
memory limit with container swap disabled. Model runs have networking disabled.
The server export completed successfully using the same pinned dependencies.
All 147 matrices per adapter passed their exact formula/hash audit; 146 differ
between tasks. Native merge maximum component errors are 1.601875e-7 (query)
and 1.266599e-7 (passage), with minimum cosines 0.9999999999991226 and
0.9999999999995227. See the [server manifest](validation/server-merged-export-manifest.json),
[query audit](validation/server-merged-query-weight-audit.json) and
[passage audit](validation/server-merged-passage-weight-audit.json).
Each server graph is 6,350,470 bytes plus 2,233,278,464 bytes of external tensors
(2135.88 MiB per task). The graph protobufs have identical hashes because their
structure is identical; the external tensor hashes differ, as expected for the
independently merged adapters. Artifact identity includes both files.
The [server three-way semantic comparison](validation/server-merged-onnx-parity.json)
passed with eight ONNX intra-op threads and the unchanged production Torch
semantic reference. All 26 task/text vectors are finite and 1024-dimensional;
token counts, all full ranking orders, task distinction, exact repeats and
sequential request-size stability pass.

| Server candidate vs Torch | Max component difference | Minimum cosine | Max norm error |
|---|---:|---:|---:|
| upstream ONNX | 5.252659e-7 | 0.9999999999932306 | 7.833922e-8 |
| locally merged ONNX, disabled optimization | 4.712492e-7 | 0.9999999999949032 | 7.673905e-8 |
| locally merged ONNX, basic optimization | 3.948808e-7 | 0.9999999999952320 | 7.457872e-8 |

All **80 server timing cases** completed successfully: 48 baseline, 16 basic-pass and 16 padded-batch cases. Each stage exited successfully; see the [stage completion record](validation/server-experiment-state.json). These measurements are separate from the laptop results.
The benchmark uses eight threads for Torch as well as both ONNX backends;
`--torch-threads 8` is a benchmark-only override. Semantic reference comparisons
continue to use the unchanged production Torch configuration.

The server baseline command is:

```sh
JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/benchmark_backends.py \
  --model-root /absolute/audit-cache --merged-root /absolute/derived/merged-v1 \
  --backends torch onnx onnx-merged --threads 8 --torch-threads 8 \
  --lengths 32 128 480 1024 --batch-sizes 1 4 --warmups 2 --samples 5 \
  --output validation/server-merged-onnx-benchmark.json
```

Optional basic and padded runs use separate output files and only
`--backends onnx-merged`, after their corresponding parity command succeeds.
All server stages run sequentially; any nonzero exit stops subsequent stages.


### Server benchmarks

All backends use eight intra-op threads and one inter-op thread. Each case has two warmups and five raw samples, in a fresh worker per backend/configuration. Torch uses a benchmark-only override; production remains unchanged. CPU model: QEMU Virtual CPU version 2.5+. Container affinity: CPUs 0–7; memory limit 7 GiB, no container swap. Other existing host services remain active. Load is process-cold; the OS file cache is not flushed.

| Backend | Load seconds | Peak RSS MiB | Completed cases |
|---|---:|---:|---:|
| torch | 9.130 | 3889.04 | 16/16 |
| upstream | 11.907 | 5227.55 | 16/16 |
| merged disabled | 18.011 | 4211.81 | 16/16 |
| merged basic | 22.065 | 3736.18 | 16/16 |
| merged basic padded | 29.581 | 4294.60 | 16/16 |

#### p50 / p95 request latency, milliseconds

| Tokens | Kind | Request size | torch | upstream | merged disabled | merged basic | merged basic padded |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | query | 1 | 320.95 / 356.96 | 768.13 / 840.25 | 358.12 / 405.52 | 168.80 / 176.09 | 134.32 / 148.72 |
| 32 | query | 4 | 1152.32 / 2017.47 | 2873.33 / 2973.64 | 1455.97 / 1498.17 | 802.58 / 847.48 | 474.30 / 516.43 |
| 32 | passage | 1 | 334.01 / 565.40 | 696.06 / 709.57 | 336.66 / 353.09 | 154.09 / 166.08 | 157.24 / 177.58 |
| 32 | passage | 4 | 1210.96 / 1315.06 | 2849.13 / 2941.94 | 1438.06 / 1483.39 | 624.85 / 700.31 | 457.27 / 478.31 |
| 128 | query | 1 | 750.70 / 2014.42 | 1090.51 / 1150.43 | 702.96 / 766.54 | 466.43 / 522.93 | 444.23 / 463.68 |
| 128 | query | 4 | 2400.68 / 3576.67 | 4136.65 / 4675.09 | 2581.08 / 2718.99 | 1810.81 / 1928.29 | 1559.81 / 1703.10 |
| 128 | passage | 1 | 630.61 / 741.02 | 1071.42 / 1108.02 | 663.08 / 673.36 | 478.89 / 497.06 | 478.84 / 600.16 |
| 128 | passage | 4 | 2530.51 / 2848.29 | 4021.68 / 4175.05 | 2812.13 / 2829.62 | 1826.51 / 1875.75 | 1631.88 / 1728.68 |
| 478 | query | 1 | 2686.65 / 3413.94 | 2615.62 / 2687.97 | 1886.99 / 2649.43 | 1589.26 / 1705.09 | 1628.59 / 1716.07 |
| 478 | query | 4 | 9311.42 / 10270.31 | 10397.87 / 11351.14 | 7486.78 / 7894.83 | 6142.42 / 6731.58 | 6063.47 / 6234.31 |
| 478 | passage | 1 | 2231.92 / 2791.70 | 2602.39 / 2766.30 | 1884.04 / 1898.10 | 1642.05 / 1716.14 | 1513.68 / 1562.83 |
| 478 | passage | 4 | 9821.66 / 10295.41 | 10251.09 / 10726.77 | 7310.15 / 7806.26 | 6403.34 / 6753.23 | 6156.86 / 6278.37 |
| 1023 | query | 1 | 5396.74 / 6762.99 | 5918.60 / 6349.32 | 4289.44 / 4369.61 | 3572.47 / 3837.91 | 3280.84 / 3525.80 |
| 1023 | query | 4 | 22744.38 / 23469.55 | 23425.85 / 23755.89 | 16672.07 / 17172.44 | 15577.27 / 16059.72 | 13840.42 / 14296.42 |
| 1023 | passage | 1 | 6004.76 / 6724.18 | 6095.09 / 6196.56 | 4484.86 / 4806.88 | 3791.81 / 4020.93 | 3694.38 / 3809.92 |
| 1023 | passage | 4 | 22907.95 / 24266.60 | 23318.18 / 24136.09 | 17064.38 / 17678.09 | 15039.25 / 15567.03 | 14325.79 / 15236.59 |

#### Throughput, texts/second, request size four

| Tokens | Kind | torch | upstream | merged disabled | merged basic | merged basic padded |
|---:|---|---:|---:|---:|---:|---:|
| 32 | query | 2.932 | 1.400 | 2.743 | 5.324 | 8.519 |
| 32 | passage | 3.289 | 1.433 | 2.829 | 6.316 | 8.613 |
| 128 | query | 1.453 | 0.941 | 1.532 | 2.194 | 2.498 |
| 128 | passage | 1.580 | 0.986 | 1.449 | 2.210 | 2.477 |
| 478 | query | 0.421 | 0.377 | 0.530 | 0.639 | 0.677 |
| 478 | passage | 0.417 | 0.390 | 0.540 | 0.620 | 0.654 |
| 1023 | query | 0.175 | 0.171 | 0.240 | 0.260 | 0.290 |
| 1023 | passage | 0.176 | 0.171 | 0.236 | 0.267 | 0.278 |

#### Equally weighted 32–480-token comparisons against Torch 8

| Backend | p50 ratio | Throughput ratio | RSS ratio |
|---|---:|---:|---:|
| upstream | 1.5903 | 0.6708 | 1.3442 |
| merged disabled | 0.9712 | 1.0868 | 1.0830 |
| merged basic | 0.6337 | 1.6880 | 0.9607 |
| merged basic padded | 0.5608 | 1.8954 | 1.1043 |

These geometric means weight both tasks and request sizes 1/4 equally; they do not represent production traffic. Raw reports retain every sample and metadata. Only eight threads was measured on this server at the operator’s request; the 1/2/4/8 sweep above belongs to the development host.


The straightforward merged export does **not** meet the material-improvement gate on this server: only 2.9% lower aggregate p50 and 8.7% higher throughput, with 8.3% higher peak RSS than Torch at the same eight threads. It improves long texts but regresses on several short-text cases.

The separately validated **basic-pass** variant meets the target: 36.6% lower aggregate p50, 68.8% higher throughput, and 3.9% lower peak RSS versus Torch 8. Every one of the twelve representative matched cases improves p50 by at least 24%. Loading is slower (22.07 seconds versus 9.13 seconds), and both task graphs require about 4.17 GiB of artifact storage. The fixed backend's Python pooling and sequential request processing remain unchanged.

Raw evidence: [baseline benchmark](validation/server-merged-onnx-benchmark.json), [basic-pass parity](validation/server-merged-onnx-basic-parity.json), [basic-pass benchmark](validation/server-merged-onnx-basic-benchmark.json), [execution constraints](validation/server-execution-context.json), [graph signatures and initializer audit](validation/server-merged-graph-audit.json).

### Real padded-batch semantics

The [server padded-batch comparison](validation/server-merged-onnx-padded-parity.json)
passes the original gate, including all rankings, equal token counts, task
separation and exact repeats. It includes the ordinary groups of four and the
short/912-token pair. Both query and passage request-size comparisons have
maximum component difference **0.0**; minimum cosine is 0.9999999999999998
(float64 comparison rounding). Against Torch, maximum component difference is
3.948808e-7, minimum cosine 0.9999999999952320 and maximum norm error 7.457872e-8.
Padded-batch latency and RSS are measured in a separate fresh worker, not inferred
from the shared sequential/padded parity process. The [raw padded benchmark](validation/server-merged-onnx-padded-benchmark.json)
contains all 16 cases. Relative to basic sequential inference, the six matched
32–480-token, request-size-four cases show 17.5% lower p50 and **20.8% more
throughput** by geometric mean. Across all eight request-size-four cases,
including 1023 tokens, those figures are 15.2% and 17.5%.

The gain is concentrated in short texts: 32-token query throughput rises from
5.324 to 8.519 texts/sec, whereas the 478-token query rises from 0.639 to 0.677.
Peak RSS rises from 3736.18 to **4294.60 MiB** (+14.9% versus basic sequential,
+10.4% versus Torch). Single-item controls also vary between these fresh runs;
not every timing difference can be attributed solely to batching on a shared
host. Timing batches contain equal-length texts; mixed-length correctness passes,
but real mixed-length traffic and padding overhead are not performance-validated.
Keep batching a separately measured experiment rather than enabling it in the
HTTP backend in this PR.

### Embedding compatibility

The validated sequential basic-pass configuration preserves the existing
`embedding_version`: it still describes the pinned native retrieval embedding
contract, not the execution backend. Model revision, adapters, dimensions,
tokenization, pooling, normalization and chunking are unchanged, and the frozen
semantic/ranking gate passes. **No reindex is required** for this validated
configuration. The experimental batching path also passes that semantic gate,
but remains outside the service's HTTP inference path.

## Limits of this experiment

Real-model semantic validation includes up to 912 tokens; the timing grid reaches
1023 actual tokens. The exported shape contract allows 8192 tokens, but this run
does not establish latency, memory capacity or semantic measurements at that
maximum. Dynamic batch capture supports up to 16, while the separate real batching
experiment uses groups of four. The equal weighting in the comparison is not a
production traffic distribution. Only eight threads was tested on the server,
as requested; the development-host 1/2/4/8 sweep does not establish a server thread
optimum. The recommended eight-thread setting is the validated server setting.
No request size 16 performance run or maximum-context capacity claim is made.


## Tests and completion

CI passes **204 tests**, with three real-model integration tests
excluded from normal CI. Ruff passes. Generated tests cover native adapter merge
and tiny-model export, dynamic query/passage graphs without task selection,
pooling/normalization, task separation, manifest and tamper failures, padded
batch invariance, and failed/incomplete benchmark workers. Real-model parity and
all 80 server timing cases were run separately against the existing read-only
test cache. The runtime image was built and import-tested offline, with its
unneeded-package absence verified. The completed server experiment container
exited; derived artifacts and raw evidence remain in the isolated workspace.

## API references

The implementation was checked against the installed pinned package source and
[PEFT's merge semantics](https://huggingface.co/docs/peft/developer_guides/checkpoint)
and [PyTorch's export documentation](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html).
Published documentation can describe newer releases; the executable contract
here is pinned and tested.
