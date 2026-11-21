# UMA-Nitro・MD整合性監査

2026-10-31。作業開始点は `intel-xpu-fp32` の
`496f8a4e851c1c2fe63aada052dcedeb66481058`。
確認したupstream mainは `61266df4188eeca70fc56baebfba8abd74a0c695`。
作業開始時点のNitroディレクトリはこのupstreamと同一だった。
NitroはPR [#2245](https://github.com/facebookresearch/fairchem/pull/2245)で導入されている。
この監査では既存A/Bの所有権・精度設定を維持し、Nitro、通常UMA、ASE、TorchSim、
バッチサーバー、並列predictorの境界を検査・修正した。

## 適用範囲

**NitroはCUDA専用のTriton/CuTe実装であり、Intel XPU向けSYCLへの移植ではない。**
対応モデル・GPU・制約は[公式calculator文書](https://facebookresearch.github.io/fairchem/ase-calculator/)
および[UMA changelog](https://facebookresearch.github.io/fairchem/uma-changelog/)を参照。
XPUでは既存のportable PyTorch経路を使用する。今回の通常UMA・ラッパー改善はその経路にも
適用されるが、NitroのCUDA Graphや専用カーネルがIntel GPUで動くようになったわけではない。

Nitroは通常の厳密FP32設定と同一ではなく、アーキテクチャ依存のTF32等を利用する。
Aの既定精度やBのopt-in方針は変更していない。BとNitroの未対応な併用を自動で有効にしない。
配布checkpoint・実CUDA・実XPUでの精度／性能の検証は今回実施していない。

## 修正した不整合と実行時の保持量

| 境界 | 修正内容 |
| --- | --- |
| Nitroの入力・CUDA Graph | 入力バッファーのアドレスをcapture keyに含め、キャッシュ置換後の古い入力再生を防止。共有runtimeを既存精度ロックで保護し、モデル所有者を識別して解放する。 |
| 異なるGPU・stream | current deviceをモデルのdeviceに合わせて一時切替。warmup／side stream、staging eventとpinned bufferをdeviceごとに管理する。出力copyの完了eventを次のstreamが待つため、再生による出力上書きと競合しない。GPU常駐入力では不要なCPU staging待機を行わない。 |
| 可変サイズのMD／job | device/pinned input cache、capacity record、定数tensor cacheの保持量を制限。使用中graphの定数はresetまで保持。セル変化だけで再計測せず、毎回のoverflow guardで必要時に再計測・再captureする。 |
| 失敗・モデル解放 | warmup/captureや途中転送の失敗で作業streamを合流。明示解放／finalizerは他モデルのlive graphを破棄しない。コピー・pickle・device移動・state読込時に派生weight cacheとgraph identityを継承しない。 |
| checkpoint／MoLE | CPU FP32 checkpointのexpert tensorsをモデルが所有するコピーにする。再prepare時の組成違いを検査。task登録を途中失敗で半更新しない。standalone preparationでもambient autocast等を抑止する。 |
| 呼出し設定 | Hessian/HVP/embeddingsはNitro割当の前にgeneral eagerへfallback。Nitroが無視していた明示model overrideもHydra経路へ切り替える。明示的なedge_chunk_sizeも無視せず通常経路へfallback。実行中はambient autocastを抑止し、呼出し元の状態を復元する。 |
| ASE | Nitroの一度のE/F/stress計算をASEの結果cacheに保持し、同じ構造への別property要求で重複再生しない。volume、実装property、inference modeの条件を検査する。 |
| バッチ／server | 明示Nitroでは独立リクエストをsingletonに分け、バッチ化だけを理由とする恒久fallbackを避ける。組成等の通常のfallback判定は維持する。 |
| TorchSim | 同じ組成を持つstate.clone()でも、所有する値snapshotとの一致を確認してメタデータを再利用。新しいownerに置換して古いstateのstorageを保持しない。charge/spinのNone・欠落・不正なatom extraへの遷移も再検証する。 |
| AtomicData | ASE変換のtensor factoryをCPUに固定し、他ライブラリのdefault deviceから隔離。外部graph生成では非周期分子の未指定／部分指定cellを所有コピー上で補完し、PBCと呼出し元の構造を維持する。 |
| 並列job | 全metadataの所有snapshotを比較し、変更時は全体refresh、通常MDはpos/cellのみ送る。入力検査・CPU化・直列化・転送準備をcollectiveの通知前に完了する。MoLE fallback後も通信方式を維持する。 |
| 並列リソース | 全placement groupのreadyを待ち、worker futureの失敗を確認。起動途中の失敗、部分的なGP初期化、解放中の中断でも所有リソースの解放を試行。標準UMAではvalidatorのbound methodによる余分なCPUモデル保持も解消する。 |
| RayのGPU予約 | `cuda:N`でもGPUを予約し、actor内部にはbackendだけを渡す。driverで実行するrank0と予約actorのnode・GPU UUIDを照合し、異なるGPUを黙って使う構成を起動前に拒否・解放する。 |

キャッシュの上限は保持量の**境界**であり、任意に大きい入力に対する固定VRAM上限ではない。
定数cacheは現在の一つのlayoutだけでsoft byte budgetを超えることがある。
allocatorのreserved bytes、PyTorch compile/Triton cache、CUDA driverの保持量は別に評価する。
診断のcache byte値はtensor要素数に基づく概算で、共有storageを重複計上する場合がある。
モデルごとのfull `AtomicData.clone()`、外部aliasからのmetadata変更検査、必要な出力コピーは残した。

## 高速経路とCPUでの測定

通常UMAのgraph処理では、ゼロ距離辺がない場合の不要な索引・gatherを削減し、Verletの
近傍数が上限以下のときの再選択と、再構築直後の重複コピーを省いた。
境界条件でfallbackしても指定されたcell／pairwise探索方式を維持する。
初期化済み設定の比較で毎step deepcopyせず、全対象fieldを比較する。
Nitroの組成検査では所有snapshotを使い、CPU入力をGPUへ送った直後に再びCPUへ戻す処理を避けた。
容量見積り用のgeometry/composition keyも、CPU由来入力では実行dtypeに合わせて転送前に作る。
immutableな値をその呼び出しの引数として渡し、data・モデル・大域状態にhintを残さない。
overflow時の再試行では同じkeyを使い、通常モデルへのfallback時には破棄する。
GPU由来入力とhintなしの直接呼び出しは従来のkey生成経路を使う。

凍結linearのfolded GEMMをno-grad／energy-only経路にも適用した。
これはupstreamの公開PR [#2264](https://github.com/facebookresearch/fairchem/pull/2264)と同じ方向の変更で、
学習時の重み勾配と入力の一次・二次微分を回帰試験した。
並列metadataとASE array-like入力についても、公開PR
[#2259](https://github.com/facebookresearch/fairchem/issues/2259)、
[#2260](https://github.com/facebookresearch/fairchem/pull/2260)、
[#2263](https://github.com/facebookresearch/fairchem/pull/2263)を参照した。
未merge PR全体をそのまま取り込んだものではない。

以下はPyTorch 2.15.0+cpu、FP32、2 threadsの**個別処理**測定。
並行監査中のためノイズがあり、Nitro全体・B70・CUDAの速度やMDのns/dayを表さない。
数値の詳細は[intel_xpu_nitro_measurements.json](intel_xpu_nitro_measurements.json)に記録した。

| 対象 | 修正前 | 修正後 |
| --- | ---: | ---: |
| 15,000原子・750,000辺の幾何計算 | 53.21 ms | 29.09 ms |
| 同forward＋backward | 83.76 ms | 55.06 ms |
| 同forwardの累積正allocation | 84.75 MB | 51.75 MB |
| Si 1,000原子・warm Verlet・近傍上限300 | 5.72 ms | 4.44 ms |
| TorchSim cloned-state変換・15,000原子 | 1.403 ms | 0.627 ms |
| Nitro組成確認・15,000原子 | 72.6 µs | 19.7 µs |
| 準備済み設定の比較 | 70.75 µs | 4.71 µs |
| strided frozen linear・65,536辺・no-grad | 228.27 ms | 156.43 ms |

allocation欄はprofilerの累積正割当量であり、peak RAM/VRAMではない。
linearの高速化には一時contiguousコピーが必要で、このケースでは累積正割当が
67.1→201.3 MBに増えた。既存の保存力計算では既にfolded経路を使用しており、今回の差分は
主にno-grad／direct-force経路に作用する。全処理が省メモリ化したという意味ではない。
FP32演算の並べ替えには丸め差があり、linear測定の最大絶対差は約3.8e-5だった。
低精度化や近傍削除は導入していないが、bitwise一致を保証する変更ではない。

## 実行した試験

Python 3.12.14、PyTorch 2.15.0+cpu、Triton 3.9.0、NumPy 2.4.6、ASE 3.29.0、
torch-sim-atomistic 0.6.2、e3nn 0.6.0、Hydra 1.3.8、OmegaConf 2.3.1、Ray 2.59.0。
実NitroのPython host moduleをimportし、GPU依存操作を模擬して制御経路を検査した。
CUDA／XPUカーネルをCPU上で実行したという意味ではない。

実際の小型ランダム重みUMAによるE/F/stress、Hessian/HVP/embedding fallback、
ASE VelocityVerlet、TorchSim NVE、cell変更を含む近傍cache更新を試験した。
配布UMA-S checkpointによる物性精度・長時間MD安定性の検証とは区別する。
parallel試験はCPUのthreaded Ray/Gloo代替で実受信loopを通したもので、live NCCLではない。
実Glooの追加試験はmultiprocessingのsocket作成がEPERMとなり、分散初期化前に阻まれた。
finite guardは実Triton関数をCPU interpreterで36ケース追加確認した。Triton 3.9 interpreterが
論理maskをfloat32の0/1としてatomic ABIへ渡す問題には、0/1であることを検査した上でboolへ
変換するinterpreter専用adapterを使用した。これはCUDAコード生成・競合の検証ではなく、
通常pytestの成功件数にも加算していない。

最終版の統合回帰試験は **1,525 passed、42 skipped、0 failed、0 errors**
（445.15秒、4 workers）。既存A/Bと、今回変更が影響するdatasets・ASE・TorchSim・server・
predictor・graph・Nitro・MD script・分散cleanupをまとめて実行した。
追加修正を含む最終ソースと、変更後の仕様を検査する既存テストでの結果である。
途中の個別試験の件数は重複加算していない。

42件のskipは、CUDAが必要な36件、実XPUを要する5件、Nitro対応GPUを要する1件。
実機がないことによるskipを模擬試験の成功で置き換えていない。
全変更28ファイルのpre-commitと`git diff --check`も通過した。

再実行コマンド（repository root、開発依存関係・extras・pytest-xdist・Triton導入済み）:

```bash
mapfile -t test_paths < docs/core/common_tasks/intel_xpu_nitro_test_paths.txt
CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest "${test_paths[@]}" \
  -c packages/fairchem-core/pyproject.toml -o addopts='' \
  -q -n 4 -m 'not pretrained' --disable-warnings --tb=short -r s \
  -k 'not task_initialization and not homogeneous_batching and not stress_computation and not empty_batch_error and not charge_spin_handling and not model_output_validation and not invalid_task_name_raises_error and not test_2_atoms_2_ranks_gloo_a2a_collect'
```

`-k`は、この環境で禁止されるsocketに依存する既存分散fixtureの試験を除く。
実分散環境ではこの除外を外す。配布checkpointが必要なpretrained試験も除外した。


## 実機でのMDベンチマーク

追加した`fairchem.core.scripts.benchmark_md`は、固定cellのASE NVEを実行し、初回計算、
warmup後のstep時間、有限値検査、全エネルギーdrift/atom、allocatorのallocated/reserved/peak、
Nitro capture/replay/probe/overflow件数をJSONに保存する。
軌跡やGPU出力tensorを蓄積しない。呼出し元のAtomsとNumPy大域乱数状態を変更しない。
時間にはASE処理と出力検査・同期を含む。Nitro指定時のsilent fallbackは成功扱いせず停止する。

Intel FP32の例（Si 1,000原子。実際の用途の構造は`--input structure.extxyz`で指定）:

```bash
python -m fairchem.core.scripts.benchmark_md \
  --model uma-s-1p2p1 --device xpu:0 --inference-settings high-precision \
  --task omat --repeat 5 5 5 --steps 1000 --warmup 20 \
  --timestep-fs 0.1 --sample-interval 25 --output md-xpu-fp32.json
```

Nitro対応CUDA機ではmarginを**別プロセスごと**に比較する。
この環境にはGPUがなく、以下の実機tuningは未実施である。

```bash
for margin in 1.1 1.3 1.5; do
  python -m fairchem.core.scripts.benchmark_md \
    --model uma-s-1p2p1 --device cuda:0 --inference-settings nitro \
    --task omat --repeat 5 5 5 --steps 1000 --warmup 20 \
    --timestep-fs 0.1 --edge-margin "$margin" \
    --sample-interval 25 --output "md-nitro-margin-${margin}.json"
done
```

`UMA_NITRO_EDGE_MARGIN`はimport時に評価されるため、同一processでの後変更では比較しない。
速度だけでなくoverflow、capture回数とVRAMを比較し、その後に大きい構造・長いstep数で確認する。
allocatorのreservedだけの増加をleakと断定しない。モデルを破棄した後の保持量やdriver側の
メモリも必要なら別途測る。driftはmodelとdtに依存し、このharness自体は合格閾値を決めない。

## 残る制約

- **実B70/XPU、実CUDA、live Ray/NCCLは未検証。** 模擬APIの成功は実カーネルの正しさ・
  driver互換性・速度・長時間の非leakを保証しない。
- Nitroはsingle-system/single-GPU。異なる組成や未対応propertyでは通常UMAへfallbackする。
  singleton schedulingは同一predictorで無制限に異種組成をNitro実行できる機能ではない。
- 並列CUDAではrank0予約がdriverと同じnode・driver論理CUDA 0と同じUUIDである必要がある。
  `cuda:N`のNによる物理GPU固定は行わず、割当はRayが管理する。UUIDを取得できない環境も
  明示的に拒否する。複数node／GPUの自動remapを実装・検証したものではない。
  `close()`のSTOP broadcastは通信groupのtimeoutに従い、その後のRay待機だけが30秒指定である。
  終了処理全体が30秒以内で完了する保証はない。
- Aのneighbor_skin・graph_search_method・graph_edge_order・quaternion等はHydra用のhintで、Nitro専用graphを変更しない。compile指定もfallback側の設定である。
- Nitroのarchitecture選択と低level `set_force_precision` はprocess-wide。
  共通ロックはfairchem内の競合を防ぐが、外部ライブラリの直接大域設定変更は管理しない。
  大規模jobで異なるNitroモデルを交互使用すると一つのlive graphを再captureするコストが残る。
- コピー境界とmetadata／displacement確認は意図して残した。省略すると外部NumPy等からの
  更新を見逃すため、物理GPUでのprofileなしに同期をすべて除去していない。
- checkpoint expertの独立コピーは、初期load中のCPUメモリを増やす。merge後には解放する。

この監査は記載した入力・依存関係・実行経路での確認であり、すべての構成で不具合がないという保証ではない。
