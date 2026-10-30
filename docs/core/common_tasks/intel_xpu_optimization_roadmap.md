# UMA / Intel Xe optimization research roadmap

調査日: 2026-10-30。対象: `intel-xpu-fp32` ブランチの単一 XPU 推論。
本資料は実装候補の一覧であり、B70 上の速度・精度の実測結果ではない。
既実装の改善と CPU 検証は [performance audit](intel_xpu_performance.md) を参照。

## 分類と現状

- **①**: checkpoint、物理カットオフ、近傍上限、FP32 ネットワークと既存の
  FP64 集約を維持する候補。並列加算・演算順序の変更による丸め差はあり得る。
  bitwise 一致や長時間 MD の軌跡一致を意味しない。微分を含む比較が採用条件。
- **②**: 演算精度、モデル、収束・時間積分条件のいずれかを変える候補。
  「精度低下に見合う高速化」はまだ未検証であり、既定値にはしない。
- 優先度「高」は開発順序であり、速度改善率の予測ではない。
- 今回の 50 件は新規開発・有効化・実機検証の候補。既実装の sparse graph v2、
  Wigner permutation の index 化、active-system MoLE、入力コピー統合は重複計上しない。

現在の XPU preset は `execution_mode="general"`, `compile=False`,
`merge_mole=True`, `tf32=False`, graph v2 を用いる。UMA 全体を独自 SYCL に
書き直した実装ではなく、PyTorch の通常演算を `torch.xpu` 上で実行する。
下層は ATen XPU、torch-xpu-ops の SYCL カーネル、GEMM の oneDNN、
一部線形代数の oneMKL 等で構成される。Intel Extension for PyTorch は必須ではない [S1]。

## ① 精度方針を維持する 33 候補

| ID | 優先 | 候補・削減対象 | 実装箇所・成立条件 |
| --- | --- | --- | --- |
| A01 | 高 | 変位検査付き Verlet skin による近傍候補の再利用 | `graph/compute.py:generate_graph`。外側候補は cutoff+skin で作り、原子変位を毎回確認。現在位置で cutoff と既存 neighbor cap を毎回適用。外側候補を cap で切らない [S5,S6]。 |
| A02 | 中 | 全ペア / cell list / BVH の形状別選択 | 微小分子・通常密度・疎な大箱で使い分ける。現在の疎セル法との損益分岐を測定。同じ有向 edge / PBC image を生成する [S5,S7]。 |
| A03 | 中 | cell stencil と幾何学的下限距離による候補除外 | `radius_graph_pbc.py`。真の cutoff を変えず、入り得ないセル組を除外。非直交セル・整数オーバーフロー・境界誤差を検証。 |
| A04 | 中 | Morton / cell 順並べ替えと安全な CSR 再利用 | gather/scatter の局所性を改善。原子 ID と force の逆置換、同距離 tie 順序を保持。topology generation ID と所有権を使う [S3,S8]。 |
| A05 | 高 | PBC の edge ごとの 3x3 cell 複製を削減 | `get_pbc_distances` の `repeat_interleave(cell, neighbors)` が対象。出力は必要な E x 3 offset のみ。cell 勾配・stress・二階微分を比較。 |
| A06 | 中 | 距離・基底・Wigner 幾何生成の融合 | 距離、Gaussian、envelope、回転の中間テンソルを削減。ゼロ距離、軸近傍、PBC、応力の連鎖律を維持 [S3]。 |
| A07 | 中 | Wigner の block-diagonal 表現を経路全体で維持 | UMA-S の lmax=2 では 9x9 dense への展開を避ける。異なる lmax/mmax は general fallback。既実装の permutation 除去とは別 [S3]。 |
| A08 | 高 | gather + Wigner + radial scale + conv1 pack 融合 | `ExecutionBackend.node_to_edge_wigner_permute` と edgewise conv1。大きい edge message を DRAM に書かず GEMM 入力へ。位置・cell backward も実装 [S2,S3]。 |
| A09 | 高 | conv2 unpack + 逆 Wigner + scatter 融合 | `permute_wigner_inv_edge_to_node`。edge 中間配列と kernel launch を削減。FP64 が必要な集約を維持し、二階微分経路を用意 [S2,S3,S9]。 |
| A10 | 高 | radial 第一層の元素依存部分を事前計算 | 線形性を使い距離部分と元素 embedding 部分を分離。重み・MoLE・元素 embedding の変更で table を無効化。FP32 丸め差を評価 [S3]。 |
| A11 | 中 | radial GEMM + LayerNorm + SiLU 融合 | 中間 E x H 配列の往復を削減。元の eps・dtype と微分式を保持。厳密 FP32 モードを明示 [S3]。 |
| A12 | 中 | radial 最終線形層を gather 側へ融合 | 大きい radial 出力の保存を避ける。レジスタ圧、spill、再計算量との損益を Xe2 で測定 [S3]。 |
| A13 | 中 | gating・residual 加算・初期化の融合 | pointwise launch と一時配列を削減。autotune の試行前に出力を復元し、二重加算を防ぐ [S3]。 |
| A14 | 高 | 既存 `umas_fast_pytorch` の XPU 評価 | SO2 block GEMM と `UnifiedRadialMLP` は既に存在する。新規発明ではなく、適用条件・数値比較・選択の実装。全層 radial 保持による VRAM 増も測る。 |
| A15 | 中 | CSR / segment 単位の階層集約 | 高次数 atom の atomic 競合を削減。feature と energy/stress それぞれの既存集約 dtype を維持。並列順序差を検証。 |
| A16 | 高 | 数値計算コアだけを `torch.compile` | 動的 graph 生成から分離し、線形層・回転・norm・gate を融合。force の backward と Hessian を別に検証し eager fallback を残す [S10]。 |
| A17 | 高 | atom/edge 数による shape bucket と padding | 再コンパイルを抑える。dummy atom/edge が MoLE、norm、集約、勾配へ影響しない mask が必須。現在の `edge_chunk_size` は UMA の実処理 padding として未接続。 |
| A18 | 中 | 固定形状・固定アドレスの XPUGraph replay | 小さい kernel の起動負担を削減。動的 edge と graph pool メモリが制約。2.15 の `make_graphed_callables` は高階微分非対応なので Hessian は別経路 [S11]。 |
| A19 | 高 | 最適化 / MD の状態と更新処理を GPU 常駐 | 既存 TorchSim 接続を活用し、残る変換・複製と ASE/NumPy 往復を減らす。座標更新・収束判定精度を維持する [S12]。 |
| A20 | 中 | pinned staging buffer 再利用・非同期二重バッファ | 転送と独立 batch の計算を重ねる。Stream/Event と入力所有権を管理し、転送中の buffer 上書きを防ぐ。 |
| A21 | 高 | NEB 像・AFIR 試行・有限差分変位の microbatch | atom 数だけでなく edge 数と VRAM で batch を決める。構造間の graph / routing / 集約を分離。差分幅・計算式は不変 [S13,S14]。 |
| A22 | 中 | GPU ごとに独立ジョブを実行 | 複数 GPU がある場合の全体処理量を改善。model parallel とは異なり、VRAM は自動合算されない。明示 device ordinal と checkpoint cache を分離。 |
| A23 | 高 | MoLE 統合重みの容量制限付き cache | model/version、重み世代、組成、charge、spin、task、device/dtype 等をキーにする。UMA-S-1.2 は組成比だけでなく絶対原子数が必要。 |
| A24 | 中 | workspace 容量再利用と断片化対策 | 可変 edge buffer を段階的に増やす。autograd に保存中の領域は上書きしない。XPU allocator で実装される設定だけを評価し hot loop で empty_cache しない。 |
| A25 | 高 | edge chunk の streaming 集約と選択的再計算 | `escn_md_block.py` の chunk 出力 stack を減らす。checkpoint は主にメモリ施策であり追加計算が発生する。速度改善は batch 拡大時も含めて測る。 |
| A26 | 高 | 要求された energy / force / stress だけ計算 | checkpoint の head 設定とは別に API の要求出力を伝える。要求した出力の微分を維持し、不要な cell 微分等を省く。 |
| A27 | 高 | 全 Hessian の行ブロック化 | `outputs.py` の全 3N 単位行列と一括 vmap のピークを抑える。全 Hessian 出力自体の O(N^2) 保存量は残る。 |
| A28 | 高 | 自動微分による Hessian-vector product | H*v が必要な用途では全 Hessian を作らない。forward AD 非対応時は reverse-over-reverse。全スペクトルの代替ではない [S15]。 |
| A29 | 中 | Xe2 専用 tile / subgroup / register 調整 | CUDA warp 前提を移植せず、oneDNN と比較しつつ work-group、SLM、spill を測る。autotune cache は GPU・driver・dtype・shape を区別。 |
| A30 | 低 | kernel 内 index の安全な int32 化 | 原子数・辺数・stride・offset の最大範囲を確認し、超過時 int64 fallback。浮動小数点精度を変えず index 帯域を減らす。 |
| A31 | 高 | 探索 grid を原点からの絶対距離に依存させない | `radius_graph_pbc.py` の絶対座標による grid 粗化を見直す。探索座標だけを局所原点へ移し、微分する物理座標は維持。巨大な平行移動と tie 順を試験。 |
| A32 | 中 | 必要な境界原子だけ周期画像を生成 | 全原子を全 image へ複製する量を減らす。非直交・小セルの複数画像を含め、cutoff 内候補を完全に列挙する。 |
| A33 | 中 | 整数 stencil / PBC image index の cache | device/index、繰返し範囲、stencil 形状で immutable table を再利用。cell 由来の浮動小数点 offset は毎回更新し、共有 table を in-place 変更しない。 |

A01 ではセル、PBC、原子数・種類、cutoff、近傍設定変更時に再構築する。
初期実装ではセルが変形したら必ず無効化する。周期境界で wrap した位置の差を
そのまま変位判定に使わず、保守的に再構築するか正しい変位追跡を使う。
connectivity が同じでも距離・Wigner・radial・force は再計算する。
小箱で複数の周期画像が cutoff 内に入る場合も元の graph と同じ画像を列挙する。

## ② 精度・モデル・計算条件の交換を伴う 17 候補

| ID | 優先 | 候補 | 高速化の狙いと採用条件 |
| --- | --- | --- | --- |
| B01 | 中 | GEMM 限定 BF16 + FP32 蓄積 | XMX 利用を狙う。座標・幾何・norm・重要集約は元精度。B70 issue #1001 の形状試験を先行。 |
| B02 | 中 | GEMM 限定 FP16 + FP32 蓄積 | BF16 より細かい仮数と狭い数値範囲の交換。activation/gradient の overflow と underflow を確認。 |
| B03 | 研究 | oneDNN の TF32 許可モード | FP32 tensor の内部演算精度を緩める。B70 で実際に選ばれる kernel と速度を確認。CUDA 用設定の単純転用は不可 [S16]。 |
| B04 | 研究 | BF16x2 / BF16x3 の分割 GEMM | XMX で FP32 に近い精度を目指す。IEEE FP32 と同一の保証ではない。公開 Max1550 測定を B70 へ外挿しない [S17]。 |
| B05 | 中 | 中間 feature 保存のみ BF16/FP16 | VRAM と転送量を削減し計算時に FP32 へ戻す。cast 負担、微小 force、Hessian への量子化誤差を評価。 |
| B06 | 低 | weight-only INT8 / INT4 | 重み帯域・容量削減。座標から energy/force への微分を維持する kernel が必要。LLM 用 quantizer の直結は不可。 |
| B07 | 低 | FP64 集約を補償 FP32 集約へ変更 | FP64 が律速と実測された場合のみ。Kahan 等でも FP64 と同一とは限らず、相殺の強い energy/stress を検証。 |
| B08 | 低 | 局所 fast math | exp / rsqrt / trig の近似。元関数との値・一階・二階微分誤差を検証し、global fast-math は既定にしない。 |
| B09 | 研究 | 対象化学系専用 student への蒸留 | 小型 MLIP で energy・force・stress を再現。TS、短距離、非平衡構造を学習へ含め、必要なら曲率も評価 [S18]。 |
| B10 | 研究 | 小 cutoff / 近傍数 / lmax / channel の再学習 | edge と演算量を減らすモデル変更。既存 checkpoint の設定だけを削る方法は採用しない。 |
| B11 | 研究 | 小型モデル + UMA 補正の multiple time stepping | 高価な補正 force の評価頻度を下げる。差分 force の時間スケール分離、共鳴・NVE drift を確認。MD 用であり AFIR にそのまま適用しない [S19]。 |
| B12 | 中 | 粗いモデル / 精度で初期探索、FP32 で最終精密化 | 探索全体のコスト削減。最終 residual が合っても、別の極小・TS・経路へ進む可能性は残る。 |
| B13 | 中 | HVP + 選択固有モード / 近似 Hessian | Lanczos や準 Newton で必要な曲率だけ推定。A28 の HVP 自体とは別に、打切り・収束許容値・対象モードの制限を明記 [S15,S20]。 |
| B14 | 中 | NEB の像ごとに収束精度を配分 | DyNEB 型で鞍点付近を重点化。緩い像の閾値を使う場合は経路精度との交換。同一閾値で厳密再確認する省略方式は条件付き① [S13]。 |
| B15 | 低 | radial 関数の滑らかな補間 table | 基底 / MLP の反復計算を削減。値だけでなく一階・二階微分が滑らかである必要があり、最近傍 table 引きでは force/Hessian を壊す。 |
| B16 | 研究 | 統合済み Linear の低ランク圧縮 | 行列積を低ランク因子へ置換し再学習。MoLE は既に一つの Linear へ統合されるので、単なる expert 削除では定常推論の演算量は減らない。 |
| B17 | 低 | HVP による Hessian trace / diagonal の確率推定 | 統計量だけが必要な用途へ限定。分散・信頼区間・seed を管理し、全 Hessian、負固有値数、TS 検証の代用にしない。 |

調査したが初期採用しない方式: 変位検査なしの固定頻度近傍更新、無条件の
cutoff / max_neighbors 削減、非保守的な直接 force student、MD 刻み・質量の
変更。これらは計算対象や反応・動力学の解釈を変え得る。
GROMACS の統計的 energy-drift tolerance 方式は近傍の取りこぼしを許容するため
①の skin cache とは異なる [S21]。Xe2 の FP8/FP4 を native 高速演算と仮定しない。

## 既知問題と参照フォークの扱い

1. **B70 低精度 GEMM**: Intel compute-runtime #1001 は調査時点で Open /
   Needs Feedback。Ubuntu 24.04 の Proxmox CT、runtime 26.35/26.27、torch
   2.14/2.8 等で FP16/BF16 の形状依存 NaN/巨大値が報告されている。
   FP32 は正常との報告だが、全 B70 環境の確定不具合でも、2.15 で修正済みの
   証拠でもない [S22]。
2. **Windows 旧 Arc の同期**: PyTorch #197520 は #202188 による解決が報告される。
   修正は Windows の BMG より前の GPU の device-wide wait を避けるもので、
   Ubuntu B70 にそのまま当てはめない。v2.15.0 タグには当該 guard はない [S23]。
3. **Aurora PR #2182**: open / 未マージ。対象は主に PVC / Xe-HPC。
   `xpu:1` の index を失う正規化や、アクセラレータ存在だけで fused path を
   許可する変更を無条件に取り込まない。B70 / Xe2 の保証にはならない [S4]。
4. **FlashUMA**: 独立 repo の headline は CUDA、TF32 有効での比較。
   最大 2x 等を B70 の厳密 FP32 性能と扱わない。参考にするのは演算融合と
   データ配置。PR #2185 は closed / 未マージであり、そのブランチ全体が
   upstream に採用されたものとは扱わない [S3]。
5. **cache の所有権**: FlashUMA の data_ptr / version 等による CSR cache を
   そのまま転用しない。入力解放後のアドレス再利用、device ordinal、stride、
   topology 変更を区別する設計が必要。
6. **FP64**: Xe2 で FP64 が一切不可能とはしない。逆に PVC と同じ演算率とも
   仮定しない。FlashUMA の register 内 FP64 集約を使う場合は B70 での性能も測る。
7. **compile / graphs**: API が存在することと UMA の全微分経路が動くことを分ける。
   カスタム kernel は backward / double backward / vmap / forward AD を個別確認する。

PyTorch 2.15 の XPU oneDNN Matmul は `allowTF32OneDNN()` を参照する。
`torch.backends.mkldnn.allow_tf32` と CUDA 向けに説明される
`set_float32_matmul_precision("high")` を同じものとして扱わない。
`is_tf32_supported()` の true だけで B70 の実 GEMM 性能や演算精度を断定せず、
oneDNN verbose と数値試験で実行経路を確認する [S16]。

## CUDA との比較設計

SYCL という言語だけを理由とする劣後率は算出できない。現在の general 経路では
CUDA 専用融合 kernel より launch と DRAM 中間配列が多い部分がある一方、
oneDNN GEMM 自体は最適化ライブラリを利用する。比較を次の三段に分ける。

1. 同じ checkpoint、構造、cutoff、FP32 方針、要求出力で CUDA general と XPU general。
2. 同じ NVIDIA GPU で CUDA general と fused backend。これで専用最適化の効果を分離。
3. 同じ B70 で XPU general と各提案。これで改修そのものの利益を測る。

cold load / 初回 JIT / warm latency / throughput / allocated-reserved-peak memory を分離。
graph、geometry、radial、rotation/SO2、scatter、backward、転送の時間を記録する。
GPU 自体の演算・帯域差があるので 1 だけを「SYCL の overhead」と呼ばない。

## 今回確認・修正した正しさの問題

UMA-S-1.2 の composition embedding は学習時の仕様を維持するため、平均に
初期値のゼロを含む。したがって原子数 N に対して sum(embedding)/(N+1) となり、
H2O と H4O2 のような同じ組成比でも routing 係数が異なる。
従来の MoLE merge guard は組成比だけを比較していたため、誤った統合重みを
再利用する可能性があった。

`escn_moe.py` を修正し、当該モデルで composition embedding が有効な場合は、
初回の混合 batch と統合後の入力の両方で絶対元素数も比較する。
他のモデル世代、composition embedding 無効時の従来挙動は維持する。
checkpoint の学習済み計算式自体は変更しない。

新規 12 ケースを含む関連回帰試験は **56 passed / 1 skipped / 1 deselected**。
skip は実 XPU が必要なケース、deselected は選択条件外の GPU ケース。
Python 3.12.14 / PyTorch 2.15.0+cpu で実施し、pre-commit と diff check も成功。
B70 実測、全 pretrained checkpoint の一致、全体テストスイートの成功を意味しない。

## 実装順と検証条件

- 第1段: 正しさの guard、A05/A14/A26/A27/A28、A01 の CPU 参照実装と edge 一致。
- 第2段: workload に応じ A21（小分子多数）または A01（同じ構造の連続評価）、
  A16/A17、A19。
- 第3段: profile で律速の A08/A09/A10/A11 等だけを SYCL/Triton-XPU 化。
- 第4段: B01/B02 を独立オプションとして評価し、B09/B11 等は別の研究モード。

最低限の検証対象は energy/atom、相対 energy、force 最大・RMS 誤差、stress、
有限差分 force/stress、Hessian 対称性・HVP、並進/回転対称性、極小/TS の残差、
必要な用途では NVE drift と反応障壁。ゼロ原子/edge、混合 PBC、歪んだ小セル、
cutoff 境界、近傍同距離、複数 dataset/charge/spin、cache invalidation も含める。
許容誤差は「何倍速いか」より先に用途ごとに定義する。

## 一次資料

- S1: [Torch XPU Operators implementation](https://github.com/intel/torch-xpu-ops)
- S2: [fairchem Wigner/SO2 fusion PR #2073](https://github.com/facebookresearch/fairchem/pull/2073)
- S3: [FlashUMA](https://github.com/ArcelliEmanuele/flash-uma), [PR #2185](https://github.com/facebookresearch/fairchem/pull/2185)
- S4: [Aurora XPU PR #2182](https://github.com/facebookresearch/fairchem/pull/2182), [Rootstock consumer](https://github.com/Garden-AI/rootstock/blob/main/sample_model_configurations/aurora_configs/uma.py)
- S5: [LAMMPS neighbor](https://docs.lammps.org/neighbor.html), [neigh_modify](https://docs.lammps.org/stable/neigh_modify.html)
- S6: [HOOMD neighbor lists and buffer](https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/module-nlist.html)
- S7: [HOOMD BVH Tree](https://hoomd-blue.readthedocs.io/en/v7.1.2/hoomd/md/nlist/tree.html)
- S8: [LAMMPS atom sorting](https://docs.lammps.org/atom_modify.html)
- S9: [OpenEquivariance: CUDA/HIP, not a drop-in Intel backend](https://github.com/PASSIONLab/OpenEquivariance/blob/main/openequivariance/README.md)
- S10: [PyTorch 2.15 Intel GPU support](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html)
- S11: [PyTorch 2.15 XPU graph implementation](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/xpu/graphs.py)
- S12: [GROMACS GPU-resident execution](https://www.gromacs.org/topic/heterogeneous_parallelization.html)
- S13: [ASE NEB / DyNEB](https://docs.ase-lib.org/ase/neb.html)
- S14: [ASE vibration displacement implementation](https://docs.ase-lib.org/_modules/ase/vibrations/vibrations.html)
- S15: [PyTorch Jacobian / Hessian / HVP tutorial](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html)
- S16: [XPU oneDNN Matmul.cpp, v2.15.0](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mkldnn/xpu/detail/Matmul.cpp), [Context](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/Context.cpp)
- S17: [Intel BF16x2 / BF16x3 GEMM](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/boost-matrix-multiplication-performance-with-intel.html)
- S18: [MLIP distillation research](https://arxiv.org/abs/2502.05379)
- S19: [OpenMM MTSIntegrator](https://docs.openmm.org/latest/api-python/generated/openmm.mtsintegrator.MTSIntegrator.html), [MLIP time-scale learning](https://arxiv.org/abs/2310.13756)
- S20: [SciPy eigsh / LinearOperator algorithm reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
- S21: [GROMACS buffered pair lists](https://manual.gromacs.org/documentation/2026.0/reference-manual/algorithms/molecular-dynamics.html)
- S22: [B70 low-precision GEMM issue #1001](https://github.com/intel/compute-runtime/issues/1001)
- S23: [Windows pre-BMG synchronization issue](https://github.com/pytorch/pytorch/issues/197520), [fix #202188](https://github.com/pytorch/pytorch/pull/202188)
