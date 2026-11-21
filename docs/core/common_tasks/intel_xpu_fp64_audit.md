# FP64対応・共通最適化・upstream再調査

2026年11月10日（日本時間）。開始点は`intel-xpu-fp32`の
`0b30a9fdfab64a837bdd798001e7b64ec3399f33`。
参照したupstream mainは`36b17c6dca30fd95f0f154837ab297191974a4c1`。
未マージPRは下記の確認時点の実装を参照し、upstream全体のmergeは行っていない。
導入・実行コマンドは[README](../../../README.md#fork-fp64)にまとめた。

## 現状のFP64最適化と対象環境

近傍再利用、MoLE統合、folded GEMM、portable radial／Wigner、出力選択、
バッチ転送・分割など、既存A系列の多くはdtype共通であり、FP64にも適用される。
同じ処理をFP64専用に複製する必要はなかった。
一方、開始点にはXPU FP64の一律拒否と、入力・内部係数をFP32に丸める経路が残っていた。
今回これらを修正し、FP64の実演算検査と明示presetを追加した。

| 機能 | FP64での扱い |
| --- | --- |
| MoLE統合・組成cache | 共通実装。未統合／統合／cache再利用の小型UMA比較を追加 |
| Verlet近傍再利用 | 共通実装。cutoffと近傍上限を毎回評価し、曖昧な境界は再構築 |
| folded GEMM・radial／Wignerのportable演算 | dtypeを保つ既存実装。FP64の順伝播・一次／二次微分を回帰検査 |
| 出力選択・バッチ分割・転送 | 共通実装。入力と返却値の所有権を維持 |
| activation checkpointing | 精度低下を追加せず保存量を減らす。再計算分の時間が増える場合がある |
| FP32専用融合カーネル | FP64には適用せず、dtypeを保つ通常演算を使用 |
| compile／XPU Graph | 既存の任意設定。新FP64 presetでは無効。実機での受入試験は別途必要 |
| UMA Nitro | CUDA専用で、FP64／XPUには使用しない |

新しい`fp64`と`xpu-fp64`は同じ設定で、general backend、eager、MoLE統合を使用する。
TF32、Nitro、B系列は既定で無効。従来の`high-precision`はFP32であり、FP64を意味しない。
FP32 checkpointのFP64化は、以後の丸め誤差を抑えるための計算精度指定である。
元の重みの桁数や学習モデルの物理的精度が回復するわけではない。

Intel Compute Runtimeの[BMG能力定義](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/xe2_hpg_core/hw_info_bmg.cpp)
には`ftrSupportsFP64=true`、`ftrSupportsFP64Emulation=false`があり、
[BMG G31のdevice ID定義](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/xe2_hpg_core/bmg/definitions/device_ids_configs_bmg.h)
には[IntelのB70製品仕様](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html)
に記載された`0xE223`が含まれる。
ただし、この宣言はPyTorchの全演算の成功やFP64演算速度を保証しない。
[PyTorch 2.15のXPU属性](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/xpu/__init__.py)
の`has_fp64`を記録し、実際の演算結果を検査する方式にした。
FP64／FP32の速度比やB70のFP64 TFLOPSは、今回の検証からは示せない。

## 段階的に反映した改修

Aは低精度近似を追加しない変更、Bは誤差を伴う任意設定を表す。
Aでも演算順序や以前の不正な丸めを直すことで、旧版との最終bitは変わり得る。

| 順序 | 発見事項・条件 | 実装と確認 |
| --- | --- | --- |
| A1 | XPUではFP64を一律拒否し、対応環境でも実行できない | GEMM、index_add、幾何演算、一次／二次微分をCPU FP64と比較する事前probeを追加。dtype、device、FP32より細かい桁の保持を確認。失敗時の低精度化・CPU fallbackはしない |
| A2 | 事前probeをconstructorだけに置くと、公開factoryが先に大きなcheckpointを取得する | 明示的XPUの`get_predict_unit`／`FAIRChemCalculator.from_model_checkpoint`で取得前に検査。`device=None`の既存自動選択は維持し、constructor側で検査 |
| A3 | 線形charge/spin embeddingが`.float()`を使い、FP64重みと不一致。spin sentinel代入が呼出元tensorを変更する | 重みdtypeへ変換し、sentinelはout-of-placeで生成。入力不変、FP32／64の一次・二次微分を検査 |
| A4 | 応力の球面成分・Cartesian変換で係数や出力が既定FP32になり、FP64を失う | 入力dtype/deviceで係数を生成し、成分の共通dtypeで復元。FP64往復・微分を検査 |
| A5 | NVIDIA外部近傍adapterがASEの座標・cellをFP32化し、FP64ではcutoff境界の選択を変える | `AtomicData`のtarget dtypeをadapterへ伝達。実CPU nvalchemiでFP64境界を確認。直接呼出しの既定値はFP32を維持 |
| A6 | TorchSimのdtype省略がFP32固定で、FP64 predictorに渡す前に座標を丸める | 省略時はpredictor精度を継承。明示dtypeはシミュレーション側の指定として維持。`False`／`0`はロード前に拒否 |
| A7 | 最小イメージ補助関数が微小なcellせん断をゼロ扱いする | 対角分岐は非対角成分が正確にゼロのときに限定。旧版で約`5e-5 Å`ずれた例をASEと比較。この補助関数は現在のUMA主要経路では未使用 |
| A8 | EdgeDegreeEmbeddingのcheckpointがchunkごとに大きなnode入力を保持する | upstream #2219を移植。共有zero入力を使い、chunk出力をcheckpointの外で加算。FP32／64の出力・勾配・二次微分を検査 |
| A9 | エネルギー微分で力を求める推論の返却tensorを保存すると、計算グラフも残る | upstream #2136を移植。要求されたHessian/HVP等を含む後処理が完了してからdetach。学習、direct-force、raw modelの勾配経路は維持 |
| A10 | 近傍cacheの再利用時、距離と丸め誤差の評価で同じ端点を重複gatherする | 独立したgather bufferを再利用して割当てを削減。入力・autograd保存値は変更せず、cutoff／近傍上限の判定式を維持 |
| A11 | positional spinのゼロ判定が可変長nonzeroを生成する | 固定形状の`where`へ変更。動的索引を除去。CPU FP64では速度改善を確認していない |
| A12 | 設定・registryだけのimportでPyTorch／計算／server実装をまとめて読み込む | upstream #2286を参考に`core`／`calculate`の公開APIだけを遅延import化。公開名、型情報、`dir()`、star import、pickle同一性を検査 |
| A13 | 検証・MD・B比較のツールがFP32前提で、FP64の評価を揃えにくい | dtype指定、モデル取得なしのprobe、同一基底精度でのB比較、実効dtype記録を追加 |
| A14 | 有限差分テストの変位・歪みにASE拘束が再適用され、検査が誤る。CPU参照probeが外部default deviceに影響される | 検査用コピーの変位・復元は`apply_constraint=False`、CPU参照tensorは明示配置。原本の拘束・座標を維持 |
| B1 | FP64の微分用保存特徴が大きい | `SeriesBSettings(saved_feature_dtype="fp32")`をFP64専用の任意設定として追加。前向きはFP64、保存対象の学習済み特徴だけをFP32へ圧縮。力・stress・Hessianは近似 |

A4により、内部でFP64 reductionを行うdirect-stress headの返却値は、FP32モデルでも
FP64を保持する場合がある。旧版の暗黙のFP32化を直した結果であり、混合dtypeの入力による障害ではない。
A9の`predict`は完了済みの推論結果を返すAPIとして扱う。
独自の損失や追加微分を構築する用途では、raw model APIを使用する。
A12は起動時の不要な読み込みを減らす変更であり、calculatorを使う際のPyTorch・Ray Serve読み込みは残る。

## FP64でのB設定の境界

FP64で許可するB設定は`SeriesBSettings(saved_feature_dtype="fp32")`だけである。
適用範囲はparameterを持たない選択されたgate／SiLU／sigmoidの保存特徴で、
重み、座標、正規化、前向きのGEMMを低精度化しない。
前向きエネルギーが同じでも、丸めた中間値を使う力は、そのエネルギーの厳密な微分と
一致しない場合がある。NVEでのエネルギー保存や長時間の軌道への影響は別途検証する。
`gemm_precision="fp32"`という既存の既定値は「GEMM近似を追加しない」指定であり、
このFP64設定のGEMMをFP32へ変える意味ではない。

既定ではBは無効。FP32 baseへの`fp32`保存指定、FP64 baseへのBF16／FP16・量子化・
低ランク・radial近似・TF32等の併用は拒否する。
activation checkpointing、compile、XPU Graph、Nitroとの併用も拒否する。
`torch.func/vmap`内部でのforwardには保存tensor hookの制約があるため、明示的なエラーを維持する。
設定を切り替えるときは別predictorを作り、準備済みinstanceを書き換えない。

圧縮scale計算も、独立した絶対値bufferをin-placeで処理して一時配列を削減した。
有限値・NaN・Infの扱いは維持する。元tensorのin-place変更検出と所有権の検査も維持した。
省メモリ量は圧縮対象に限られ、pack/unpackの一時領域も必要になる。
モデル全体のpeak VRAMが半分になることや、高速化することを意味しない。

## upstreamのissue・PRとの照合

確認時点の関連提案から、今回の実行経路へ適用できるものを選別した。

| 参照 | 判断・反映内容 |
| --- | --- |
| [#2219](https://github.com/facebookresearch/fairchem/pull/2219)（merged、`d9716429092b4339dded8fd8e64fd3fdec261204`） | checkpointでのnode buffer共有を反映。FP32にも適用 |
| [#2136](https://github.com/facebookresearch/fairchem/pull/2136)（merged、`d6b387873eba7cee3cbdeecadfda872847f33d08`）／[#2072](https://github.com/facebookresearch/fairchem/issues/2072) | 推論結果のグラフ保持対策を反映。Hessian/HVPを計算した後にdetachするよう既存機能と統合 |
| [#2286](https://github.com/facebookresearch/fairchem/pull/2286)（open、`4e7c10bdd49b3e2f55ba634e0b04096592c28ef0`） | 公開package入口の遅延importだけを採用。predict、worker、学習コードの広範なimport変更は行わない |
| [#2296](https://github.com/facebookresearch/fairchem/pull/2296)（open、`13a21cbcd4946dd78577c1ad3de5bc174b304a6e`） | FP64基準の数値検証設計を参考に、E/F/stress、Hessian、MDの評価を整備。大規模提案全体は取り込まない。FP64との数値一致とDFT精度は区別 |
| [#2264](https://github.com/facebookresearch/fairchem/pull/2264)、[#2212](https://github.com/facebookresearch/fairchem/pull/2212)、[#2270](https://github.com/facebookresearch/fairchem/pull/2270) | folded linear、formation cache、非連続reductionは既存フォーク側の対応を再確認。重複実装は追加しない |
| [#2263](https://github.com/facebookresearch/fairchem/pull/2263)、[#2259](https://github.com/facebookresearch/fairchem/pull/2259)、[#2291](https://github.com/facebookresearch/fairchem/pull/2291) | ASE array-like／metadata更新は既存対応を維持。固定入力の追加仮定はIntel単GPUの既定値に採用しない |
| [#2125](https://github.com/facebookresearch/fairchem/pull/2125)（closed、unmerged） | `compute_energy`のcompiler barrier削除は、大規模／複数GPUの数値確認が不足。既存barrierを維持 |
| [#2118](https://github.com/facebookresearch/fairchem/pull/2118) | graph接続の変更は計算内容を変えるため、精度を維持するAとして導入しない |
| [#2203](https://github.com/facebookresearch/fairchem/pull/2203) | 学習lossのhost同期削減であり、今回の推論MD経路の改善対象外 |

## CPUで測定した効果と限界

測定は共有CPU環境で実施した。B70／CUDAのthroughputやVRAM測定ではない。
生の測定値と条件は[測定JSON](intel_xpu_fp64_measurements.json)に保存した。

| 対象・指標 | 変更前 | 変更後 | 解釈 |
| --- | ---: | ---: | --- |
| checkpoint保持node入力、FP64 | 72 MiB、16 buffer | 4.5 MiB、1 buffer | 4096 node、32768 edge、16 chunk。対象入力の保持量であり全体peakではない |
| 同FP32 | 36 MiB | 2.25 MiB | dtype共通の改善 |
| Verlet再利用の正のCPU割当て累計、FP64 | 64.21 MB | 43.57 MB | 4096原子、143360候補辺。PyTorch profilerの割当てイベント累計で、peak／RSS／VRAMではない |
| 同FP32 | 35.25 MB | 24.92 MB | 割当て削減を確認 |
| Verlet再利用のFP64中央値 | 21.46 ms | 18.46 ms | 2 threads、10回交互測定。この部品・条件に限る |
| 同FP32中央値 | 18.77 ms | 20.08 ms | 安定した速度改善を主張しない |
| `import fairchem.core`の新規process max RSS | 約545–548千KiB | 14080 KiB | Pythonを含むプロセスRSS。calculator利用時の重量依存は残る |
| 同import時間、3回 | 6.22–10.22 s | 0.075–0.149 s | OS cache未消去・共有負荷あり。MDステップの改善率ではない |

positional spinの`aten::nonzero`は1回から0回になったが、FP64のCPU測定では
同等または遅くなっており、速度向上とは扱わない。
推論結果のdetachは、出力を保存した際の計算グラフ参照を除く対策である。
長時間のGPU MDでメモリリークがなくなったと実証したものではない。

B1は1,048,576要素の保存特徴で8,388,608 bytesから
4,194,304 bytes＋scale 8 bytesへ減少した。CPUでpack約9.80 ms、unpack約11.49 ms。
固定seed 41、3原子・1層・4 channelのランダム小型UMAでは、
E/F/stressの中央値は11.365→11.496 msで、高速化は認めなかった。
最大絶対差はenergy 0、force `2.89e-11`、stress `4.68e-13`、Hessian `2.21e-10`だった。
これは当該試験の値であり、配布モデルや大規模MDでの誤差上限ではない。
測定のB閾値は全対象を通す`min_elements=1`で、既定閾値1024とは異なる。

## 検証環境・再実行方法

Python 3.12.14、PyTorch 2.15.0+cpu、NumPy 2.4.6、ASE 3.29.0、e3nn 0.6.0、
TorchSim 0.6.2、Ray 2.60.0、Hydra 1.3.8、OmegaConf 2.3.1、SciPy 1.18.1、
nvalchemi-toolkit-ops 0.5.0、Warp 1.19.0、Triton 3.9.0。
`uv pip check`で240 packageの依存整合を確認した。
CUDA／XPU実機は利用できず、CPUの実計算とGPU制御経路の模擬試験を区別した。

FP64 probeはCPUでGEMM、index_add、幾何演算、一次・二次微分を通過した。
小型ランダムUMAで、E/F/stress/Hessian/HVP、TorchSim、MoLE統合と近傍再利用、
学習時の勾配保持、保存特徴圧縮、有限差分、短いMDを検査した。
実モデルのweights取得、GPU実演算、長時間NVE／NPTの物理的検証は未実施である。
`validate_xpu --dtype float64`は演算精度を切り替えるが、有限差分の刻み・許容誤差の
既定値は従来設定を維持する。用途に合わせて引数で調整し、FP64指定だけでより厳しい
精度判定を行ったと解釈しない。B比較ツールの成功も、実行・有限値の確認を意味し、
対象用途での精度合格ではない（出力の`accuracy_qualified=false`）。

最終統合回帰は**1,864 passed、47 skipped、0 failed、0 errors**
（4 workers、743.77秒）。対象は[61 pathの一覧](intel_xpu_fp64_test_paths.txt)。
個別試験の件数は重複加算していない。skipはCUDAが必要な37件、XPUが必要な9件、
Nitro対応GPUが必要な1件だった。
試験開始時の38ソース・テストファイルのSHA-256を保存し、終了後に不変であることを確認した。
変更42ファイルのpre-commit全hookと`git diff --check`を確認し、
READMEの元本文保持、追加目次・ローカルリンク、20コード例の構文も検査した。

```bash
mapfile -t test_paths < docs/core/common_tasks/intel_xpu_fp64_test_paths.txt
CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  "${test_paths[@]}" -c packages/fairchem-core/pyproject.toml -o addopts='' \
  -q -n4 -m 'not pretrained' --disable-warnings --tb=short -r s \
  -k 'not task_initialization and not homogeneous_batching and not stress_computation and not empty_batch_error and not charge_spin_handling and not model_output_validation and not invalid_task_name_raises_error and not test_2_atoms_2_ranks_gloo_a2a_collect'
```

除外したTorchSimの旧分散fixture依存ケース、gated pretrained model、実Gloo collectは
成功件数に含めない。Glooの複数process実行は前回、環境のsocket権限制限で検査できなかった。
今回の変更に含まれる実GPU試験も、CPU環境ではskipとして記録する。
FP64の実運用開始時には、READMEのprobe、対象checkpointでの数値比較、
対象サイズでのMD時間・peak VRAM・安定性を順に確認する。
