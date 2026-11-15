# A/B系列の併用・メモリ再監査

2026年11月15日。対象は`intel-xpu-fp32`、開始点は
`8472d8f90d717af08c58ca726e79d3d077d801f7`。
前回の[FP64監査](intel_xpu_fp64_audit.md)後の実装を対象に、
A01–A33、B01–B17とFP64保存特徴圧縮の実行経路・併用制約を点検した。
既存機能を有効にする既定値や、Bを明示指定する方針は変更していない。

## 確認範囲と判断

CPUの実演算、制御経路の模擬試験、実GPUが必要な試験を分けて扱う。
今回の試験で、すべての入力・組合せ・ドライバーで停止や不安定化が起きないことを
保証するものではない。特にB70実機、CUDA Nitro、XPU Graph／Triton-XPUの
ネイティブ実行は、この環境では確認できない。

| 機能群 | 今回の点検・試験 |
| --- | --- |
| A01–A05：近傍再利用、探索法、順序、cell offset | FP32／64、cell／auto／all-pairs／BVH、PBC・cell・組成変更、cap境界、所有権。追加探索試験は864回のcached／fresh比較 |
| A06–A15：geometry／Wigner／radial／SO2／reduction | 実PyTorchの通常・portable演算、一次・二次微分、外側の精度設定、部分的に凍結したparameter、state reload |
| A16–A18：compile、padding、XPU Graph | paddingを実UMAの併用試験へ追加。compile／graphの失敗・復旧・所有権は既存模擬試験も実行。実XPU captureは未検証 |
| A19–A24：TorchSim、転送、microbatch、device pool、MoLE cache | 連続ジョブ、並行呼出し、設定・出力の独立性、lease解放、cache切替、失敗後の再実行 |
| A25–A28：checkpoint、出力選択、Hessian／HVP | 実小型UMAの微分と要求切替。HessianとHVPの積の一致、返却済み出力の不変性 |
| A29–A33：XPU kernel選択、index、graph最適化 | CPU参照と既存dispatch／境界試験。実機のtile tuning・性能認定は未実施 |
| B01／02／04：BF16・FP16・split GEMM | CPUの明示emulationで実モデルを検査。Intel XMXや実GPUの蓄積精度・性能を再現する試験ではない |
| B03：oneDNN TF32 | 設定の範囲・復元・禁止組合せ・cache keyの模擬試験。ネイティブXPUは未検証 |
| B05／06／07／08／15／16：保存特徴、量子化、集約、radial近似、低ランク | 単独と併用、FP32／対応するFP64、微分、copy／pickle、hook・state load、cache切替 |
| B09／10：教師ラベル・蒸留 | 実PyTorch学習、所有コピー、逐次データ収集、入力勾配、export、失敗時解放。実用studentの新規学習ではない |
| B11／12／14：RESPA、粗探索→精密化、DyNEB | ASEで積分・最適化・拘束・収束再判定・段階切替と例外処理を検査 |
| B13／17：Lanczos、Hutchinson | 実HVP、残差・標本統計、dtype、基底の寿命、例外を保持した場合の解放 |
| UMA Nitro | 既存のfallback、直列化、stream／capture、wrapper制約の回帰。今回Nitro専用コードは変更していない |

実モデル併用の追加試験は、次の10軸の値について全二因子組合せを含む
**23通りのcovering array**を使った。373組の値ペアを覆うが、
23,040通りの直積全体や三因子以上の全組合せを試したわけではない。

| 軸 | 検査値 |
| --- | --- |
| GEMM | FP32、BF16、FP16、BF16 split2／split3 |
| 重み | 変更なし、INT4、INT8、低ランク0.5 |
| 保存特徴 | 変更なし、BF16、FP16 |
| compensated reduction | 無効／有効 |
| radial | 変更なし、fast exp、65点table |
| graph search | cell、all_pairs、bvh、auto |
| edge order | original／spatial |
| neighbor skin | 0／0.4 Å |
| edge padding | なし／8 |
| MoLE cache | 0／2 entries |

各ケースで実小型UMAのE/F/stress/Hessian、HVPとの整合、入力不変、移動後の再利用、
外側autocastからの分離、設定のHydra往復とModelSpecを確認した。
MoLE cache有効時には電荷を変更した後の再利用も確認した。
比較相手には同じB設定を使い、A側の探索・順序・padding変更の整合を検査している。
Bの科学的精度を一律の閾値で認定する試験ではない。
さらに13種類のB設定×5種類の非対応backend設定、計65通りの拒否を確認した。
covering arrayの網羅確認を含め、追加ファイルは89試験である。

ASE／TorchSimにも、INT8＋保存BF16＋補償集約＋radial table、split2＋保存FP16＋fast exp、
低ランク＋保存BF16、FP64＋保存FP32の4組合せを追加した。
いずれもMoLE／近傍cache等を併用し、wrapperを経由しない同条件の結果と比較した。

## 発見事項と修正

| 発見した条件・影響 | 反映した修正 |
| --- | --- |
| CPU／CUDAで外側のautocastが厳密推論へ混入し、意図せず低精度化する | 推論中だけautocastを無効化。明示的な内部AMP・B演算・raw trainingは維持し、外側の状態を復元 |
| UnifiedRadialMLPがautocast等の異なる精度条件で作った元素別tableを再利用する | autocast、FP32 matmul、XPU oneDNN TF32の条件をcache keyへ追加。compile／meta経路の互換性も確認 |
| 親module経由のstate reload後、古いradial tableと置換前の重み参照が残る | 再帰ロードhookで直ちに解放 |
| モデル準備の失敗後、部分構築モデルやruntime循環参照を保持したまま交換モデルをロードする | 失敗モデルのリソースと参照を解放し、循環参照を回収してから再構築 |
| 準備済みの完全CPU入力コピーがB変換・device uploadに重なる | 準備完了直後にローカル参照を解放。元入力の完全clone境界は維持 |
| `AtomicData`のclone補助関数が自己参照closureを持ち、tensor memoを循環GCまで保持する | finallyでclosureの自己参照を解消。利用者が渡したmemo、alias、back-referenceは保持 |
| `AtomicData.apply`が元のcontainerを順次置換する間にPython IDが再利用され、後続dataclassのdeepcopy memoと衝突し得る | 変換中だけ元のtop-level fieldを保持。slotted dataclassを含むmetadataを保全 |
| stagingを`inference_mode()`内で初期化すると、後続の通常呼出しでbufferへのin-place更新が失敗する | 再利用bufferの生成・copy時だけinference modeを解除。入力勾配を受け付けない既存契約と外側の状態は維持 |
| microbatch／serverが前バッチの大きな出力・入力を次の計算まで保持する | 所有出力への分割後に元出力を解放。OOM時も失敗batchをallocator cleanup前に解放 |
| multiplexed serverが同じrouteの先頭ModelSpecしか検証せず、後続requestに別設定が混在し得る | 全requestのspecとroute IDを実行前に照合。不一致ならモデル実行前に拒否 |
| serverのloop-local参照がLRUから消えた旧predictorを次モデルのロード中も保持し得る | group終了時に不要なownerを解放。cache自体が保持する正当な参照は維持 |
| 外部serialization／load hookや個別compileを持つGaussianをradial近似へ置換すると、保存キーや呼出経路を変える | 置換前に非対応状態を検出して拒否。失敗時に元module・checkpoint契約を保持 |
| 保存特徴の復元で複数の全精度中間配列を作る | 新しい所有buffer内でscale・clamp。元のNaN／Inf、一次・二次微分を維持 |
| split GEMMの積和ごとに新しい出力を確保する | 最初の所有GEMM出力へ同じ順序で加算。共有入力・weightは変更しない |
| 凍結したLayerNorm／SiLU重み・biasにも不要な勾配を作る | 必要な入力の勾配だけを計算。部分凍結・学習・二次微分も検査 |
| portable SO2が同じradial係数を繰り返し配列化する | broadcastへ置換。通常・lazy projection・adjointで検証 |
| graph探索の大きなgrid／AABB／candidate作業配列が後段まで残る | 最終使用直後に参照を解放。判定式・cutoff・cap・辺の選択は変更しない |
| 教師データ収集で中間sample一覧と所有コピーを同時に保持する | sample単位で逐次消費し、GPU出力・中間CPUコピーを順に解放。返却データの防御的コピーは維持 |
| 蒸留の前ステップ出力・勾配、exportの検証専用モデルが次の確保に重なる | optimizer更新後にgraph・勾配を解放し、export検証モデルをEMAコピー前に解放 |
| Lanczosの全基底が残差HVPまで残り、粗探索optimizerのHessian/historyが次段階まで残る | 独立したRitz vector生成後に基底を解放。完了optimizerをreference段階の構築前に解放 |
| Bの失敗例外をrunnerが保存すると、traceback経由で私有model・optimizer・作業配列が残る | 既存の例外cleanupを蒸留・曲率・比較にも適用。例外型・位置を保ち、実行中caller／generatorと利用者所有modelは閉じない |
| RESPA内側stepの有限値確認が不要な全momentumコピーを作る | 検査だけの経路はコピーせず、返却値等に必要な所有コピーは維持 |

外側BF16 autocastの混入は、修正前の小型UMAでenergy最大差`2.24e-4`、
forces最大差`4.59e-5`として再現した。
radial cacheでも、BF16 warmup後のFP32出力がfresh FP32から最大約`1.84e-3`ずれた。
今回の修正は、異なるlibraryの精度設定が厳密推論へ持ち越される経路を除くものである。

測定で利点を確認できない変更は取り込まなかった。
parameter／buffer走査を`itertools.chain`へ変更する案は、Python allocation peakが
751,500→786,500 bytesへ増えたため撤回した。

FP64のB比較では、ランダムな回転角を毎forward生成する既存UMA経路により、
A/Aの独立実行でもenergyに`2.78e-17`の差が生じた。
比較テスト内だけで乱数列を揃えるとA/Bのenergy差は正確に0だったため、
本体の計算や許容誤差を変更せず、試験の乱数条件を揃えた。
最初の全体試験でも、別のFP64 predictor比較に同じ問題があり、energy差
`4.16e-17`で厳密一致検査が失敗した。ここも比較する2回のCPU乱数状態だけを
揃え、外側の乱数状態と`rtol=atol=0`を維持した。
同じベクトルのHVPを5回計算した小型UMAでは、反復間の最大絶対差をHVPの
最大絶対値で割った相対変動がA32で約`5.67e-7`、
BF16 GEMMのCPU emulationでは約`2.39e-5`だった。
確認した6条件はLanczosの既存対称性・残差判定を通過したが、Bの微分が
すべての系で同程度に安定するという保証ではない。既存の残差検査・未収束報告は維持する。

最初の全体試験では、checkpointの新旧加算経路を比較する既存テストも、
FP32二次微分の成分差`6.87e-5`で失敗した。このテストは乱数seedを固定していなかった。
失敗を再現するseed 2、13、15を調べ、同じFP32入力・重みをFP64へ上げた参照と比較すると、
失敗成分を含むtensorでは、新旧両経路のFP32最大誤差は約`9.0e-5`–`3.3e-4`、
対応するtensorの最大絶対値は224–785だった。
FP64同士の二次微分差は約`2.0e-13`–`4.6e-13`であり、加算順序の丸め差と判断した。
この再現seedを回帰試験に残し、値・一次微分とFP64同士の比較を維持しつつ、
FP32二次微分は両経路を共通のFP64参照に照合する。
参照tensorの最大絶対値（1未満なら1）で規格化した最大誤差は、実測でFP32のmachine epsilonの
約5.1倍以内だった。[回帰試験](../../../tests/core/models/uma/nn/test_embedding.py)の予算は16倍とした。
任意のnetworkの条件数に対する誤差保証ではなく、今回の再現条件での数値回帰検査である。
本体のcheckpoint計算は変更していない。

## メモリ・処理削減の実測

以下はPyTorch 2.15.0+cpu、共有CPU環境での部品測定。
**GPU全体のVRAM、プロセス全体のRAM、MD全体の速度ではない。**
指標が異なる値は合算できない。生値・条件は[測定JSON](intel_xpu_combination_measurements_20261115.json)に保存した。

| 指標・条件 | 変更前 | 変更後 |
| --- | ---: | ---: |
| 1000 Si原子・28000辺、FP32 graph探索の追跡tensor peak | 13.65 MB | 8.03 MB |
| 同FP64 | 17.97 MB | 12.60 MB |
| 4096原子の完全cloneを32回解放後、GC前に残ったtensor領域：FP32 | 416 tensor、5.77 MB | 0 |
| 同FP64 | 416 tensor、7.34 MB | 0 |
| 保存特徴復元の正のCPU割当て累計：FP64、1,048,576要素 | 40.89 MB | 24.12 MB |
| 同FP32 | 22.02 MB | 13.63 MB |
| BF16 split2の正のCPU割当て累計：入力1024×64、重み512×64 | 18.42 MB | 12.12 MB |
| 同split3 | 42.83 MB | 26.05 MB |
| portable producerの正のCPU割当て累計：FP32、8192辺 | 53.48 MB | 47.19 MB |
| 同FP64 | 106.95 MB | 94.37 MB |
| frozen norm backwardの正のCPU割当て累計：FP32、8192×128 | 75.70 MB | 71.50 MB |
| 同FP64 | 151.39 MB | 143.00 MB |

graphのpeakはprofilerのallocation timelineから計算し、native libraryの確保は含まない。
cloneの測定は循環GCを無効にして、解放後に残るtensor storageをweak referenceで確認した。
ユーザーが作った循環metadataのGCを不要にする変更ではない。
割当て累計は正の`self_cpu_memory_usage`イベントを合計し、入れ子演算の二重計上を避けた。

時間は改善・不変・悪化が混在した。例えばCPU split3の中央値は6.18→5.70 msだったが、
FP64 graph探索は15.66→15.84 msであり、共有環境の部品測定からGPU高速化率は算出しない。
RESPAの10万原子相当momentum検査はNumPy／tracemallocのpeakで2,400,144→301,066 bytes、
中央値0.159→0.0647 msとなった。これもMD全体の測定ではない。

## 検証結果と再実行

Python 3.12.14、PyTorch 2.15.0+cpu、NumPy 2.4.6、ASE 3.29.0、e3nn 0.6.0、
TorchSim 0.6.2、Ray 2.60.0、Hydra 1.3.8、OmegaConf 2.3.1、SciPy 1.18.1、
nvalchemi 0.5.0、Warp 1.19.0、Triton 3.9.0、Sella 2.6.0。
`uv pip check`で240 packageの依存整合を確認した。

最終の直列実行は**2,091 passed、47 skipped、54 deselected、0 failed**、
2,413.09秒（40分13秒）で完走した。選択した2,138件のうち、
skipはCUDA不在による37件、実XPUが必要な9件、NitroのGPU条件による1件だった。
初回のworker異常終了9件はすべて直列再試験で通過し、再現しなかった。
対象は[62 pathの一覧](intel_xpu_combination_test_paths_20261115.txt)。
個別試験と最終統合試験の件数は重複加算しない。

最初の3-worker実行は1,014.66秒で、2,073 passed、47 skipped、11 failedだった。
11件の内訳はworker異常終了9件と数値比較2件である。
worker終了の原因はログだけでは確定できないが、メモリ負荷との関連を疑い、
数値比較の原因確認後、全対象を直列で再実行した。並列実行を成功扱いにはしない。

最終試験の開始から終了まで、変更したソース・テスト35ファイルのハッシュが
変わっていないことを確認した。文書を含む全変更39ファイルはpre-commitを通過し、
READMEのupstream本文を保持した。掲載したPython／bashの21例の構文とローカルリンクも検査した。

統合試験と並行していたgraph／datasetの419件の個別試験は、354件終了時点で
OSに終了された（exit 137）。その時点の空きRAMは約1.3–1.6 GiBで、assertionの
失敗は記録されていないが、完走扱いにはしない。以後は追加の試験プロセスを起動せず、
同じ対象を含む全体試験に一本化した。

```bash
mapfile -t test_paths < docs/core/common_tasks/intel_xpu_combination_test_paths_20261115.txt
CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  "${test_paths[@]}" -c packages/fairchem-core/pyproject.toml -o addopts='' \
  -q -n0 -m 'not pretrained' --disable-warnings --tb=short -r s \
  -k 'not task_initialization and not homogeneous_batching and not stress_computation and not empty_batch_error and not charge_spin_handling and not model_output_validation and not invalid_task_name_raises_error and not test_2_atoms_2_ranks_gloo_a2a_collect'
```

旧TorchSimの分散fixture依存ケースと、許可・実モデル取得が必要なpretrained試験は
この統合対象から除外する。Glooの2-rank実試験も今回単独で再実行したが、
`multiprocessing.Manager`のsocket生成が`PermissionError: [Errno 1]`で拒否され、
通信試験に到達する前に失敗した。この1件は実行環境による未検証として分離し、
統合試験から除外する。実GPU試験のskipを成功として扱わない。
既存の禁止組合せは維持し、Bとcompile／XPU Graph／Nitro／activation checkpointingを
同時に指定した場合は準備段階で拒否する。FP64のBは保存特徴FP32圧縮だけを許可する。
