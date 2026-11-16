# XPUのデバイス割当て・併用・メモリ再監査

2026年11月16日。`intel-xpu-fp32`の開始コミット
`a536295dd6a3aecd4c7e1209ef6f1be1e0945cbe`を対象にした。
[11月15日の併用監査](intel_xpu_combination_audit_20261115.md)を引き継ぎ、
現在のソースとテストを取得し、新しいPython環境で再検査した。

## 修正内容

| 条件・問題 | 修正と確認方法 |
| --- | --- |
| 入力tensorが`xpu:1`でも、別GPUがcurrent deviceになっている | radial、Wigner producer、projected producer、consumerの4経路で、Triton起動を入力deviceのcontextに入れる。通常終了と起動例外の8ケースでGPU選択と復元を模擬確認。2枚の実XPUを使う値・一次微分・HVP比較も追加 |
| 数値blockのXPU Graphを、異なる外側のautocast・matmul・oneDNN TF32設定や`inference_mode`で再利用できる | Graphのkeyに実行時の精度設定とinference modeを追加。同じ形状・dtype・device・streamでも条件が異なれば別entryにする。元の条件へ戻した場合のkey一致も確認 |
| 他ライブラリでinter-op poolを初期化済み、または複数predictorで異なる`torch_num_threads`を指定 | 一度しか設定できない`set_num_interop_threads`をpredictorから呼ばない。`torch_num_threads`はCPU intra-op設定だけに適用。独立した実PyTorchプロセスで、初期化済みinter-op poolを保持したまま2→1→2 threadsへ変更できることを確認 |
| Hessianの次のvmap backward中にも、前blockと基底tensorが残る | 最終出力へコピー後にloop-local参照を解放。FP32/64でweak referenceによる寿命検査と解析Hessian比較。学習時に必要な参照はautograd・block一覧が保持 |
| resident計算で、前stepのforces等が次のmodel forwardまで残る | update後にローカル参照を解放。observerや更新後stateが必要とする参照は保持される。observerが出力を保存する場合としない場合を検査 |
| Lanczosの再直交化で、射影を引くたびに新しい残差vectorを確保する | 残差の所有コピーを一度作り、そのbufferへ同じ順序で減算。元のHVP積と基底は変更しない。FP32/64で変更前との厳密一致、入力不変性、既存の縮退・極端なscale・残差試験を実施 |
| Hutchinsonの次のHVP計算中も前sample・probe・deltaが残る | online統計への反映後に不要なvector参照を解放。推定式、乱数列、標本分散・標準誤差は変更しない |
| 分散process groupを一度も使っていないcleanupでも0.5秒待機する | 実際にgroupを破棄した場合だけ既存のport解放待機を行う。非分散・分散・繰返しcleanupの分岐を検査 |

TritonのJIT実装は起動時のcurrent deviceからstreamとdevice cacheを選ぶ。
GPUを指定したtensorを渡すだけでは、この選択を保証できない。
今回の修正ではカーネルの演算式、tile、精度、autotune候補は変更していない。
CPU上の模擬試験が確認するのはdevice contextへの入退場であり、
Intelドライバーによる起動やGPU演算結果の確認ではない。

`torch_num_threads`は引き続きプロセス全体のCPU設定であり、predictorごとに独立した
thread poolを作るものではない。複数ライブラリや複数device workerがthread方針を
共有する場合は、アプリケーション側で先に設定し、predictorには`None`を指定する。
inter-op数を変更する場合は、利用者がプロセス開始時、並列処理の開始前に一度だけ
`torch.set_num_interop_threads(...)`を呼ぶ。

## 併用・所有権の確認範囲

| 機能群 | 今回の検査 |
| --- | --- |
| A系列のgraph・geometry・radial・SO2 | graph探索、近傍再利用、PBC、padding、portable演算、一次・二次微分、FP32/64 |
| A/Bの同時利用 | 既存23ケースの二因子covering array、非対応backend設定65ケースの拒否、外側autocastからの分離 |
| B系列 | GEMMのCPU emulation、量子化、低ランク、保存特徴圧縮、radial近似、補償集約、教師データ・蒸留、RESPA、coarse/refine、DyNEB、Lanczos、Hutchinson |
| 入出力・引数 | requested properties、ModelSpec、Hydra設定往復、Tensor dtype/device、入力・出力の独立性、metadataのalias、copy/pickle |
| メモリ・並行利用 | MoLE cache、staging slot、device pool、stream/eventの模擬制御、例外後の復旧、weak referenceによる不要bufferの解放 |
| wrapper | ASE、TorchSim、microbatch、batch server、NitroのCPUから検査可能な制御経路 |

Bとcompile／XPU Graph／Nitro／activation checkpointingの既存の禁止組合せは維持した。
FP64で許可するB設定も、保存特徴をFP32へ圧縮する既存の範囲を維持した。
二因子covering arrayは三因子以上の全組合せや全入力を網羅する試験ではない。
学習済み配布checkpointの科学的精度認定にも該当しない。

## 測定

PyTorch 2.15.0+cpu、2 threadsで、開始コミットの関数と変更後の関数を比較した。
[測定JSON](intel_xpu_runtime_measurements_20261116.json)と
[再実行スクリプト](measure_runtime_audit_20261116.py)に条件・生値を残した。
以下のMBは10⁶ bytes、MiBは2²⁰ bytesである。

| CPU部品測定の指標 | 変更前 | 変更後 |
| --- | ---: | ---: |
| Lanczos再直交化・FP32の累積割当量、32,768成分・32基底・2 pass | 16.78 MB | 8.52 MB |
| 同FP64の累積割当量 | 33.55 MB | 17.04 MB |
| 同FP32の追加ピーク | 393,216 bytes | 262,148 bytes |
| 同FP64の追加ピーク | 786,432 bytes | 524,296 bytes |
| 解析Hessian・FP32、768座標・64行/blockの追加ピーク | 3,161,608 bytes | 2,965,000 bytes |
| 同FP64 | 6,322,696 bytes | 5,929,480 bytes |
| synthetic resident loop・FP32の追加ピーク | 16 MiB | 12 MiB |
| 同FP64 | 32 MiB | 24 MiB |

累積割当量とピークは別の指標であり、合算できない。
ピークはPyTorch CPU allocatorのmemory eventから求めた追加確保量で、
測定開始前に作った入力・基底、Pythonオブジェクト、native libraryの独自確保は含まない。
再直交化では同じ順序の減算を維持し、両dtypeで変更前と`rtol=atol=0`の比較で一致した。
Hessianは四次関数の解析例で、学習済みUMAの全計算を測った値ではない。
resident loopは出力1,048,576要素と一時作業領域2,097,152要素を持つ模擬modelを
4 step呼んだ結果で、前stepの出力が次の作業領域に重ならないことを確認している。

速度は共有CPU環境での参考値としてJSONへ収録した。
例えばFP64再直交化は、最初の測定では2.561→2.624 ms、
再実行スクリプトの確認時には2.465→2.290 msとなり、測定間の変動があった。
メモリ削減は両回で同じだったが、この時間からUMA全体やGPUの高速化率は算出しない。
非分散cleanupでは不要だった0.5秒の固定待機を除き、
実際にprocess groupを破棄したときの待機は維持した。

## 検証結果

最終の直列統合試験は**2,167 passed、58 skipped、54 deselected、0 failed**、
316.11秒（5分16秒）で完走した。実行対象2,225件のうち、skipは
CUDA不在37件、NitroのGPU条件1件、XPU実機が必要な20件である。
追加した25ケースのうち21件は通過、2-GPU実機試験4件はskipになった。
実行開始から終了まで、変更したソース・テスト14ファイルのhashが
変わっていないことを確認した。
[検証JSON](intel_xpu_runtime_validation_20261116.json)に件数、skip理由、hashを記録した。

修正前には、スレッド再設定の停止、4 kernelの誤ったcurrent device、
Graph keyへの精度条件の欠落、不要bufferの保持、非分散cleanupの待機を
新しい回帰試験で再現した。修正箇所の先行試験は116 passed・7 skippedだったが、
統合試験と重複するため最終件数へ加算していない。

Glooの2-rank実通信試験は統合試験とは別に実行した。
`multiprocessing.Manager`のsocket生成が`PermissionError: [Errno 1]`で拒否され、
親processは`EOFError`で終了した。通信の初期化に到達していないため、
この1件を製品コードの合格にも、通信障害の修正確認にも数えない。

実行環境はPython 3.12.14、PyTorch 2.15.0+cpu、NumPy 2.4.6、ASE 3.29.0、
e3nn 0.6.0、TorchSim 0.6.2、Ray 2.60.0、Hydra 1.3.8、OmegaConf 2.3.1、
SciPy 1.18.1、nvalchemi 0.5.0、Warp 1.19.0、Triton 3.9.0、Sella 2.6.0。
`uv pip check`で196 packageの依存整合を確認した。
文書・測定スクリプトを含む全変更20ファイルでpre-commitを通過した。

### 統合試験の再実行

[対象69 path](intel_xpu_runtime_test_paths_20261116.txt)を使う。
`pretrained`試験、旧TorchSimの分散fixture依存ケース、実通信が必要なGlooの1件を
このCPU統合コマンドから除いている。

```bash
mapfile -t test_paths < docs/core/common_tasks/intel_xpu_runtime_test_paths_20261116.txt
CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  "${test_paths[@]}" -c packages/fairchem-core/pyproject.toml -o addopts='' \
  -q -n0 -m 'not pretrained' --disable-warnings --tb=short -r s \
  -k 'not task_initialization and not homogeneous_batching and not stress_computation and not empty_batch_error and not charge_spin_handling and not model_output_validation and not invalid_task_name_raises_error and not test_2_atoms_2_ranks_gloo_a2a_collect'
```

### Intel GPU実機で追加確認する試験

追加した2-GPU試験は、`xpu:0`が選択された状態で`xpu:1`のtensorを4種類の
独自kernelへ渡し、通常PyTorch演算との値・一次微分・HVPの一致、返却deviceと
呼出し後のcurrent deviceを検査する。対応するTriton-XPUが必要である。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/core/models/uma/test_xpu_kernel_device_scope.py \
  tests/core/models/uma/test_xpu_radial_kernels.py \
  tests/core/models/uma/test_xpu_portable.py \
  tests/core/units/mlip_unit/test_inference_runtime.py \
  -c packages/fairchem-core/pyproject.toml -o addopts='' \
  -q -n0 --disable-warnings -r s
```

この環境にはXPU/CUDAがない。Arc Pro B70のnative kernel、実XPU Graph capture、
CUDA Nitro、配布checkpointを使う長時間MD、実RAM/VRAMの上限・速度は未検証である。
機能が全条件で停止・不安定化しないことを保証するものではない。
