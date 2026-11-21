# Intel XPU・A/B系列・UMA Nitro 最終再監査

2026年11月9日。開始点は`intel-xpu-fp32`の
`7b404662ef606c132e26c099b22712c0eba081c9`。
確認したupstream mainは`36b17c6dca30fd95f0f154837ab297191974a4c1`。
今回の変更一覧と検証結果を記録する。過去の監査結果は
[Nitro監査](intel_xpu_nitro_audit.md)と[B系列監査](intel_xpu_series_b_interoperability_audit.md)
に残し、今回の結果と合算していない。

## 結論と適用範囲

既存の公開API、A系列のFP32基準、B系列の明示指定方針を維持し、
再現できた不整合と不要な処理・メモリ保持を修正した。
導入と運用の入口を[README](../../../README.md)の上部に追加し、元の本文は保持した。
この版はコード監査・CPU回帰試験を完了した版であり、B70の実機認定版を意味しない。

NitroはCUDA専用で、Intelでは通常UMAのPyTorch XPU経路を使う。
本フォークを独自SYCL実装へ置き換えたわけではない。
Nitro本体についてはupstream初回導入以降の追加変更が見当たらず、
今回参照した直近の関連修正は一般近傍探索の左手系セル対応
[#2134](https://github.com/facebookresearch/fairchem/pull/2134)である。
upstream全体を無条件にmergeしたものではない。

## 発見事項と改修一覧

P1は誤結果・復元不備・実行停止につながるもの、P2は条件付き不整合・
メモリ保持、P3は不要な処理の削減を表す。条件と検証範囲を各行に示す。

| 優先度 | 発見した条件・影響 | 改修と検証 |
| --- | --- | --- |
| P1 | Nitroで統合radial重みだけが残る構造なのに、当該bufferが非永続でnative `state_dict()`から欠落する。未統合expertと統合時の組成契約も欠落していた | radial重みを永続化し、version付きextra stateにexpertと組成契約を保存。未統合／統合、`assign`の両設定で復元と所有権を検査 |
| P1 | 親`nn.Module`経由のロードはNitroのpublic `load_state_dict()`を通らず、古いcapture／派生重みcacheを残せる | 再帰ロードhookでリソースを解放し、置換parameterの凍結を維持。直接／親経由の両ロードを検査 |
| P1 | ASEの`FixAtoms`等が変換用コピーへの座標設定に再適用され、モデル用wrapped座標と外部近傍リスト用座標が異なる。cell拘束では補完cellが特異行列に戻る | 変換時の座標・cell設定に`apply_constraint=False`を指定。実ASE／Pymatgenで周期image offsetとcellの一致を検査 |
| P1 | バッチサーバーが入力ごとのproperty要求を大域設定で置き換え、Hessian要求へenergyだけを返す場合がある | 個別要求を正規化してpredictorへ明示し、Hessianをsingleton処理。実小型UMAで混在要求と大域設定の優先順位を検査 |
| P1 | nvalchemi固定バッファーが不足すると、近い辺も欠落したまま近傍上限を適用する | 実際の必要容量を読み、上限付きで再確保・再試行。実CPU nvalchemiで不足容量時の辺数を8本から正しい12本へ復旧 |
| P1 | nvalchemi用に加えたcutoff余裕が戻されず範囲外の辺を返す。cell offset計算をFP32へ固定し、FP64入力でも距離を失う | 入力dtypeで距離を評価し、本来のcutoffで再選別。cutoff直外の2辺が0辺になること、微小なFP64距離が残ることを実CPU backendで検査 |
| P1 | 左手系の周期cellはnvalchemi backendとの契約を満たさない | upstreamの基底反転とinteger image offset復元を反映。右手系を要求する模擬backend、元cellの所有権、各種PBCを検査 |
| P2 | モデルfallback時に`_numeric_runtimes`が旧backboneとgraph領域を保持し、置換モデルと重複して存在する | 再ロード前にruntime参照を解放。弱参照で旧モデルが再ロード前に解放されることを検査 |
| P2 | Nitroの直接呼出しは外側のCUDA autocastを引き継ぎ、内部のFP32前提演算に低精度が混入し得る | 直接forwardの範囲でautocastを無効化し、正常／例外終了時に外側の設定を復元。GPU状態を模擬して検査 |
| P2 | Nitroの並列stream分岐で途中例外が起きると、先行分岐のjoinとtensor利用宣言を飛ばす | 例外経路でも分岐完了を順序付け、元の例外を保持。stream/eventの模擬試験で検査 |
| P2 | `r_forces=False`、`r_stress=False`でもASE calculator結果／`atoms.info`の一部ラベルを読み、不要な変換や読込エラーが起きる | 除外したラベルを読み込まない。壊れた除外ラベルを含む入力で回帰検査 |
| P2 | pinned host bufferのfactoryにCPU指定がなく、他ライブラリのdefault device設定を継承する | `device="cpu"`を明示。外側のdevice contextを維持した模擬転送テストを追加 |
| P2 | nvalchemi呼出しでCPUの`natoms`とGPU座標を混在させると、近傍制限処理のdeviceが一致しない | 原子数tensorを座標deviceへ正規化。CUDA専用回帰を追加したが、この実機ケースは環境上skip |
| P2 | Bのlinear変換helperを外側`inference_mode()`内で直接使うと、準備bufferが後の座標微分で保存できない | 準備時のみ推論モードを解除。INT4／INT8／低ランク／BF16参照演算で一次・二次入力微分を検査 |
| P2 | NumPy1.26互換manifestが通常版からずれ、必須`backoff`不足、extra不足、非対応Pythonの許容がある | 意図したNumPy／Python制約以外を同期。Pythonを3.11～3.12へ限定し、依存・extra・XPU wheelのversion受入を検査 |
| P3 | 出力選択だけの変更でMoLE cacheが外れ、同じ組成のモデルを再準備・重複保持する | 準備keyから`requested_properties`だけを除外。実小型UMAで同一モデルの再利用と出力を検査 |
| P3 | RPC通信経路でも、使わないpickle payloadとmetadata snapshotを作成する | broadcast経路に限定。broadcastでは初回を含む事前直列化検査を維持 |
| P3 | バッチ出力分割で、systemとpropertyごとに全原子をboolean索引する | 検証済みの連続system順と原子数からsliceを作る。返却時の独立CPUコピーは維持 |
| P3 | 内部graph生成でも、使用しないASEコピーへwrapped座標を書き戻す | 書き戻しを外部graph生成時だけに限定。通常MD経路の原子数に比例する余分なコピーを除去 |
| P3 | Lanczosが基底／HVPのリストと積層行列を同時保持し、教師ラベル収集は前回GPU出力を次の予測まで保持する | コピー後・使用後に参照を順次解放。数値結果と寿命を回帰検査。無制限リークを確認したという意味ではない |
| P3 | nvalchemiのdense隣接bufferを距離計算時まで保持し、辺ごとの3×3 cellも展開する | 独立した辺出力の取得後にdense bufferを解放し、共通のcell offset演算を利用。保持期間・出力の回帰検査 |

Nitro native state dictは今回から必要な重みとmetadataを保存する。
旧版で保存したnative state dictには復元不能な欠落があるため、strict loadで欠落を通知する。
公式配布checkpointの構造、モデル許可リスト、`from_checkpoint`の読込方式は変更していない。
`strict=False`による不完全な重み復元を検証成功とみなしてはいけない。
親wrapper経由のロードは逐次利用を検証したもので、同じモデルへの重み再ロードと
forwardを別threadで同時実行するatomicな更新を保証するものではない。

## 性能・メモリについて確認できた範囲

CPU上でバッチ返却処理のみを比較した。PyTorch 2.15.0+cpu、2 threads、合計16,384原子、
energy／forces／stressの分割で、モデル計算とGPUからの転送は含まない。
共有実行環境のため測定ノイズを含む。

| system数 | 修正前の中央値 | 修正後の中央値 |
| --- | ---: | ---: |
| 1 | 0.271 ms | 0.0169 ms |
| 8 | 4.290 ms | 0.369 ms |
| 32 | 6.060 ms | 0.658 ms |

生の値と条件は[測定JSON](intel_xpu_final_measurements.json)に保存した。
これはB70／CUDA全体やMDの速度改善率ではない。
一般nvalchemi adapterでも実CPU backend、10,000原子、FP32、2 threads、cutoff 2 Å、
buffer容量128、厳密近傍上限6で同一出力を確認した。
PyTorch profilerが追跡したCPU割当のpeakは39.07 MBから26.67 MBへ減少した。
これはnative Warpの確保量、プロセス全体のRSS、GPU VRAMを含まない測定である。
交互に10回測った時間の中央値は28.11 msから25.73 msだったが、測定値の範囲が広く重なるため、
安定した速度改善率としては扱わない。
モデル再準備、不要な直列化、重複buffer保持の削減は、回数・所有権・生存期間の検査で確認した。
GPU peak VRAM削減率やNitro実機のstream性能は未測定である。

Lanczosは依然として`O(dimension × max_iterations)`の保存領域を必要とする。
nvalchemiの完全性確認には必要容量の読出しや追加選別が必要なので、
正確性の修正自体がすべて高速化になるとは限らない。
推論入力の完全な所有コピー、metadata変更検査、返却出力の所有権は保持した。
キャッシュやコピーの制限を外して見かけの速度を上げる変更はしていない。

## 検証環境・結果

Python 3.12.14、PyTorch 2.15.0+cpu、NumPy 2.4.6、ASE 3.29.0、e3nn 0.6.0、
TorchSim 0.6.2、Ray 2.60.0、Hydra 1.3.8、OmegaConf 2.3.1、SciPy 1.18.1、
Pymatgen 2026.10.24、Pymatgen-core 2026.10.31、nvalchemi-toolkit-ops 0.5.0、
Warp 1.19.0、Triton 3.9.0。`uv pip check`は全依存関係の整合を確認した。
CUDAとXPUの実機はどちらも存在しない。

最終統合回帰は**1,663 passed、44 skipped、0 failed、0 errors**
（55対象path、4 workers、487.34秒）。個別試験の件数は重複加算していない。
44件のskipは、CUDAが必要な37件、XPUが必要な6件、Nitro対応GPUが必要な1件。
変更28ファイルのpre-commit全hookと`git diff --check`を通過した。
試験開始時と完了後の実装・テスト・manifestのblob hash一致も確認した。
READMEは元の本文とのbyte単位の一致を確認し、追記したPython例3件・shell例3件の
構文、設定例2件の実行、相対リンクも検査した。モデルをダウンロードするREADME例は未実行。

実小型ランダム重みUMA、ASE／TorchSimの計算、実CPU nvalchemi、
ソフトウェアで模擬したGPU境界を区別して検証した。
Ray／Pymatgen／Pandas等の任意ライブラリがない状況でも、公開UMA APIとモデル登録が使え、
未導入のRay機能の入口では明確なエラーになることを別プロセスで検査した。
NumPy1.26環境自体へのインストール試験は行っていない。

実2-rank Gloo試験も別途試したが、`multiprocessing.Manager()`のsocket作成が
`PermissionError: [Errno 1] Operation not permitted`となり、親側で`EOFError`になった。
分散初期化より前の環境制限であり、実通信の合格結果には含めていない。
統合回帰ではこの試験と同種の分散fixture、配布checkpointを要する試験を除外する。
除外をpassやskipへ書き換えてはいない。

再実行（repository root、開発依存・extras・TorchSim・pytest-xdist・Triton導入済み）:

```bash
mapfile -t test_paths < docs/core/common_tasks/intel_xpu_final_test_paths.txt
CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest "${test_paths[@]}" \
  -c packages/fairchem-core/pyproject.toml -o addopts='' \
  -q -n 4 -m 'not pretrained' --disable-warnings --tb=short -r s \
  -k 'not task_initialization and not homogeneous_batching and not stress_computation and not empty_batch_error and not charge_spin_handling and not model_output_validation and not invalid_task_name_raises_error and not test_2_atoms_2_ranks_gloo_a2a_collect'
```

実分散環境では上の`-k`除外を外して通信試験も実施する。
実B70では[READMEの受入試験](../../../README.md#実機での受入試験とmd測定)を実行し、
対象構造・task・時間刻みでエネルギー／力／stress、長時間drift、allocated／reserved memory、
driver側の保持量を確認する必要がある。今回のCPU・模擬試験でこれらの実測を代替してはいない。
