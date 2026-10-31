# Series A適用後のライブラリ間整合性チェック

2026-10-30 / `intel-xpu-fp32`

対象はSeries A適用済みのコミット`1d35e09`からの追加修正である。
ASE、TorchSim、NumPy、PyTorch、正規化設定・チェックポイント、バッチ実行をまたぐ入力・出力・キャッシュの扱いを点検した。
単独計算では見えにくい、同じオブジェクトの再利用、連続ジョブ、並行呼び出し、キャンセル後の再実行も対象にした。
この監査時点ではSeries Bの低精度化・近似計算は追加していない。
その後の2026-10-31再監査・既定無効のB実装は
[series B and audit follow-up](intel_xpu_series_b.md) を参照。

## 再現して修正した不整合

以下はソースの推測だけではなく、回帰テストで再現条件を固定した問題である。
GPU固有のストリーム・capture・メモリ不足処理については、CPU上の仮想テストで確認した範囲を含む。

| 境界・条件 | 修正前の問題 | 修正後 |
| --- | --- | --- |
| AtomicDataのdtype変換 | 座標の精度変更が原子番号・索引・PBCの型まで変え得る | 浮動小数点フィールドのみ変換し、整数・boolを維持 |
| AtomicDataの複製・分割 | 1系の取り出しが元オブジェクトを返す、複数系の切り出しが親ストレージを共有する | 入力から独立した出力を返し、入れ子メタデータとautogradを保持 |
| 汎用メタデータのバッチ化 | 名前に`index`を含むジョブIDまでシフトする、追加フィールドが脱落する、scalar/bool/tupleの型が変わる | 原子索引の`edge_index`だけをシフトし、型・フィールド・deviceの不整合を検出。namedtupleや循環参照も検証 |
| ASEのキャッシュ | NumPy配列を含む`Atoms.info`の比較で例外、入れ子の変更を見逃す、予測バッファの再利用で過去の結果が変わる | メタデータを所有・比較し、返却配列を独立化。失敗・中断後の古い結果も残さない |
| ASEのラッパー | 標準引数での呼び出し、FormationEnergyCalculatorの状態管理、stress/Hessianの形状に不整合 | ASE標準の呼び出しとキャッシュ規約に合わせ、必要な出力の型・形状を検証 |
| TorchSimの変換 | 小数のcharge/spin/原子番号を整数へ切り捨てる、デバイス転送時にセルや追加情報が先に丸められる | 元のSimStateを検証してから必要なAtomicDataだけを転送。OMOLタスクの既定spinをASEと一致させる |
| TorchSimのメタデータ再利用 | NumPy等からの変更がTensorのversionに現れず古い情報を再利用、返却値から私有キャッシュを変更できる | 内容のスナップショットで比較し、入力・キャッシュ・返却値を独立化。要求stressの欠落も検出 |
| microbatch・非同期投入 | 単独入力だけが書き換わる、投入後の変更が待機中ジョブへ混入、要求プロパティやdtypeの混在、Hessianの誤分割 | 投入時点の入力を所有し、互換な要求だけをまとめる。dense Hessianは1系ずつ実行 |
| 転送バッファ・終了処理 | 例外やKeyboardInterruptで転送中のスロットを再利用、最初の終了処理失敗で残りのworkerを解放しない | 転送完了を待つかスロットを隔離し、元の例外を保持。全workerの終了処理を試みる |
| 近傍グラフ・CSRキャッシュ | 部分グラフ設定・公開整数テーブルが他ジョブを汚染、skin変更やCPUの外部配列変更を見逃す | 呼び出し単位の状態を分離し、設定・内容の変更で無効化。forward時の索引をbackwardまで保持 |
| 不正な構造・配置 | NaN位置が空グラフになる、系ごとの原子が交互に並んだbatchを別系として誤読する | 非有限値、無効なセル、batch/natoms不整合を計算前に拒否 |
| 同一モデルの連続・並行利用 | 古いscatter target、MoLE準備キャッシュ、キャッシュ消去との競合、前ジョブの完了前の上書き | 毎回のトポロジーを使い、同一predictorの推論・結果整形・公開キャッシュ消去をロックで保護。GPU提出間はイベントで順序付け |
| compile・captureの再利用 | 二重ラップ、deepcopy後の元モデル参照、返却勾配がcaptureの再利用バッファを共有する | 設定変更を復元可能にし、copy/pickle時に実行キャッシュを破棄。返却値・勾配を所有 |
| モデル読み込み・大域設定 | EMAの`n_averaged`を読み込み元から削除して再読込に失敗、設定辞書の共有、生成時の乱数・設定変更が残る | 読み込み元メタデータを独立化し、生成時の状態を復元。CPU/XPU推論はCUDAの精度設定を変更しない |
| 正規化・設定ファイル | 整数指定でエラーまたは後続の値が切り捨て、NPZの配列真偽判定でエラー、ベクトル・FP64統計の再読込に不整合 | 所有する浮動小数点統計を使い、形状・精度を保持。NPZのキー優先順位を明示し、`fitted`制御フラグをfit関数へ渡さない |
| バッチサーバー境界 | task/dtype混在で出力行の対応が崩れる、CPU結果の参照共有、OOM後に違うdeviceのallocatorを処理する | 要求を互換性で分離し、出力数・atom/system配置を検証。deviceに応じて解放し、互換でないcheckpoint更新を拒否 |

実装の主な参照先：

- [AtomicData](../../../src/fairchem/core/datasets/atomic_data.py)
- [ASE calculator](../../../src/fairchem/core/calculate/ase_calculator.py)、[TorchSim interface](../../../src/fairchem/core/calculate/torchsim_interface.py)
- [microbatchとworker pool](../../../src/fairchem/core/calculate/batched_inference.py)、[転送バッファ](../../../src/fairchem/core/calculate/device_buffers.py)
- [近傍グラフ](../../../src/fairchem/core/graph/radius_graph_pbc.py)、[CSR処理](../../../src/fairchem/core/models/uma/nn/portable_wigner.py)
- [predictor](../../../src/fairchem/core/units/mlip_unit/predict.py)、[実行キャッシュ](../../../src/fairchem/core/units/mlip_unit/inference_runtime.py)
- [normalizer](../../../src/fairchem/core/modules/normalization/normalizer.py)、[設定読み込み](../../../src/fairchem/core/modules/normalization/_load_utils.py)
- [バッチサーバー](../../../src/fairchem/core/components/batch_server.py)、[クライアント管理](../../../src/fairchem/core/calculate/_batch.py)

## ライブラリを実際に呼び出した試験

実行環境はPython 3.12.14、PyTorch 2.15.0+cpu、ASE 3.29.0、torch-sim-atomistic 0.6.2、NumPy 2.4.6、e3nn 0.6.0、Sella 2.6.0、JAX 0.12.0.1である。
Ray 2.59.0、Hydra 1.3.8、OmegaConf 2.3.1もインストールされている。
CUDA/XPUデバイスは存在しない。

| 試験 | 実行内容・比較対象 |
| --- | --- |
| ASEの短い実ジョブ | 小型UMAの実演算・非空グラフ・非ゼロ力を用いてBFGS、FIRE、FixAtoms、NEB、SumCalculator、FrechetCellFilterを実行 |
| 振動・遷移状態探索の呼び出し | ASE Vibrationsの有限差分と、UMAの解析的Hessianを返すSellaコールバックを実行。探索の収束や物理的妥当性の保証ではない |
| TorchSimの短い実ジョブ | 解析ポテンシャルでFIREとNVE velocity-Verlet、小型UMAでFIREと周期セル最適化を実行。シミュレーションの入出力dtypeはFP32/FP64、UMAネットワークはFP32。最終構造のE/F/stressを直接predictorと比較 |
| バッチ処理 | 小型UMAのHVP・dense Hessian・embeddingsを、単独ジョブ、microbatch、stagingあり/なしで比較 |
| 共有・連続・並行呼び出し | 入力の外部変更、古い出力の保持、copy/pickle、別組成への切替、同じpredictorへの並行投入、キャッシュ消去競合を確認 |
| NumPy・設定・ファイル | データセット全体の回帰試験に加え、配列共有、入れ子・循環メタデータ、型変換、NPZ/PT統計、EMA checkpointの再利用を確認 |
| GPU固有処理の仮想試験 | FakeTensorやstream/event/capture代替物でdevice番号・同期順序・返却勾配・OOM・中断処理を検証。実GPUのカーネル実行とは区別する |

小型UMAの重みはテスト用に生成したもので、配布されている学習済みUMAの精度確認ではない。
FP32を中心に、変換・保存の境界ではFP64も用いた。合算する浮動小数点演算の順序に由来する丸め差は許容誤差で比較している。

## 統合回帰の結果と再現方法

最終統合回帰は**1,151 passed、38 skipped、3 failed、11 errors**（計1,203件、642.54秒）。failure/errorの14件はすべて下記のソケット制約に起因する。これ以外の未解決のテスト失敗は、この試験範囲では残っていない。skipは実GPUを必要とする37件と、上流で無効化されている元素参照値テストモジュール1件である。新規の元素参照値ファイル往復試験は別ファイルで実行し、成功している。

変更した41ファイルはpre-commitの全フックと`git diff --check`を通過した。上の成功件数は最後の統合実行1回の値であり、途中の個別試験を重複して加算していない。

初回には、CPU/XPU推論でCUDAの大域精度を変更しない新しい仕様と、CUDAを明示していなかった既存の仮想TF32テストとの不整合も検出した。テストの対象deviceをCUDAと明示し、設定適用と呼び出し後の復元という検証目的を維持したまま、最後の統合実行で2件とも成功した。

開発用依存関係と上記の追加ライブラリを導入した環境で、リポジトリのルートから実行する。`pytest-xdist`を使わない場合は`-n 2`を省く。事前学習モデルをダウンロードする試験は除外し、実GPU必須の試験は利用可能性に従ってskipされる。以下の再実行用コマンドには、本環境でソケット操作が許可されず失敗した分散関連14件を除く`-k`条件も記載した。ソケットを使用できる環境では、この条件を外して分散関連も検証できる。

```bash
CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/core/datasets \
  tests/core/graph \
  tests/core/calculate/test_ase_interop_jobs.py \
  tests/core/calculate/test_batch_interoperability.py \
  tests/core/calculate/test_batched_inference.py \
  tests/core/calculate/test_requested_properties.py \
  tests/core/calculate/test_resolve_checkpoint.py \
  tests/core/calculate/test_torchsim_interface.py \
  tests/core/calculate/test_torchsim_interop.py \
  tests/core/calculate/test_torchsim_resident_metadata.py \
  tests/core/calculate/test_xpu_pretrained.py \
  tests/core/calculate/test_xpu_torchsim_dispatch.py \
  tests/core/calculate/test_xpu_unsupported_backends.py \
  tests/core/components/test_batch_server.py \
  tests/core/components/test_batch_server_interop.py \
  tests/core/modules/test_normalizer.py \
  tests/core/modules/test_normalizer_interoperability.py \
  tests/core/modules/test_element_references.py \
  tests/core/units/mlip_unit/test_inference_runtime.py \
  tests/core/units/mlip_unit/test_runtime_interoperability.py \
  tests/core/units/mlip_unit/test_inference_settings.py \
  tests/core/units/mlip_unit/test_predict_interoperability.py \
  tests/core/units/mlip_unit/test_predict_memory.py \
  tests/core/units/mlip_unit/test_xpu_predict.py \
  tests/core/units/mlip_unit/test_xpu_preparation.py \
  tests/core/units/mlip_unit/test_unit_utils.py \
  tests/core/models/uma/test_compat.py \
  tests/core/models/uma/test_csr_interop.py \
  tests/core/models/uma/test_curvature_streaming.py \
  tests/core/models/uma/test_empty_graph_fixup.py \
  tests/core/models/uma/test_job_input_interop.py \
  tests/core/models/uma/test_moe_routing_memory.py \
  tests/core/models/uma/test_outputs_memory.py \
  tests/core/models/uma/test_portable_fused_backend.py \
  tests/core/models/uma/test_portable_geometry.py \
  tests/core/models/uma/test_portable_wigner_memory.py \
  tests/core/models/uma/test_quaternion_wigner.py \
  tests/core/models/uma/test_requested_derivatives.py \
  tests/core/models/uma/test_series_a_chunking.py \
  tests/core/models/uma/test_series_a_fast_pytorch.py \
  tests/core/models/uma/test_series_a_integration.py \
  tests/core/models/uma/test_xpu_graph_portability.py \
  tests/core/models/uma/test_xpu_portable.py \
  tests/core/models/uma/test_xpu_radial_kernels.py \
  tests/core/models/uma/nn/test_portable_matmul_memory.py \
  tests/core/models/uma/nn/test_series_a_radial.py \
  tests/core/models/uma/nn/test_unified_radial.py \
  tests/core/units/mlip_unit/test_predict.py::test_prepare_inference_gradients \
  tests/core/units/mlip_unit/test_predict.py::test_predict_uses_inference_tf32_and_restores_caller \
  -c packages/fairchem-core/pyproject.toml -o addopts='' \
  -q -n 2 -m 'not pretrained' --disable-warnings --tb=short \
  -k 'not task_initialization and not homogeneous_batching and not stress_computation and not empty_batch_error and not charge_spin_handling and not model_output_validation and not invalid_task_name_raises_error and not test_2_atoms_2_ranks_gloo_a2a_collect'
```

元のTorchSimテスト13件は、ラッパー呼び出し前に学習用チェックポイントを生成するフィクスチャを使う。この環境ではGlooのTCP初期化が`Operation not permitted`で失敗した。また、2-rankのGlooテスト1件も、multiprocessing ManagerのIPCソケット作成が拒否されて停止した。合計14件は本環境の制約により検証できていない。2件はフィクスチャをテスト本体から取得するため、初回ログではsetup errorでなくfailureとして記録される。分散学習の成功を意味する代替試験ではないが、同じTorchSim標準バリデーターは分散初期化を必要としない小型UMAで実行し、stressの有無の両方で通過した。stressを検証する非周期分子には、テスト側で有限体積の参照セルを明示している。

## 運用上の境界

1. **B70実機での確認は未実施。** XPUのコンパイル、graph capture、カスタムTritonカーネル、複数GPUのストリーム・OOM・性能をCPU試験から保証することはできない。live Ray Serveのデプロイも今回の対象外で、高水準Ray APIによるXPU割当は引き続き未対応である。
2. **モデル重み・設定の外部変更時はpredictorを再作成する。** NumPyや`.data`で重みを直接変更するとPyTorchのversionを迂回できる。毎回の全重み比較は実装していない。GPUメモリを外部ランタイムから変更する場合も、PyTorchの更新・同期規約を守る必要がある。
3. **サーバーのcheckpoint更新後は既存ASE calculatorを再作成する。** 高水準の更新APIは出力スキーマ・設定・参照値の非互換変更を拒否するが、既存クライアント側のASE結果キャッシュは遠隔で一括無効化できない。低水準の更新APIは呼び出し側で管理する。
4. **プロセス全体の状態は任意の他ライブラリから完全に隔離できない。** fairchemのコンストラクター間では乱数・既定dtypeの一時変更を保護・復元する。ロックを共有しない別ライブラリが同時に乱数や既定dtypeを使う場合まで独立にはならない。乱数を使うforward演算の生成器まで分離する仕様ではない。明示指定するPyTorchスレッド数もプロセス全体の設定である。
5. **所有権と検証にはコストがある。** 入力・出力の複製、内容比較、有限値チェックには追加のメモリ転送や同期が生じ得る。静的配置の再利用や有界キャッシュは維持しているが、今回の修正がB70上でどの程度の時間・メモリ差になるかは未測定である。
6. **無効な入力は早くエラーにする。** 小数のcharge/spin、異なるdtypeを無指定で混ぜるbatch、欠落した要求出力、統計値を読み込まずに`fit.fitted=True`だけを再利用する設定は拒否する。以前の黙示的な切捨てや欠落を互換動作として残さない。

今回の試験は、記載した経路とバージョンで再現した不整合の修正を確認するものであり、未検証のライブラリや全組合せで不具合がないという保証ではない。
