# A/B系列のライブラリ間整合性再監査

2026-10-31 / `intel-xpu-fp32` / 監査開始点 `7830a466`

B系列の実装後に、PyTorch、ASE、TorchSim、NumPy、Hydra/OmegaConf、
checkpoint/EMA、サーバーのバッチ処理をまたぐ境界を再点検した。
以下の不具合を再現し、修正と回帰試験を追加した。B機能は引き続き既定で無効である。
精度を変えるオプションを自動的に有効にする変更は含まない。

## 再現した問題と変更

| 境界 | 修正前の問題 | 修正後の動作 |
| --- | --- | --- |
| TorchSimの公開コンストラクター | `FairChemModel`からB設定を渡せず、内部predictorの差し替えが必要 | `inference_settings`と`overrides`を起動時に受け付け、既定の`batch`は維持 |
| バッチサーバー | optional fieldの有無、追加Tensorのdtype/末尾形状が異なる正常なジョブを混ぜ、バッチ全体が失敗 | microbatchと同じ互換性判定で分離し、入力順に結果を戻す |
| ModelSpecのモデル識別 | 公開設定の変更後に、等しいと判定されるモデルのhash・実際のB設定が異なる | 正規化した私有スナップショットを同一性・hash・ロードに共用 |
| 初回B準備の失敗と再試行 | 変換済みモデルにB変換を再適用し、再試行不能または古い準備状態を使用 | 失敗・中断を記録し、次回は未変換templateまたはcheckpointから再構築。正常時の全モデル追加コピーは不要 |
| fallback中のcheckpoint読込失敗 | 旧モデルの解放後に読込が失敗すると、次の呼出しでmodel属性がなく復旧できない | 未完了状態を保持し、予測・calculatorの入力検証から再読込を試行 |
| 外部の既定device | CPUロード、Gaussian表のFP64準備、蒸留sampler、exportの新規Tensorが外側のdevice設定を継承 | CPUで行う工程はdeviceを明示。外側の設定は保持 |
| 外部のinference_mode | モデル構築・準備後の組成切替・失敗後のtemplateコピーがinference tensorとなり、後の微分が失敗 | 対象の構築・コピー範囲でのみinference modeを解除し、呼出し元の設定を復元 |
| B変換済みstate_dict | 不正なmetadataを検出する前にbias/packed bufferを書き換える。浮動小数のpacked値も暗黙cast | 変換層の構造、設定、キー、shape、dtype、有限値、正のscaleをコピー前に照合 |
| 独自Linear/Gaussianとの併用 | 独自forwardやhookをB置換で黙って失う | Linearは未適用理由をmanifestに記録して維持。Gaussianの対応外置換は変更前に拒否。同一設定の再適用は既存wrapperを維持 |
| 入れ子のジョブメタデータ | frozen/slotted dataclass内のTensorがclone、dtype/device転送、容量検査から漏れる | フィールドを再帰走査し、循環・共有関係・必要なautogradを維持して複製・転送・追跡 |
| 保存特徴量の圧縮 | NaN/Infが一つあるだけでscaleが無効になり、他の正常な大きい有限値までInfになる | 有限要素だけからscaleを算出。元の非有限値は隠さない |
| 蒸留入力 | 独自メタデータ内に教師のautograd graphや元deviceのTensorが残る | 教師入力を再帰detachして私有CPUデータ化し、各学習stepで必要なdtype/deviceへ移動 |
| 試運転済みUMAの蒸留 | 前forwardのMoLE内部キャッシュが非leaf Tensorを持ち、学生のdeepcopyに失敗 | 既知の再計算対象キャッシュだけをdetachした私有コピーに置換し、元学生の勾配・共有関係を保持 |
| 蒸留の呼び出し・乱数 | 外側のinference_modeや既定deviceで失敗。内部の異なる乱数退避ロックが干渉し得る | 学習範囲で微分を有効化、CPU samplerを使用。fairchem内の構築・蒸留は共通ロックを使用 |
| 蒸留checkpoint/EMA | `get_extra_state()`の辞書で保存失敗。version metadataが欠落。meta deviceのEMA counterを保存し得る | 非Tensor状態も再帰コピーし、version metadataとEMAのprefixを保持。CPUで構築・構造照合して保存 |
| ASE RESPAのinplace結果 | positions更新後にread-only momentaで失敗し、呼出し元が部分更新される | モデルを作る前に更新対象配列の書込み可否を確認 |
| 失敗したワークフローの保持 | ジョブ管理側で例外を保存すると、tracebackが内部モデルを保持。RESPA成功時にも循環参照が残る | 完了済み例外frameのlocalsと内部循環参照を解放。例外型・cause・発生位置は保持 |
| Lanczosの数値範囲 | 有限の大きい値でnormがInf、小さい値で残差が0となり誤って収束 | device/dtypeを維持したスケール付きnormと、小さい射影行列のスケール調整を使用 |
| Hutchinsonの統計 | HVP自体が有限でもtrace/分散のFP64累積がoverflowし、NaN/Infを返す | 統計の非有限化を明示的にエラーとし、HVPのスケール変更を案内 |

不正なB変換層を含むcheckpointは`strict=False`でも拒否する。
その層全体を省く、無関係なモデル部分のpartial loadは許可する。
変換対象を含むモデルのpre-load hookも検証するが、任意の外側モジュール・ユーザーhookが
書換え後に例外を出す一般の`nn.Module.load_state_dict`までトランザクション化したものではない。

ワークフローの例外解放では、post-mortem debuggerで参照できる完了frameのローカル変数が
消える。例外本文、型、連鎖、tracebackのファイル・行情報は残る。
呼出し元の生存中generator/coroutine/async-generatorのframeは解放対象から外す。
呼出し元が所有するcalculatorをreset/closeする処理は行わない。

## 実行した試験

環境はPython 3.12.14、PyTorch 2.15.0+cpu、NumPy 2.4.6、ASE 3.29.0、
torch-sim-atomistic 0.6.2、e3nn 0.6.0、Hydra 1.3.8、OmegaConf 2.3.1。
Ray 2.59.0、Sella 2.6.0、JAX 0.12.0.1も導入した。CUDA/XPUは利用不可。

実際の小型UMA・非空グラフ・autogradを用い、以下を確認した。

- INT8でのTorchSim FIREと、公開コンストラクターからの設定転送。
- 異種メタデータを持つジョブのサーバーバッチ予測と単独予測の一致。
- 失敗・KeyboardInterrupt後のB初期化再試行と、新規モデルによる予測の一致。
- energy/forces/stress、loop/vmap Hessian、HVP、低精度・量子化状態の保存再読込。
- ASE SumCalculator/BFGSの並行ジョブ、RESPA/粗探索→精密化/DyNEBの失敗時解放。
- 蒸留の学習、標準checkpoint/EMA exportとpredictor再読込。
- コピー、pickle、入れ子・循環参照、変更した入力・過去出力の独立性。
- 大小の作用素スケールでの曲率残差と、統計overflowの検出。

最終統合試験は **933 passed、5 skipped、0 failed、0 errors**（527.88秒）。
Bの全関連試験と、変更が影響するAのdatasets/ASE/TorchSim/server/predictor/runtimeを
2 workerでまとめて実行した。今回追加した回帰ケース159件を含む。
途中の個別試験の件数は重複加算していない。

5件のskipは、実XPUのstaging、streamをまたぐTorchSim metadata、numeric graph、
XPUでのMoLE準備、およびCUDAでのAtomicData変換である。
実機がないことによるskipを仮想テストの成功で置き換えてはいない。
全変更26ファイルのpre-commitと`git diff --check`も通過した。

再実行コマンド（開発依存関係と上記の追加ライブラリ、pytest-xdistを導入した環境）:

```bash
CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/core/calculate/test_ase_interop_jobs.py \
  tests/core/calculate/test_b_wrappers_audit4.py \
  tests/core/calculate/test_batch_interoperability.py \
  tests/core/calculate/test_batched_inference.py \
  tests/core/calculate/test_requested_properties.py \
  tests/core/calculate/test_series_b_curvature.py \
  tests/core/calculate/test_series_b_curvature_interop.py \
  tests/core/calculate/test_series_b_distillation.py \
  tests/core/calculate/test_series_b_distillation_interop.py \
  tests/core/calculate/test_series_b_interop.py \
  tests/core/calculate/test_series_b_workflows.py \
  tests/core/calculate/test_series_b_workflows_interop.py \
  tests/core/calculate/test_torchsim_interface.py \
  tests/core/calculate/test_torchsim_interop.py \
  tests/core/calculate/test_torchsim_resident_metadata.py \
  tests/core/calculate/test_xpu_torchsim_dispatch.py \
  tests/core/calculate/test_xpu_unsupported_backends.py \
  tests/core/components/test_b_server_audit4.py \
  tests/core/components/test_batch_server.py \
  tests/core/components/test_batch_server_interop.py \
  tests/core/datasets \
  tests/core/models/uma/nn/test_approximate_features_interop.py \
  tests/core/models/uma/nn/test_approximate_linear.py \
  tests/core/models/uma/nn/test_approximate_linear_interop.py \
  tests/core/models/uma/nn/test_approximate_ops.py \
  tests/core/models/uma/nn/test_approximate_saved_features.py \
  tests/core/scripts/test_compare_series_b.py \
  tests/core/units/mlip_unit/test_inference_runtime.py \
  tests/core/units/mlip_unit/test_inference_settings.py \
  tests/core/units/mlip_unit/test_predict_audit_round3.py \
  tests/core/units/mlip_unit/test_predict_interoperability.py \
  tests/core/units/mlip_unit/test_predict_memory.py \
  tests/core/units/mlip_unit/test_runtime_interoperability.py \
  tests/core/units/mlip_unit/test_series_b_integration.py \
  tests/core/units/mlip_unit/test_series_b_runtime_audit4.py \
  tests/core/units/mlip_unit/test_unit_utils.py \
  tests/core/units/mlip_unit/test_xpu_predict.py \
  tests/core/units/mlip_unit/test_xpu_preparation.py \
  -c packages/fairchem-core/pyproject.toml -o addopts='' \
  -q -n 2 -m 'not pretrained' --disable-warnings --tb=short \
  -k 'not task_initialization and not homogeneous_batching and not stress_computation and not empty_batch_error and not charge_spin_handling and not model_output_validation and not invalid_task_name_raises_error'
```

`-k`は、この環境でソケット生成が禁止される分散checkpoint fixtureに依存する既存TorchSim試験を
除く。分散実行を許可した実機ではこの条件を外して確認する。配布checkpointのdownloadを
必要とするpretrained試験も除外し、実GPUを要する試験は利用可能性に従ってskipする。

## 検証の限界と運用条件

- **実B70/XPU試験は未実施。** device/stream/OOM等の代替API試験は、Intelの実カーネル、
  ドライバ、速度、peak VRAMの確認ではない。学習済み配布UMAの化学的精度も未検証。
- **サーバーのローカル処理とlive deploymentは別。** 実UMAでサーバーmixinを呼び出したが、
  live Ray Serveの複数プロセス・GPU割当を確認したものではない。XPU高水準Ray起動は未対応。
- **Bは精度評価を要する実験オプション。** CPUのBF16/FP16エミュレーションはIntel XMXの
  数値特性・速度を再現しない。B07のFP32補償和にもFP32中間値のoverflow範囲があり、
  通常のpredictorでは非有限出力検査が検出する。FP64への自動fallbackはない。
- **重みを直接変更する場合はモデルを作り直す。** `.data`や外部NumPy等からの更新を
  常時全重み比較する仕様ではない。HVP解析の途中に作用素の重み・設定・構造を変更しない。
- **大域状態の完全隔離は行わない。** 共通ロックはfairchem内の一時設定・乱数退避を調整する。
  蒸留callbackと別threadのpredictor再ロードが競合した場合、ロック待ちの行き詰まりを避けて
  再ロード側は明示的なエラーを返す。競合処理の完了後に再試行する。通常の準備済み予測を
  全体ロックで直列化しない。外部ライブラリの同時乱数利用や大域精度変更まで制御しない。
  B03は専用プロセスで使う。
- **所有権の検証・コピーにはコストがある。** 今回の修正によるGPU時間・メモリ差は未測定。

この記録は、記載したバージョンと試験経路で確認した結果であり、全ライブラリ・設定・
入力の組合せで不具合がないという保証ではない。
