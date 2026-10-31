# Series A/B validation record — 2026-10-31

この記録はB実装時点の試験結果である。以後の追加修正と統合再試験は
[B系列を含む整合性再監査](intel_xpu_series_b_interoperability_audit.md)に記録する。

CPU環境: Python 3.12、PyTorch 2.15.0+cpu。実GPU・学習済みUMA checkpointは使っていない。
実際の小型UMA、ASE最適化・MD、PyTorch自動微分、保存・再読込を実行した。
仮想XPU/CUDA試験はAPI選択・stream・設定復元等の検証であり、実GPU演算の代用ではない。

## 最終統合試験

B系列・準備後設定固定・既存runtime/設定/XPU dispatchの最終再試験は
**330 passed, 1 skipped**（186.38秒）。skipは実GPU必須ケース。
下記の再実行コマンドから`test_series_b_interop.py`を除く12ファイルを、2 workerで実行した。
個別試験の成功数はこの数へ重複加算していない。
追加の公開ラッパー試験 `test_series_b_interop.py` は **3 passed**（5.93秒）。
実際の小型UMAでBF16 CPUエミュレーション＋ASE BFGS、INT8＋ASE FIRE、
INT8＋TorchSim FIREを実行し、別のFP32計算への影響と入力・出力の所有権も確認した。
この3件ではcheckpoint取得境界だけを差し替え、モデル計算とoptimizerは実行している。

A系列の広域回帰試験は **1,183 passed, 38 skipped**（676.19秒）。
対象はdatasets/graph/ASE/TorchSim/server/normalizer/predictor/UMA等。
その後に追加した準備後設定固定の影響は上記330件の再試験でも確認した。
2つの範囲には重複があるため、成功数を足して独立ケース数とはしていない。
skipには実GPU必須ケースと既存の無効化されたモジュールが含まれる。

## 同一小型UMAでの数値比較

[機械可読の全結果](assets/intel_xpu_series_b_cpu_smoke.json)を保存した。
対象はランダム初期化した1-layerのUMA、H2O、FP32、有限の参照セル、CPU。
`compare_predictors`で同じ重みのFP32モデルと候補を比較し、energy/forces/stress/Hessianを
それぞれ要求した。低精度GEMMはCPUエミュレーションである。
形状・バッファ所有権・自動微分の確認用であり、実用UMAの化学的精度ではない。
独立評価間の通常のFP32丸め差も含まれるため、1e-8付近の差をB固有の誤差と断定しない。

| 設定 | 最大ΔE (eV) | 最大ΔF (eV/Å) | 最大Δstress (eV/Å³) | 最大ΔH (eV/Å²) |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 3.849e-5 | 5.224e-5 | 1.557e-7 | 8.374e-5 |
| FP16 | 1.637e-5 | 4.555e-6 | 1.054e-8 | 8.913e-6 |
| BF16 split2 | 6.333e-8 | 8.848e-9 | 1.728e-11 | 7.241e-8 |
| BF16 split3 | 1.863e-8 | 1.863e-9 | 4.547e-12 | 2.794e-9 |
| INT8 | 1.079e-4 | 3.243e-5 | 9.670e-8 | 1.346e-4 |
| INT4 | 6.587e-4 | 7.789e-4 | 2.322e-6 | 1.549e-3 |
| SVD fraction 0.25 | 5.785e-2 | 8.788e-3 | 2.620e-5 | 1.561e-2 |
| saved BF16 | 3.353e-8 | 1.318e-5 | 3.023e-8 | 2.165e-5 |
| 補償FP32集約 | 3.725e-8 | 2.328e-9 | 4.547e-12 | 3.492e-9 |
| 局所exp | 2.608e-8 | 1.863e-9 | 2.046e-12 | 6.519e-9 |
| Gaussian table 257 | 0 | 1.397e-9 | 1.413e-12 | 1.304e-8 |

測定時は別のテストも並行実行している。JSONに保存したCPUの時間比は動作確認用で、
速度順位やB70の高速化率として用いない。`accuracy_qualified`はすべてfalse。
特にINT4/SVDには実データでの評価や再学習が必要である。

B05の別の固定入力試験では、選択した保存特徴量が12,096→6,088 bytesになった。
これは選択テンソルだけの保存容量であり、モデル全体のVRAMやpeakメモリが半減した
という意味ではない。pack/unpackの一時領域・演算も発生する。

## アルゴリズム試験

- RESPA: 解析ポテンシャルをASE VelocityVerletと比較。teacher呼出し101回対501回、
  検証条件でのenergy driftは約3.40e-4 eV。任意の時間刻みでの安定性は保証しない。
- 粗探索→精密化: 同じ解析系でreference呼出し4回対12回、別途coarse呼出し14回。
  これは計算時間の倍率ではない。
- DyNEB: mask付きの見かけの収束を、maskなし残差0.743 eV/Åとして検出。
  reference精密化後の残差約0.000934 eV/Åを確認。
- Lanczos: 負固有値、縮退、残差、未収束、固定入力の実UMA HVPとの比較。
  乱数用CPU generatorは呼出し元のdefault deviceに影響されない。
- Hutchinson: trace/diagonal、標本分散・標準誤差、seedの分離、不定行列を検証。
- 蒸留: 解析teacherのE/F/stressを学習し、force-only損失の二階微分も確認。
  小型UMAを実際に2 step学習し、構造照合・checkpoint/EMAの保存再読込を試験。

## 検出して対処したB固有の不整合

- FP32出力のGEMM overloadに依存せず、入力微分・二階微分を別実装。
- INT4の入力幅3と4でpackedサイズが同じになる保存形式の衝突を、構造メタデータで検出。
- 量子化重みをFP32のままbackward用に保存してしまう問題を避け、packed重みから再計算。
- 準備後の近似設定変更による表示と実体の食い違いを拒否。
- ContextVarの入れ子呼出しからFP32 predictorへ補償集約が漏れないよう、無効設定も明示的にscope。
- NumPy整数設定のJSON/YAML型不整合を正規化。
- checkpointing/fast backendの明示overrideによる非対応組合せを拒否。
- ambient autocast、別スレッド、例外、copy/pickle、元テンソル変更時のhook検知を試験。

## 再実行

通常の開発依存関係に加え、CPU試験でASE/TorchSim/Sella等を使う既存のinterop環境を用いる。
全変更ファイルへpre-commitを実行する。テストはリポジトリのpytest設定を必ず指定する。
pytest-xdistがない場合は`-n 2`を省く。

```bash
CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/core/calculate/test_series_b_curvature.py \
  tests/core/calculate/test_series_b_interop.py \
  tests/core/calculate/test_series_b_distillation.py \
  tests/core/calculate/test_series_b_workflows.py \
  tests/core/models/uma/nn/test_approximate_linear.py \
  tests/core/models/uma/nn/test_approximate_ops.py \
  tests/core/models/uma/nn/test_approximate_saved_features.py \
  tests/core/scripts/test_compare_series_b.py \
  tests/core/units/mlip_unit/test_series_b_integration.py \
  tests/core/units/mlip_unit/test_predict_audit_round3.py \
  tests/core/units/mlip_unit/test_inference_runtime.py \
  tests/core/units/mlip_unit/test_inference_settings.py \
  tests/core/units/mlip_unit/test_xpu_predict.py \
  -c packages/fairchem-core/pyproject.toml -o addopts='' \
  -q -n 2 --disable-warnings --tb=short
```

A系列の広い試験範囲は[前回の実行コマンド](intel_xpu_interoperability_audit.md)に準拠し、
`test_predict_audit_round3.py`を追加した。ソケット制約で実行できない既知の分散14件と、
ダウンロードが必要なpretrained試験を除外し、実GPU必須ケースはskipする。
全変更ファイルでpre-commit全フックと`git diff --check`を通過した。
この範囲での成功を、分散実行・ドライバ・B70の保証に読み替えない。
