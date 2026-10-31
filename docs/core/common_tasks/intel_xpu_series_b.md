# Series B: opt-in approximations and audit follow-up

2026-10-31 / `intel-xpu-fp32`

B系列を既定で無効の実験機能として実装した。A系列のFP32設定は維持する。
この文書のCPU試験は実際のPyTorch・ASE演算を実行するが、学習済みUMAの
対象化学系での精度や、Arc Pro B70での速度を保証するものではない。
B09/B10には学習用の実装があり、学習済みの実用studentを同梱しているわけではない。
実装後の追加修正と検証は[ライブラリ間整合性再監査](intel_xpu_series_b_interoperability_audit.md)を参照。

## 今回のA系列の再監査

- MoLEの整合性エラー後、portable/PyTorch fast backendとXPU graph設定が
  未マージモデルへ残る問題を修正。明示的なbackbone overrideも整合させた。
- 名前付きpresetを独立したコピーとして返し、ジョブ間の設定汚染を防止。
- 準備後にmerge_mole等の設定だけを変更すると整合性検査と重みが食い違うため、
  出力選択以外の設定変更を検出して拒否。内部fallbackでは再構築する。
- calculator factoryのseed未転送、循環メタデータの比較、入れ子の独自オブジェクト内の
  Tensorの転送・stream追跡を修正。
- サーバーのdense Hessian要求を単一構造ずつ処理するよう修正。
- 周期境界のFP32桁落ちでゼロ距離になった辺の除去後、offsetと辺数を同時に更新。
  捨てたゼロノルムから二階微分へNaNが混入する経路も修正。
- キャッシュ付きgraphでlist/NumPyのpartitionを受け付けるよう修正。
- portable backendの統合試験が実際にはgeneralを使っていた問題を修正。
  正しくportableを実行すると再現した、FP64解析係数とFP32入力の型不一致も修正。
- 設定のYAML化でsetを通常のリストへ変換し、OmegaConfの直接読み込みを可能にした。

## 有効化

設定は`InferenceSettings.series_b`へ渡す。辞書と`SeriesBSettings`の両方を受け付ける。
`SeriesBSettings()`は全機能無効。構築後に元の辞書を書き換えても設定は変わらない。

```python
from dataclasses import replace

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu
from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings

settings = replace(
    inference_settings_xpu(),
    series_b=SeriesBSettings(gemm_precision="bf16"),
)
calc = FAIRChemCalculator.from_model_checkpoint(
    "uma-s-1p1",
    task_name="omol",
    device="xpu:0",
    inference_settings=settings,
)
# atoms.calc = calc
# atoms.get_forces()
# print(calc.predictor.approximation_manifest)
```

FP32へ戻すときは`series_b`を省いて新しいcalculator/predictorを作る。
準備済みモデルの設定だけを変更する操作は、重み・保存hook・設定表示の食い違いを
防ぐため拒否する。CPUでBF16/FP16を数値的に試す場合だけ
`low_precision_emulation=True`を指定する。これは丸めた入力をFP32で計算する参照実装で、
Intel XMXの実行や性能の再現ではない。
今回のB実装もUMA全体をSYCLへ書き直したものではなく、PyTorch演算とXPU backendを使う。

初版のモデル内BオプションはUMAの`general` backend・FP32 base precision・
単一worker・eager実行を対象とする。`compile`、`xpu_graph`、`nitro`、
`activation_checkpointing`との組合せを拒否し、backbone overrideによる迂回も確認する。
低ランク化には`merge_mole=True`が必要。組成変更時は`mole_cache_size>0`で別の
統合済みモデルを準備するか、predictorを作り直す。

設定は`ModelSpec`のモデルIDに含まれ、異なるB設定のサーバーモデルを区別する。
サーバー起動時に渡すpredictorも同じ設定で生成する。

```python
from fairchem.core import pretrained_mlip
from fairchem.core.components.batch_server import setup_batch_predict_server
from fairchem.core.units.mlip_unit.api.model_spec import ModelSpec

server_settings = replace(
    inference_settings_xpu(),
    merge_mole=False,
    series_b=SeriesBSettings(weight_bits=8),
)
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p1", device="cpu", inference_settings=server_settings
)
# 対応済みのサーバー環境で、その環境のdeployment設定を併せて指定する。
# handle = setup_batch_predict_server(predictor, deployment_config=deployment_config)
spec = ModelSpec("uma-s-1p1", device="cpu", inference_settings=server_settings)
```

ここでのサーバー設定・モデルID・バッチ処理はCPU/仮想試験を実施した。
既存の高水準Ray起動ヘルパーはXPUを明示的に拒否するため、B70では直接calculator／
predictorを使う。上のサーバー例はCPUのINT8設定である。RayのXPU割当・複数GPU
スケジューリングを新たにサポートしたものではない。
`ParallelMLIPPredictUnit`のB実行は明示的に拒否する。

TorchSimも起動時に同じ設定を指定できる。複数組成を扱う例ではMoLEの固定統合を無効にする。
`inference_settings`を省略すると従来どおり`"batch"`を使用する。

```python
from fairchem.core.calculate.torchsim_interface import FairChemModel

torchsim_settings = replace(
    inference_settings_xpu(),
    merge_mole=False,
    series_b=SeriesBSettings(weight_bits=8),
)
model = FairChemModel(
    "uma-s-1p1",
    device="xpu:0",
    task_name="omol",
    inference_settings=torchsim_settings,
)
```

## B01–B17の実装と範囲

| ID | 指定方法／API | 実装範囲と注意点 |
| --- | --- | --- |
| B01 | `gemm_precision="bf16"` | backbone内の通常LinearをBF16入力・FP32出力GEMMへ置換。backwardはFP32。geometry、norm、head、SO3 einsum等は元の計算を維持。 |
| B02 | `gemm_precision="fp16"` | B01と共通。FP16 castのoverflowと非有限出力を検出。 |
| B03 | `onednn_tf32=True` | XPUの`mkldnn.allow_tf32`を推論範囲内で設定・復元。CUDA設定を転用しない。対応APIがないbuildではエラー。 |
| B04 | `gemm_precision="bf16_split2"` / `"bf16_split3"` | 両入力を2/3成分へ分解し、全4/9積をFP32へ加算。Intel oneMKLの同名に近いモードと同一実装ではない。 |
| B05 | `saved_feature_dtype="bf16"` / `"fp16"` | Gate/SiLU/Sigmoid系で保存する特徴量だけを圧縮。`saved_feature_min_elements`で下限を指定。forward値は維持し、微分値には圧縮誤差が入る。 |
| B06 | `weight_bits=8` / `4` | 行ごとの対称量子化、INT4 nibble packing。入力微分を維持して必要時に復元。packed構造情報を照合。ネイティブ整数GEMMではなく、復元コストで遅くなる場合がある。 |
| B07 | `compensated_reduction=True` | node→systemのFP64集約をFP32 Neumaier pairwise集約へ変更し、公開出力dtypeをFP64へ戻す。スレッド／入れ子呼出しごとに分離。 |
| B08 | `radial_fast_exp=True` | Gaussianのexpだけを局所多項式近似。C2遷移と元関数へのfallback。rsqrt/trigやglobal fast-mathは変更しない。 |
| B09 | `collect_teacher_labels` / `distill_student` | E/F/stress教師データとstudent学習。出力単位・正規化を明示するadapter、入力所有権、有限値を検証。対象データは利用者が用意する。 |
| B10 | B09へ独立した小型studentを渡す | cutoff、lmax、channel等を別のモデル構築設定として学習。元checkpointの設定を切り詰めない。構造照合後に標準checkpointを保存。 |
| B11 | `run_respa` + `RESPAOptions` | 固定セル・拘束なしNVEのteacher−student補正RESPA。外側half kickと内側Verlet。評価数とenergy driftを返す。 |
| B12 | `run_coarse_refine` + `CoarseRefineOptions` | 粗いcalculatorで探索後、別reference calculatorで精密化。referenceの残差と収束を再判定。粗探索による別の極小への移動はあり得る。 |
| B13 | `lanczos_modes` + `LanczosOptions` | HVPによる最小代数値のRitz固有モード、再直交化、残差・未収束の報告。未探索の低固有値がないという証明ではない。 |
| B14 | `run_dyneb` + `DyNEBOptions` | serial DyNEB、任意のreference NEB精密化、最後は必ずmaskなしNEB力で再確認。 |
| B15 | `radial_table_points=257`等 | GaussianをC2 quintic Hermite tableで補間。原子間距離・cutoffは変えず、表の外は元関数。B08との同時指定は拒否。 |
| B16 | `low_rank_fraction=0.25`等 | MoLE統合後の通常Linearをtruncated SVDへ置換。演算数を減らさない行列はskipしてmanifestへ記録。量子化との同時指定は拒否。fine-tuning済みとは称さない。 |
| B17 | `hutchinson_estimates` + `HutchinsonOptions` | HVPからtrace／diagonalを推定し、標本分散・標準誤差・seedを記録。不定Hessianに対応し、負固有値数やTS認証には使わない。 |

B06/B16変換後のstate_dictは元のLinear構造と異なる。通常は元checkpointから起動時に
変換する。変換済みstate_dictを保存する場合は同じ変換設定・構造へ読み込む必要があり、
そのまま元構造の標準checkpointとして読み込めるわけではない。

Linear対象と未適用箇所、圧縮・SVDの情報は初回予測後の
`predictor.approximation_manifest`で確認する。headまで丸める全モデルautocastではない。
B05はforwardを`torch.func`/`vmap`で包む使い方を拒否するが、通常のUMA Hessianの
loop/vmap経路を試験している。

B01/B02はactivationを丸めるため、微分は丸め操作に対する代理微分となる。
二階微分が計算できることと、離散的に丸めたforwardの厳密な微分・NVE保存性は異なる。
FP32出力を要求しても、実機の蓄積・kernel選択までCPU試験では立証できない。
CUDAでは対応するreduced-precision reductionが有効な場合に拒否する。
B03の大域設定ロックはfairchem間を調整するもので、無関係な外部ライブラリの
同時実行まで制御しない。B03は専用プロセスで利用する。

## ワークフロー例

モデル内のB設定と、MD／最適化／学習のワークフローは独立した明示APIにした。
calculator factoryは毎回別のcalculatorを返す。referenceにはB無効のFP32を指定する。
RESPAには保存力を返すteacher/studentを使い、B01/B02/B05等を組み合わせる場合も
エネルギードリフトを別途検証する。報告される評価数はAPI呼出し数で、kernel起動数ではない。

```python
from ase import units
from fairchem.core.calculate.series_b_workflows import (
    CoarseRefineOptions,
    RESPAOptions,
    run_coarse_refine,
    run_respa,
)

# relaxed = run_coarse_refine(
#     atoms, coarse_factory=make_coarse_calculator,
#     reference_factory=make_fp32_calculator,
#     options=CoarseRefineOptions(coarse_fmax=0.1, reference_fmax=0.01),
# )
# trajectory = run_respa(
#     atoms, student_factory=make_student_calculator,
#     teacher_factory=make_fp32_calculator,
#     options=RESPAOptions(inner_timestep=0.2 * units.fs, inner_steps=2),
#     outer_steps=100,
# )
```

曲率は次のように必要な統計・固有モードだけを明示的に要求する。
`data`は単一構造の`AtomicData`。入力構造はアダプター内でコピーする。

```python
from fairchem.core.calculate.series_b_curvature import (
    HutchinsonOptions,
    LanczosOptions,
    PredictorHVP,
    hutchinson_estimates,
    lanczos_modes,
)

# hvp = PredictorHVP(predictor, data)
# modes = lanczos_modes(hvp, options=LanczosOptions(num_modes=3, max_iterations=64))
# stats = hutchinson_estimates(hvp, options=HutchinsonOptions(samples=64, seed=41))
```

## 実機比較

FP32基準と同じcheckpoint・同じ構造で比較するCLIを追加した。

```bash
python -m fairchem.core.scripts.compare_series_b \
  --checkpoint /path/to/model.pt --structure /path/to/structure.xyz \
  --task omol --device xpu:0 \
  --series-b '{"gemm_precision":"bf16"}' \
  --properties energy,forces,hessian --warmup 2 --repeats 5 \
  --measure-memory --output series_b_bf16.json
```

実際のCLIの`--help`で利用可能な引数を確認できる。JSONはファイルを`@path.json`で指定できる。
比較は最大絶対誤差、RMS、基準のnorm、時間、適用範囲を記録し、自動で科学的な合否を決めない。
計時は指定deviceを同期する。学習済みモデル、複数構造・サイズ・組成、対象温度・密度、
NVE drift、微小力、stress/Hessianを含む評価が必要。INT4やSVDを選ぶだけで
実用精度になるとは限らない。

## 検証結果と限界

最終の統合試験結果は [audit results](intel_xpu_series_b_validation.md) に記録する。
CPU上の小型・ランダム初期化UMAは呼出し・型・所有権・微分の検証に使い、
実材料の予測精度の代用にはしていない。RESPA等は解析ポテンシャルでも比較する。
B70でのnative FP16/BF16、oneDNN TF32、実機メモリ量と速度は未検証。
有効なデータで学習・評価したstudentや圧縮モデルも今回の成果物には含まれない。

既知のB70低精度GEMM報告には、NaNだけでなく有限な異常値も含まれる。
有限性チェックの通過だけを採用基準にしない。

## 一次資料

- [PyTorch mm FP32 output](https://docs.pytorch.org/docs/2.15/generated/torch.mm.html)
- [mm.dtypeの微分未実装に関するTorchtitan issue](https://github.com/pytorch/torchtitan/issues/4273)
- [B70 FP16/BF16 GEMMの報告](https://github.com/intel/compute-runtime/issues/1001)
- [Inductorのout_dtype報告](https://github.com/pytorch/pytorch/issues/200547)
- [oneMKL compute modes](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2024-1/compute-modes.html)
- [PyTorch saved tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)
- [ASE DyNEB](https://docs.ase-lib.org/ase/neb.html)
- [SciPy eigshの選択固有値](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
