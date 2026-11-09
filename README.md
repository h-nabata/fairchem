<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/fair-chemistry-logo-dark-mode.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/fair-chemistry-logo-light-mode.png">
    <img src="docs/assets/fair-chemistry-logo-dark-mode.png" width="650" alt="FAIR Chemistry">
  </picture>
</p>

<h4 align="center">

![tests](https://github.com/facebookresearch/fairchem/actions/workflows/test.yml/badge.svg?branch=main&event=push)
![PyPI - Version](https://img.shields.io/pypi/v/fairchem-core)
![Static Badge](https://img.shields.io/badge/python-3.11%2B-blue)
[![codecov](https://codecov.io/gh/facebookresearch/fairchem/graph/badge.svg)](https://codecov.io/gh/facebookresearch/fairchem)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15587498.svg)](https://doi.org/10.5281/zenodo.15587498)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/facebookresearch/fairchem?quickstart=1)

</h4>

# `fairchem` by the FAIR Chemistry team

`fairchem` is the [FAIR](https://ai.meta.com/research/) Chemistry's centralized repository of all its data, models,
demos, and application efforts for materials science and quantum chemistry.

<a id="dependencies"></a>

## 主な依存関係

このフォークの通常版manifestに基づく範囲です。導入例・検証環境はPython 3.12を使用します。

| 用途 | ライブラリ／条件 |
| --- | --- |
| Python | `>=3.11,<3.15` |
| テンソル計算 | PyTorch `>=2.13,<2.16`。Intel用導入例は`2.15.0+xpu` |
| 数値計算 | NumPy `>=2.0,<2.5`、SciPy `>=1.15` |
| 原子シミュレーション | ASE `>=3.26`、e3nn `>=0.5` |
| 設定・学習基盤 | Hydra、OmegaConf、TorchTNT |
| 任意機能 | TorchSim用`[torchsim]`、Pymatgen等の`[extras]`、Ray用`[ray]` |

完全な依存定義は[pyproject.toml](packages/fairchem-core/pyproject.toml)を参照してください。
NumPy1.26互換版とIntel向けwheelの導入方法は下記ガイドに記載しています。

<a id="intel-xpu-guide"></a>

## Intel XPU対応フォークの導入・運用ガイド

このブランチ`intel-xpu-fp32`は、UMAをIntel GPU上のPyTorch XPUで利用するための
互換性修正、FP32／FP64の共通最適化（A系列）、明示指定する近似計算（B系列）を含みます。
upstreamのロゴ・タイトルは冒頭に、機能説明・導入例などの本文はこのガイドの後に残しています。
**Intelで最初に使用する設定は`device="xpu:0", inference_settings="xpu"`です。**
下部のupstream手順にある`pip install fairchem-core`だけでは、このフォークの修正は入りません。

2026年11月10日のFP64対応・最適化とupstream調査は
[FP64監査記録](docs/core/common_tasks/intel_xpu_fp64_audit.md)に記録しています。
今回の統合試験は1,864件成功・47件skip・失敗なしでした。
前回11月9日の21項目の修正と1,663件成功・44件skipの結果は
[前回の監査記録](docs/core/common_tasks/intel_xpu_final_audit.md)を参照してください。
CPUでの実計算とGPU制御経路の模擬試験を実施していますが、Arc Pro B70実機の速度、
VRAM使用量、配布UMAモデルによる長時間MDの安定性は未検証です。

### 目次

- [主な依存関係](#dependencies)
- [Intel XPU対応フォークの導入・運用ガイド](#intel-xpu-guide)
  - [実行方式と精度の区別](#fork-execution)
  - [Ubuntu・Arc Pro B70への導入](#fork-install)
  - [最初のFP32計算](#fork-fp32)
  - [FP64の利用と段階的な最適化](#fork-fp64)
  - [A系列の追加設定とB系列の明示指定](#fork-options)
  - [実機での受入試験とMD測定](#fork-validation)
- [UMA performance](#uma-is-now-much-faster)
- [Latest news](#latest-news)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Pretrained models](#instantiate-a-calculator-from-a-pretrained-model)
  - [Tasks](#set-the-task-for-your-application-and-calculate)
  - [Inorganic crystal relaxation](#relax-an-inorganic-crystal)
  - [Molecular dynamics](#run-molecular-dynamics-md)
  - [D3 dispersion corrections](#d3-dispersion-corrections)
  - [Multi-GPU inference and LAMMPS](#multi-gpu-inference-and-lammps)
- [License](#license)

<a id="fork-execution"></a>

### 実行方式と精度の区別

| 機能 | 実装・対象 | 運用上の扱い |
| --- | --- | --- |
| Intel FP32基準 | PyTorchのXPU演算、通常UMA backend | `xpu`設定。compile、TF32、B系列、Nitroは無効 |
| FP64基準 | CPU／CUDA、実演算検査を通るXPUの通常UMA backend | `fp64`／`xpu-fp64`設定。FP32専用カーネルは使用しない |
| A系列 | 正確な近傍再利用、MoLE統合、portable演算、出力選択など | 低精度化を加えない。演算順序による丸め差はあり得る |
| B系列 | BF16/FP16、量子化、低ランク化など | 既定で無効。対象構造で誤差と速度を比較して明示指定 |
| UMA Nitro | CUDA専用Triton/CuTeカーネルとCUDA Graph | Intel XPUでは動かない。厳密FP32基準とも異なる |

このフォークはUMA全体を独自SYCLカーネルへ書き直したものではありません。
Intel GPUへの実行は[PyTorch XPU backend](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html)
を通します。CUDA版Nitroとの性能差を示す実機比較値はまだありません。

<a id="fork-install"></a>

### Ubuntu・Arc Pro B70への導入

Python 3.12と、Intel GPUを認識するドライバー／ランタイムを用意してください。
[IntelのPyTorch 2.15向け前提条件](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-15.html)
では、B70を含むArc Pro BシリーズについてUbuntu 24.04／26.04とOMIX 0.4.0の組合せが
記載されています。対応するOSのポイントリリース、カーネル、GPUアクセス権は
[OMIX導入ガイド](https://dgpu-docs.intel.com/installation-guides/installing-omix.html)で確認してください。
公式PyTorch wheelを使う場合は、競合を避けるため`/opt/intel/oneapi/setvars.sh`をsourceしません。
別の作業でsource済みなら、source前の環境から新しい端末を開き、
シェル初期化ファイルによる自動sourceも無効にした状態で以下を実行します。

[uv](https://docs.astral.sh/uv/getting-started/installation/)とGitを導入したうえで、
既存のCUDA環境とは別の仮想環境を作ります。

```bash
git clone --branch intel-xpu-fp32 --single-branch https://github.com/h-nabata/fairchem.git
cd fairchem
uv venv --python 3.12 .venv-xpu
source .venv-xpu/bin/activate
uv pip install --python .venv-xpu/bin/python 'torch==2.15.0' \
  --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv-xpu/bin/python -e packages/fairchem-core
uv pip check --python .venv-xpu/bin/python
```

このブランチのPyTorch依存範囲は`>=2.13,<2.16`です。
既存の対応XPU wheelを先に導入すると、そのwheelを利用して依存関係を解決します。
`uv sync`の既定グループはCUDA向けなので、このXPU導入手順では使用しません。
Linuxでは共通依存関係としてCUDA／Warp系パッケージも導入されますが、通常のXPU経路では
Nitro／WarpなどのCUDA専用カーネルを読み込まずに実行します。
CUDA向けwheelへ入れ替わっていないか、次で確認します。

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
assert torch.xpu.is_available(), "Intel GPUを認識できていません"
for index in range(torch.xpu.device_count()):
    print(f"xpu:{index}", torch.xpu.get_device_name(index))
    x = torch.ones((64, 64), device=f"xpu:{index}", dtype=torch.float32)
    assert torch.isfinite(x @ x).all().item()
torch.xpu.synchronize()
PY
```

TorchSimを併用する場合は同じ環境で`uv pip install -e 'packages/fairchem-core[torchsim]'`、
Pymatgenなどを使う計算レシピは`uv pip install -e 'packages/fairchem-core[extras]'`を追加します。
NumPy 1.26が必須の既存環境には、Python 3.11／3.12向けの別manifest
`packages/fairchem-core-numpy126`があります。二つのmanifestを同時にインストールしないでください。
新規環境では上記の通常版を使用します。

<a id="fork-fp32"></a>

### 最初のFP32計算

配布モデルには[Hugging FaceのUMA利用許可](https://huggingface.co/facebook/UMA)が必要です。
許可取得後、仮想環境内で`hf auth login`を実行してください。
初回はモデルをダウンロードします。モデル名の代わりに、対応するローカルcheckpointも指定できます。

```python
from ase.build import bulk
from fairchem.core import FAIRChemCalculator

atoms = bulk("Si", cubic=True).repeat((2, 2, 2))
atoms.calc = FAIRChemCalculator.from_model_checkpoint(
    "uma-s-1p2p1",
    task_name="omat",
    device="xpu:0",
    inference_settings="xpu",
)
print("Energy [eV]:", atoms.get_potential_energy())
print("Forces [eV/Angstrom]:", atoms.get_forces())
print("Stress [eV/Angstrom^3]:", atoms.get_stress())
print("Effective settings:", atoms.calc.predictor.inference_settings)
```

`task_name`は計算対象に合わせます。上の例は無機材料向け`omat`で、分子は通常`omol`です。
電荷・スピンなどの指定は[calculatorの説明](https://facebookresearch.github.io/fairchem/ase-calculator/)
に従ってください。混在GPU機では自動選択がCUDAを優先する場合があるため、Intelには
必ず`xpu:0`などを明示します。B70が2枚ある場合、独立したjobを別プロセスで
`xpu:0`と`xpu:1`へ割り当てられますが、1計算で2枚のVRAMを自動合算する機能ではありません。
現状のRay Serve、InferenceBatcher、graph parallelによる複数XPU実行は未対応です。

<a id="fork-fp64"></a>

### FP64の利用と段階的な最適化

近傍cache、MoLE統合、folded GEMM、portable radial／Wigner演算、出力選択、
バッチ転送と出力分割などのA系列は、FP64でも共通の実装を利用します。
FP32専用の融合カーネルはFP64に適用せず、dtypeを保つ通常演算へ戻ります。
**`high-precision`はFP32設定です。FP64には`fp64`または`xpu-fp64`を明示してください。**

今回の改修では、XPUのFP64一律拒否を、対応機能と実際の演算を確認する検査へ変更しました。
Intelの公開Compute RuntimeにはBMGのFP64対応が記載されていますが、
ドライバーの能力宣言からB70上のUMA速度や全演算の成功までは判断できません。
まずモデルをダウンロードせずに確認します。

```bash
python -m fairchem.core.scripts.validate_xpu \
  --device xpu:0 --dtype float64 --probe-only --output xpu-fp64-probe.json
```

検査を通過した環境では、例えば次のように使用します。
CPUで比較基準を作る場合は`device="cpu", inference_settings="fp64"`に変更します。

```python
from fairchem.core import FAIRChemCalculator

calc = FAIRChemCalculator.from_model_checkpoint(
    "uma-s-1p2p1",
    task_name="omat",
    device="xpu:0",
    inference_settings="xpu-fp64",
)
# atoms.calc = calc
# atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
```

両FP64 presetはgeneral backend、eager、MoLE統合、TF32／Nitro／B無効です。
モデルの浮動小数点重みと入力をFP64に揃えます。FP32で保存されたcheckpointをFP64へ変換しても、
保存時に失った桁や学習誤差が回復するわけではありません。
TorchSimの`FairChemModel`は、`dtype`を省略するとpredictorの基底精度を引き継ぎます。
`dtype=torch.float64`だけを指定してもFP32モデルの演算精度は変わらないため、
モデルにも`inference_settings="fp64"`または`"xpu-fp64"`を指定してください。

大きい系では、まずA系列の`activation_checkpointing=True`を検討してください。
中間値を再計算するため時間との交換になりますが、計算精度を落とさずに保存量を削減します。
今回、chunkごとの不要なnode tensor保持と、エネルギーの微分から力を求める推論での
完了済み出力によるautograd graph保持を修正しました。
固定組成の近傍再利用や`requested_properties`もFP64で利用できます。

さらにメモリが不足するときだけ、B系列の保存活性圧縮を明示指定できます。

```python
from dataclasses import replace
from fairchem.core.units.mlip_unit.api.inference import inference_settings_fp64
from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings

settings_b64 = replace(
    inference_settings_fp64(),
    series_b=SeriesBSettings(saved_feature_dtype="fp32"),
)
# 新しいcalculatorのinference_settings=settings_b64へ渡す
```

この設定は、微分計算のため保存する対象の学習済み特徴をFP64からFP32へ圧縮します。
前向き演算と重み・幾何計算はFP64ですが、復元値を用いる力・stress・Hessian等は近似になります。
前向きのエネルギーが一致しても、力の微分整合性やNVEでのエネルギー保存は別途確認が必要です。
対象tensorの保存量はほぼ半分になる一方、圧縮・復元の時間が増えるため、高速化は保証しません。
FP64に対するB設定はこの方式だけを許可し、他の低精度GEMM・量子化等は拒否します。
活性化checkpointとB圧縮は併用せず、別設定として比較してください。
[FP64監査記録](docs/core/common_tasks/intel_xpu_fp64_audit.md)に精度比較とCPU測定条件を記載しています。

```bash
python -m fairchem.core.scripts.validate_xpu \
  --model uma-s-1p2p1 --device xpu:0 --dtype float64 --stress --compare-merged \
  --output xpu-fp64-validation.json

python -m fairchem.core.scripts.benchmark_md \
  --model uma-s-1p2p1 --device xpu:0 --inference-settings xpu-fp64 \
  --task omat --repeat 5 5 5 --steps 1000 --warmup 20 \
  --timestep-fs 0.1 --output md-xpu-fp64.json
```

検証の許容誤差はFP64指定だけでは厳しくなりません。`--energy-tolerance`、
`--force-tolerance`、`--stress-tolerance`と有限差分設定は、`validate_xpu`で用途に合わせて調整します。

B圧縮の比較には、取得済みcheckpointと対象構造を指定します。

```bash
python -m fairchem.core.scripts.compare_series_b \
  --checkpoint /path/to/model.pt --structure structure.extxyz --task omat \
  --device xpu:0 --dtype float64 --series-b '{"saved_feature_dtype":"fp32"}' \
  --properties energy,forces,stress --measure-memory --output fp64-b-comparison.json
```

<a id="fork-options"></a>

### A系列の追加設定とB系列の明示指定

固定組成のMDでは、移動量に応じて近傍候補を再利用する設定を追加できます。
再利用時も距離、物理cutoff、近傍上限は各回評価します。

```python
from dataclasses import replace
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu

settings = replace(
    inference_settings_xpu(),
    neighbor_skin=0.5,  # Angstrom; 候補数と再構築頻度のバランスを実測する
    neighbor_cache_entries=1,
    requested_properties=("energy", "forces"),  # 固定セルMD向け。NPTではstressも必要
)
# from_model_checkpoint(..., inference_settings=settings)へ渡す
```

compile、数値ブロックのXPU Graph、`umas_fast_portable`は追加の選択肢ですが、
まずeagerのFP32基準が通ることを確認してください。各設定と制約は
[A系列の仕様](docs/core/common_tasks/intel_xpu_series_a.md)に記載しています。
組成・task等を切り替える計算では、MoLE再準備／fallbackとキャッシュ保持量が増える場合があります。
キャッシュ上限は入力サイズによらない固定VRAM上限ではありません。

B系列を使用するときだけ、例えば以下のように新しい設定を作ります。

```python
from dataclasses import replace
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu
from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings

settings_b = replace(
    inference_settings_xpu(),
    series_b=SeriesBSettings(gemm_precision="bf16"),
)
# 新しいcalculatorのinference_settings=settings_bに渡す
```

BF16の例は低精度化を伴い、高速化も保証しません。準備済みpredictorの設定を書き換えて
FP32／Bを切り替えず、別インスタンスを作ってください。B系列とNitro、compile、
XPU Graphなどの未対応な組合せは拒否します。
[B系列の仕様・比較手順](docs/core/common_tasks/intel_xpu_series_b.md)を参照してください。

<a id="fork-validation"></a>

### 実機での受入試験とMD測定

まずエネルギー・力・stressと微分の整合性を確認します。
次のコマンドは配布モデルを実際に実行し、判定結果をJSONへ記録します。

```bash
python -m fairchem.core.scripts.validate_xpu \
  --model uma-s-1p2p1 --device xpu:0 --stress --compare-merged \
  --output xpu-validation.json

python -m fairchem.core.scripts.benchmark_md \
  --model uma-s-1p2p1 --device xpu:0 --inference-settings xpu \
  --task omat --repeat 5 5 5 --steps 1000 --warmup 20 \
  --timestep-fs 0.1 --sample-interval 25 --output md-xpu-fp32.json
```

MDの例はSi 1,000原子の固定セルNVEです。実際の対象には`--input structure.extxyz`を指定し、
task・温度・時間刻みを合わせてください。初回準備時間、step時間、エネルギーdrift、
PyTorch allocatorのallocated／reserved／peakを記録します。速度にはASE処理と検査・同期を含み、
reserved memoryの増加だけではメモリリークと判定できません。ドライバー等の確保量は別途観測が必要です。
短時間で有限値が得られることと、長時間の物理的安定性は別の検証です。

導入に失敗した場合は、`torch.__version__`、XPU認識結果、GPU／ドライバー情報、
使用した設定、完全な例外ログと上記JSONを保存してください。
検証範囲と残る制限は[最終監査記録](docs/core/common_tasks/intel_xpu_final_audit.md)にまとめています。

---

<!-- Upstream feature and usage sections follow. -->

## UMA is now much faster!

<p align="center">
  <a href="benchmarks/omol/README.md">
    <img src="benchmarks/omol/omol_force_mae_vs_speed.png" width="650" alt="OMol force MAE versus ASE NVE runtime">
  </a>
</p>

Uma-nitro brings blazing speedups to UMA. Currently it is only supported on Nvidia Ampere GPUs and above. Speed is heavily optimized on Hopper series and above for single-gpu single-system mode. On unsupported GPU types, the system will fallback to other non-nitro modes. Stay tuned as we bring nitro support for multi-gpu and batch modes. In the sub 5-10 ms/step regime, ASE integrators starts to become a bottleneck. To achieve the true performance capabilities of these models, we recommend moving to non-python/GPU based integrators.

The vertical axis is force MAE (meV/Å) on the public OMol25 validation set; the
horizontal axis is MD speed (ns/day at a 1 fs timestep) from the mean runtime of an ASE
NVE step on a 1000-atom silicon cell using an NVIDIA H200. See the
[benchmark details and reproduction scripts](benchmarks/omol/README.md)
for the full setup.

For more OMol model accuracy benchmarks, you can also see the
[FAIR Chemistry leaderboard](https://huggingface.co/spaces/facebook/fairchem_leaderboard).

Please use this logo for references to [UMA](docs/assets/uma-diagram-light-mode.png)!

## Latest news

### October 2026

- **UMA Nitro**, the new default inference mode, is much faster than turbo for molecular dynamics and relaxations on a single GPU.
  On a single NVIDIA H200, it reaches **~6 ns/day for 1,000 atoms** in ASE MD
  and raw inference throughput equivalent to **~33 ns/day for 150 atoms**,
  with support for systems up to **15,000 atoms**.
  Special thanks to Emanuele Arcelli and Max Zhdanov (for their [FlashUMA work](https://github.com/ArcelliEmanuele/flash-uma)), Carole-Jean Wu, Christian Puhrsch, Zachary DeVito, Aditya Venkataraman, Michael Lazos, and Animesh Jain, all of whom contributed to and helped improve UMA Nitro.
  See [UMA Nitro](https://facebookresearch.github.io/fairchem/ase-calculator#default-mode-nitro).
- **Two new UMA-S-1.2.1 models**: `uma-s-1p2p1-ft-d4rev` for molecules (ωB97M-D4rev) and
  `uma-s-1p2p1-ft-r2scan` for materials (r²SCAN, trained on [MAD-1.6](https://arxiv.org/abs/2603.02089)). These improve a wide range of observables compared to experiments. Please be aware of their [limitations](https://facebookresearch.github.io/fairchem/uma#uma-s-1p2p1-fine-tunes).
- **DFT-D3(BJ) dispersion (including GPU support)** for any ASE calculator, with PBE and r²SCAN presets. These make the respective materials tasks significantly more transferable outside of solid-state systems. See
  [D3 dispersion corrections](#d3-dispersion-corrections).

### September 2026

Read our new paper on
[quantum-accurate atomistic modeling of enzyme catalysis using a machine-learned potential](https://arxiv.org/abs/2609.09293).

<img src="https://gist.githubusercontent.com/rayg1234/bc9c41122ee5faa546b561923ec5d477/raw/75fdb26063309f31514a6ddf3ba39fd87776a3e7/ndpk_atp_to_adp_3dmol_1200px.gif" alt="NDPK-catalyzed ATP-to-ADP reaction" width="1000">

### June 2026

UMA playground released, try the interactive [educational demo](https://aidemos.atmeta.com/uma) here yourself!

<img src="https://gist.githubusercontent.com/rayg1234/bc9c41122ee5faa546b561923ec5d477/raw/b088b0eaf5f0253966458094afa3f6cac8f72b8d/uma_playground_demo.gif" alt="UMA playground demo" width="1000">

### March 2026

UMA-1.2 released! ~50% faster, ~40% more accurate on Open Molecules test set, and expanded data coverage for catalysts (oxides and interfaces), molecules, and polymers!

### October 2025

[check out our seamless Multi-node, Multi-GPU and LAMMPs interfaces to run large scale dynamics!](#multi-gpu-inference-and-lammps)

### June 2025

UMA/OMol first released! Read about the [UMA model and OMol25 dataset](https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/) release.

## Installation
Although not required, we highly recommend installing using a package manager and virtualenv such as [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer), it is much faster and better at resolving dependencies than standalone pip.

Install fairchem-core using pip
```bash
pip install fairchem-core
```

If you want to contribute or make modifications to the code, clone the repo and install in edit mode
```bash
git clone git@github.com:facebookresearch/fairchem.git

pip install -e fairchem/packages/fairchem-core[dev]
```

## Quick Start
The easiest way to use pretrained models is via the [ASE](https://wiki.fysik.dtu.dk/ase/) `FAIRChemCalculator`.
A single uma model can be used for a wide range of applications in chemistry and materials science by picking the
appropriate task name for domain specific prediction.

### Instantiate a calculator from a pretrained model
Make sure you have a Hugging Face account, have already applied for model access to the
[UMA model repository](https://huggingface.co/facebook/UMA), and have logged in to Hugging Face using an access token.
You can use the following to save an auth token,
```bash
huggingface-cli login
```

Models are referenced by their name, below are the currently supported models:

| Model Name | Description |
|---|---|
| uma-s-1p2p1 | Latest version of the UMA small model, fastest of the UMA models while still SOTA on most benchmarks (6.6M/290M active/total params) |
| uma-s-1p2p1-ft-d4rev | UMA-S-1.2.1 fine-tuned to ωB97M-D4rev for molecular systems; `omol` task only. See [new models](https://facebookresearch.github.io/fairchem/uma#uma-s-1p2p1-fine-tunes) |
| uma-s-1p2p1-ft-r2scan | UMA-S-1.2.1 fine-tuned on r²SCAN data ([MAD-1.6](https://arxiv.org/abs/2603.02089)) for inorganic materials and beyond; `omat` task only, not spin-polarized. See [new models](https://facebookresearch.github.io/fairchem/uma#uma-s-1p2p1-fine-tunes) |
| uma-m-1p1 | Best in class UMA model across all metrics, but slower and more memory intensive than uma-s (50M/1.4B active/total params) |

### Set the task for your application and calculate

- **oc20:** use this for catalysis
- **oc22:** use this for oxide catalysis (>1p2)
- **oc25:** use this for (electro)catalysis (>1p2)
- **omat:** use this for inorganic materials
- **omol:** use this for molecules+polymers
- **odac:** use this for MOFs
- **omc:** use this for molecular crystals

#### Relax an adsorbate on a catalytic surface,
```python
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")

# Set up your system as an ASE atoms object
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

slab.calc = calc

# Set up LBFGS dynamics object
opt = LBFGS(slab)
opt.run(0.05, 100)
```

#### Relax an inorganic crystal,
```python
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = bulk("Fe")
atoms.calc = calc

opt = FIRE(FrechetCellFilter(atoms))
opt.run(0.05, 100)
```

> [!WARNING]
> **Materials Project and OMat24 DFT settings:** OMat24-trained UMA and legacy
> inorganic models use DFT and DFT+U total-energy labels that are not compatible
> with Materials Project calculations. Use the OMat24 reference unary compounds
> and MP2020-style anion and GGA/GGA+U mixing corrections from the
> [OMat24 Hugging Face repository](https://huggingface.co/datasets/facebook/OMAT24).
> Do not apply Materials Project reference compounds or MP2020 corrections
> directly. Take care when comparing formation energies or energies above hull
> with Materials Project results: pseudopotentials differ, and magnetic ground
> states may differ as well.

#### Run Molecular Dynamics (MD)
Note: `pretrained_mlip.get_predict_unit()` currently uses a seed to set the global state of the numpy RNG. In order to obtain different trajectories for different runs of the following code, we have to set a random seed as shown below:
```python
import numpy as np
from ase import units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

seed = np.random.randint(0, np.iinfo(np.int32).max, dtype=int)
# the default inference mode (nitro) is the fastest for MD on a single GPU
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda", seed=seed)
calc = FAIRChemCalculator(predictor, task_name="omol")

atoms = molecule("H2O")
atoms.calc = calc

dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=300,
    friction=0.001 / units.fs,
)
trajectory = Trajectory("my_md.traj", "w", atoms)
dyn.attach(trajectory.write, interval=1)
dyn.run(steps=1000)
```

#### Calculate a spin gap,
```python
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda")

#  singlet CH2
singlet = molecule("CH2_s1A1d")
singlet.info.update({"spin": 1, "charge": 0})
singlet.calc = FAIRChemCalculator(predictor, task_name="omol")

#  triplet CH2
triplet = molecule("CH2_s3B1d")
triplet.info.update({"spin": 3, "charge": 0})
triplet.calc = FAIRChemCalculator(predictor, task_name="omol")

triplet.get_potential_energy() - singlet.get_potential_energy()
```

#### D3 dispersion corrections
`DFTD3Calculator` adds GPU-accelerated DFT-D3(BJ) energies, forces and analytic stresses (via `nvalchemiops`) to any
ASE calculator. For example, use `functional="pbe"` for PBE-trained heads and `functional="r2scan"` for `uma-s-1p2p1-ft-r2scan`.

```python
from fairchem.core import DFTD3Calculator, FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda")
calc = DFTD3Calculator(
    FAIRChemCalculator(predictor, task_name="omat"), functional="pbe", device="cuda"
)
```

See the [D3 docs](https://facebookresearch.github.io/fairchem/ase-calculator#adding-a-dft-d3-bj-dispersion-correction) for the
parameters and references.

#### Multi-GPU Inference and LAMMPs
If you have multiple gpus (or multiple nodes), we handle all the parallelism for you under the hood by a single flag (workers=N). Use `inference_settings="multigpu"` (formerly `turbo`) with `workers > 1`, since the default (nitro) runs on a single GPU. This is also compatible with LAMMPs to perform large scale MD. See our [docs](https://facebookresearch.github.io/fairchem/summary) for more details. This requires the Ray package to be installed and comes with the extras bundle.
```
pip install fairchem-core[extras]
```

```python
import time

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from fairchem.core import pretrained_mlip, FAIRChemCalculator

from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms

seed = np.random.randint(0, np.iinfo(np.int32).max, dtype=int)
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2p1",
    inference_settings="multigpu",
    device="cuda",
    workers=8,
    seed=seed,
)
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = get_fcc_crystal_by_num_atoms(8000)
atoms.calc = calc

dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=300,
    friction=0.001 / units.fs,
)
# warmup 10 steps
dyn.run(steps=10)
start_time = time.time()
dyn.attach(
    lambda: print(
        f"Step: {dyn.get_number_of_steps()}, E: {atoms.get_potential_energy():.3f} eV, "
        f"QPS: {dyn.get_number_of_steps()/(time.time()-start_time):.2f}"
    ),
    interval=1,
)
dyn.run(steps=1000)
```

### LICENSE
`fairchem` is available under a [MIT License](LICENSE.md). Models/checkpoint licenses vary by application area.
