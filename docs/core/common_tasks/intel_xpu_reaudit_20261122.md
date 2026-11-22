# Intel XPUフォーク再監査（2026年11月22日）

PR #3のマージコミット `4f97fbae` を基準に、推論・学習・軌跡・バッチ配備をまたぐ挙動を再検証した。FP32とFP64の比較では、誤った周期像、欠測ラベル、参照エネルギーの取り違えのように、修正前の値そのものが誤っていた場合を区別する。

| 対象 | 再現した問題と修正 | 代表的な回帰試験 |
| --- | --- | --- |
| 周期境界・近傍 | 座標の絶対値や非周期方向のセルベクトルに依存して近傍を誤判定することがあった。周期方向の局所座標で探索し、整数の像番号を元の座標系に戻す。3 Å周期のセルで0.25 Åと10⁸ Åにある2原子の距離0.75 Åを両方向で確認した。外部グラフでは遠方像を含む変位の微分も検査した。 | `test_periodic_topology_regressions.py`、`test_atoms_to_graphs.py`、`test_nvidia_graph_audit6.py` |
| 保存力・複数ヘッド | 同じバックボーンを共有する出力ヘッドが、後続ヘッドに必要な自動微分グラフを解放していた。力・応力・ヘッセ行列・ヘッセ行列ベクトル積を単独ヘッドと比較。デバイス変更後の自動混合精度設定と単原子の参照エネルギー切替も修正した。 | `test_precision_contracts.py`、`test_inference_runtime.py` |
| MoLE・Wigner変換 | NumPyから行数を変更した後の古いエキスパート割り当てを修正。チャンクに対応する区間の探索範囲を狭め、Wigner変換の3ブロックでは集約用インデックスを共用した。 | `test_moe_routing_memory.py`、`test_portable_fused_backend.py` |
| 学習データ・評価 | 連結データセットと部分集合で原子数・任意のメタデータの順序、重複、負のインデックスを保持。カスタム名称の力、行列形式の応力、欠測／固定原子などのマスクを修正。原子データの変換失敗時に中途半端な更新が残る問題と、蒸留時の余分な読み込み・コピーを修正した。 | `test_sampler_metadata_audit.py`、`test_task_label_masks.py`、`test_atomic_data_interop.py`、`test_series_b_distillation.py` |
| 評価指標の互換性 | 指定時のみ使う `rmse` は全有効成分の二乗誤差和を全件数で割り、最後に平方根を取る。バッチ境界とrank数に依存しない。従来のベクトルノルム平均は `mean_euclidean_error` で選択できる。旧 `rmse` の集計途中の値は計算方式が異なるため引き継げない。 | `test_evaluator.py`、`test_task_label_masks.py` |
| 分散サンプラー・起動 | 原子数メタデータがない場合の各設定、完了済みエポックの再開、バッチ数よりrank数が多い場合を修正。非シャッフル時は必要なrankの範囲を直接列挙。グラフ並列時はデータ並列グループ内で集約し、正規のパディング重複を保持する。Slurm環境欠落時には初期化エラーが呼び出し元に届く。 | `test_sampler_metadata_audit.py`、`test_distributed_subgroup_audit.py`、`test_slurm_rendezvous.py` |
| モデル取得 | 自動選択されたXPUでもダウンロード前に設定を検証する。Hubから取得したチェックポイント本体と付属の参照エネルギーを同じスナップショットに固定する。 | `test_xpu_pretrained.py`、`test_resolve_checkpoint.py` |
| ASE・MD・LAMMPS | セルのない分子の軌跡、Langevin動力学での固定原子、重なる重心拘束、ログ重複、最終ステップ後の再開を修正。新チェックポイントには熱力学ログのバイト位置を保存する。旧形式は引き続き読めるが、停止直前の余分なログ行を正確に復元できるのは新形式から。LAMMPSの力・ビリアルは複数MPIランクでも検査した。 | `test_md_runner.py`、`test_ase_interop_jobs.py`、`test_lammps_callback_contract.py` |
| ワーカー・Ray Serve | 辺数の過小申告に実測下限を適用。失敗したFutureが入力・中間テンソルを保持する問題を修正し、例外から原因と発生箇所を追跡可能にした。配備ごとの設定とモデル更新を分離。 | `test_batch_interoperability.py`、`test_batch_server.py`、`test_batch_server_live_cpu.py` |
| 回帰判定・計測 | 物性の単位、比較対象の識別子、全系列の非有限値、性能記録と実行記録の一致を検査。封入済み基準データを逐次検証し、クラスター別の統計を小さな一時配列で計算する。 | `test_measurement_integrity.py`、`test_xe_benchmark_evidence.py`、`test_end_to_end.py` |

## 精度比較で見つかった差

全ての周期軸に同じ原点移動を適用した試作では、近傍数上限を持つ結晶の応力と体積弾性率でFP32誤差が増えた。通常のセルでは元の近傍順序を保ち、長い格子軸と端点での丸めに対して座標の原点を調整した。CPU上の同一の小型UMAモデル・同一入力で、修正後の応力6成分と体積弾性率は `4f97fbae` のFP32値とビット単位で一致した。FP64基準との差は応力の平均絶対誤差が9.9953×10⁻¹¹ eV/Å³、体積弾性率の絶対誤差が8.2192×10⁻⁸ GPaだった。

## 周期セルの応力・力・ヘッセ行列

保存力型UMAでは、原子のCartesian座標を行に並べた行列を $R$、3本の格子ベクトルを行に並べた行列を $A$ とすると、力は $F=-\partial E/\partial R$ である。応力は、原子の分率座標を保って $R$ と $A$ の両方に同じ対称ひずみ $\epsilon$ を与えたときのエネルギー微分を、元のセル体積 $V=|\det A|$ で割って求める。通常の経路は位置とセルの偏微分を1回の逆伝播で求めて合成する。

$$
\sigma = \frac{1}{V}\operatorname{sym}\!\left(
R^\mathsf{T}\frac{\partial E}{\partial R}
+ A^\mathsf{T}\left.\frac{\partial E}{\partial A}\right|_{R}
\right),\qquad
\operatorname{sym}(M)=\frac{M+M^\mathsf{T}}{2}.
$$

大きく展開したFP32座標では、上式の2項が大きく逆符号になり、各勾配を丸めた後の足し合わせでは誤差を取り戻せない。この場合はエネルギーに入る各周期像つき変位 $d$ に同じ仮想ひずみを掛け、$d(I+\epsilon)$ を通じて $\partial E/\partial\epsilon$ を自動微分する。3 Å周期、体積108 Å³、$E=\tfrac12\sum_e |d_e|^2$ の2原子試験では、遠方像を使うと従来の $\sigma_{xx}$ が $-0.00347222$ eV/Å³になった。修正後は折り返した座標と同じ $+0.01041667$ eV/Å³を返す。

原子のヘッセ行列はセルを固定して $-F$ を座標で自動微分した $3N\times3N$ 行列で、格子自由度を含まない。対応タスクで `predict_untrained_hessian` を有効にするとASEの `calc.get_property("hessian", atoms)` から取得できる。既定では行を最大64行ずつまとめて計算する。特定方向の曲率だけなら、ヘッセ行列ベクトル積（HVP）を自動微分で直接求められる。近接リストの生成は離散的な操作で、微分は選択された周期像と近傍を使うエネルギーに対して行う。

通常の力、応力、原子ヘッセ行列、HVPの計算に座標やセルの有限差分は使わない。一方、[結晶フォノンのPhonopyレシピ](../../../src/fairchem/core/components/calculate/recipes/phonons.py)はスーパーセルを変位させて得た複数の力から力定数を求める。[弾性定数のレシピ](../../../src/fairchem/core/components/calculate/recipes/elastic.py)も複数のひずみで得た応力をフィットする。いずれも、各構造で応力や力を計算する段階とは区別する必要がある。

## CPUで測定した個別処理

数値は同一環境における修正前後の比較。サンプラーとモデル部品の時間は単一CPUスレッドで測定した。メモリ値はPythonの `tracemalloc` で追跡した割り当てである。

| 条件と測定対象 | 修正前 | 修正後 |
| --- | ---: | ---: |
| 1,500原子×60フレーム、座標・速度・力のParquet書き込み待ちメモリ | 41.95 MiB | 6.94 MiB |
| 50万サンプル・8ランク、シャッフル有、残り10件の再開（7回の中央値） | 19.08 ms | 2.70 ms |
| 4万構造を4分割する貪欲負荷分散（JIT初回を除く7回の中央値） | 125.53 ms | 5.87 ms |
| MoLEの1万構造から4行を選ぶチャンク処理（8×8、3エキスパート） | 2.259 ms | 0.034 ms |
| Wigner変換のフォールバック（16,384辺、500ノード、1チャネル） | 4.889 ms | 2.311 ms |
| 回帰比較の一時メモリ（96件×25,000値、8クラスター） | 74.30 MiB | 0.97 MiB |

MoLE・Wignerの上記測定では出力がビット単位で一致し、サンプラーは同じシードのPyTorch標準実装と順序が一致した。回帰比較のクラスター平均には加算順による最大8.67×10⁻¹⁹の差があった。CPUの個別測定値から、Arc Pro B70上の推論全体の速度・VRAM・数値誤差は推定していない。

上流の[バッチMoLEの速度報告](https://github.com/facebookresearch/fairchem/issues/2305)を参照してルーティングを確認した。上流報告のGPUと入力に対する速度は今回の測定範囲外。旧監査の対応表は[PR #3の監査記録](intel_xpu_audit_fixes_20261120.md)、実機での比較条件は[検証手順](intel_xpu_validation.md)に記載した。
