# Xeフォークの監査項目と回帰試験

対象は `intel-xpu-fp32` のコミット `030fc744` を基準にした75項目。L、P、G、D、MD、BA、A、Iは同日の監査番号に対応する。

| 監査番号 | 変更した動作 | 主な回帰試験 |
| --- | --- | --- |
| L1–L8 | LAMMPSの計算エラー伝播、固定セルのビリアル、入力スクリプトからの起動、単位検査、外部グラフ、MPIでの全系予測と原子IDによる力の配布、出力形状、電荷・多重度の検査 | `tests/lammps/test_lammps_callback_contract.py` |
| L9–L11 | ASEのタグ変更で結果を無効化。生成エネルギーと自由エネルギーを区別し、D3との合成では内側の計算器の状態変化も追跡する | `tests/core/calculate/test_ase_interop_jobs.py` |
| P1–P4 | 組成ごとの専門家の重みをforwardごとに保持。外部辺を系別に整列し、ヘッドの再選択とWignerキャッシュの勾配モード切替を修正 | `tests/core/models/uma/test_precision_contracts.py` |
| P5–P9 | ビリアルの原点依存誤差、無効なセル体積、カスタムGaussian展開、バッチに依存する旧pnormの扱い、融合Wigner逆伝播の中間勾配を修正 | 同上、`test_portable_fused_backend.py` |
| G1–G4 | 外部グラフでのグラフ並列、遠方の周期像の距離・微分、外部探索とモデル座標の精度、近傍数上限での縮退を扱う設定を修正 | `tests/core/graph/test_periodic_topology_regressions.py` |
| G5–G8 | 部分周期セルの物理的な軸を保持し、空の原子分割を各rankで一貫して扱う。BVHの葉の空間局在性と、セル探索の候補対数の上限を改善 | 同上、`test_graph_interop.py` |
| D1–D3 | 力のMSEで原子マスクを全座標成分へ展開。無効行は微分前に除外し、有効な非有限損失はエラーとする。既存の `per_structure` の重み付けを保持し、誤差ゼロの有効構造も数える `per_valid_structure` を追加 | `tests/core/modules/test_loss.py` |
| D4–D8 | ASE単一点ラベルの鮮度、応力6・9成分、DBと行を識別するSID、同位体質量の往復を修正。グラフの差し替えは形状・系の所属・本数を検証してから反映 | `tests/core/datasets/test_atomic_data_interop.py`、`test_ase_datasets.py` |
| D9–D13 | DB接続引数を分離。軌跡の範囲外選択にファイル名と長さを表示し、`out_of_range="skip"` に対応。蒸留では1シャードを再利用し、返すサンプルは独立。別ジョブと並行する場合は `distill_student(..., execution="process")` で乱数とロックを分離する。ASEで非ゼロ電荷・スピンを捨てる場合は警告 | 同上、`tests/core/calculate/test_series_b_distillation.py` |
| MD1–MD4 | 世代を切り替えてチェックポイントを公開。STOPFAIRで保存に失敗したらエラーを返し、拘束を復元。同じ出力先で再開した軌跡に既存フレームを引き継ぐ | `tests/core/components/test_md_runner.py` |
| MD5–MD8 | 欠測を持つParquet列、任意物性の欠測と計算失敗の区別、軌跡ごとの乱数状態、Langevinの重心拘束を修正 | 同上 |
| BA1–BA4 | ステージング対象のメタデータを検証し、共有テンソルの辞書キーと値を一貫して変換。直接submit時の待ち行列に上限を設け、バッチの空き探索を削減 | `tests/core/calculate/test_batch_interoperability.py` |
| BA5–BA8 | Rayのバッチ組立中のOOMも分割対象に含め、全試行に失敗した容量探索を明示的に失敗させる。配備削除時にモデル参照を解放し、同名配備のバッチ設定変更とモデル更新を直列化する | `tests/core/components/test_batch_server_interop.py` |
| A1–A5 | 回帰プロファイルの精度条件、封入済み基準、重みのSHA-256、逐次予測との物性差、同位体質量を検査 | `tests/core/regression_suite/test_xe_acceptance.py`、`test_end_to_end.py` |
| A6–A10 | 回転検査で運動量・拘束を扱い、XPUのコンパイルキャッシュ、同時実行の保存先、CPUの性能比較条件、ワーカーのエラー伝播を修正 | 同上、`test_settings.py` |
| I1–I4 | 部分グループの人数・通信相手を使用。明示デバイス番号の桁あふれを拒否。チェックポイント本体と参照エネルギーに同じキャッシュ指定を渡す | `tests/core/common/test_distributed_subgroup_audit.py`、`test_xpu_distutils.py`、`tests/core/calculate/test_xpu_pretrained.py` |
| I5–I8 | sdistにREADMEを収録し、Intel版とNumPy 1.26版のwheelに個別のローカルバージョン識別子を付与。Git管理外のwheelからローカル回帰を実行可能にし、日本語導入手順をIntel専用lockに統一 | `uv build packages/fairchem-core`、`uv build packages/fairchem-core-xpu`、`tests/core/scripts/test_xpu_package_metadata.py` |

精度等級を比較するときは、同じ重みと構造を使い、エネルギー、力、応力を個別に比較する。グラフの誤りや古いラベルを修正した入力では従来版との一致は合格条件にならない。CPUの回帰試験とIntel実機の測定条件は[検証手順](intel_xpu_validation.md)に記載している。
