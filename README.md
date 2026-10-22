# Fréchet Video Distance (FVD)

生成動画と実動画の分布の近さを測る指標 **FVD (Fréchet Video Distance)** を計算するスクリプトです。
大学時代の研究（モーション生成の評価）で使っていたコードで、現在整備中です。

FVD は画像の FID を動画に拡張したもので、Kinetics-400 で学習した I3D ネットワークで各動画を特徴ベクトルに変換し、
2 つの動画集合の特徴分布（平均 μ・共分散 Σ）の間の Fréchet 距離を求めます。値が小さいほど 2 つの集合が近いことを表します。

```
FVD = ||μ_real − μ_fake||² + Tr(Σ_real + Σ_fake − 2 (Σ_real Σ_fake)^{1/2})
```

## 構成

```
FVD/
├── FVD.py                      # 実行スクリプト（動画の読み込み → 埋め込み → FVD 計算）
├── frechet_video_distance.py   # Google Research の FVD 参照実装を TF2 向けに修正したもの
└── data/                       # 評価用動画（Git 管理外）
    ├── fake/...                # 生成動画
    └── real/...                # 実動画
```

## データの置き方

実動画と生成動画を別々のディレクトリに置きます（場所は任意です。上の `data/` は一例です）。
各ディレクトリ以下の動画ファイルは再帰的に読み込まれます。

- 各動画の先頭 **30 フレーム** を使用します（`--num-frames` で変更可。足りない動画はスキップされます）
- フレームは 224×224 にリサイズされます
- 共分散行列（400×400）を安定して推定するため、各ディレクトリに数百本以上の動画を用意してください

動画データはサイズが大きいため Git では管理していません。

## セットアップ

環境は [uv](https://docs.astral.sh/uv/) で管理しています（Python 3.12 も uv が用意します）。

```sh
uv sync
```

## 使い方

```sh
cd FVD
uv run python FVD.py data/real data/fake
```

最後に `FVD: <値>` が表示されます。

| オプション | 既定値 | 説明 |
|---|---|---|
| `--num-frames` | 30 | 各動画の先頭から使うフレーム数 |
| `--batch-size` | 16 | I3D に一度に入力する動画数 |
| `--threads` | CPU コア数の半分 | 計算に使う CPU スレッド数 |

初回実行時に I3D モデルを TensorFlow Hub（`https://tfhub.dev/deepmind/i3d-kinetics-400/1`）からダウンロードします。

計算は CPU で行います。TensorFlow は既定で全コアを使い切るため、`--threads` で使用スレッド数を制限しています。
Apple M4 Pro（12 コア）での目安:

| `--threads` | CPU 使用率（ピーク） | 1 本あたり |
|---|---|---|
| 6（既定） | 約 600% | 約 0.7 秒 |
| 12 | 約 1050% | 約 0.35 秒 |

Apple Silicon の GPU（tensorflow-metal）は、I3D の 3 次元畳み込みが速くならず CPU より遅かったため使っていません。

### 動作確認

`data/real` から無作為に選んだ 200 本どうしと、`data/real` 200 本と `data/fake` 200 本で計算した結果です。
同じ分布どうしでは小さく、異なる分布では大きくなることを確認しています。

| 比較 | FVD |
|---|---|
| 実動画 vs 実動画 | 57.39 |
| 実動画 vs 生成動画 | 1165.46 |

## 参考

- T. Unterthiner et al., "Towards Accurate Generative Models of Video: A New Metric & Challenges", arXiv:1812.01717, 2018.
- [google-research/frechet_video_distance](https://github.com/google-research/google-research/tree/master/frechet_video_distance)
  — `frechet_video_distance.py` はこの実装を元にしています（Apache License 2.0）。
