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
├── frechet_video_distance.py   # Google Research の FVD 参照実装（I3D による埋め込み）
└── data/                       # 評価用動画（Git 管理外）
    ├── fake/...                # 生成動画
    └── real/...                # 実動画
```

## データの置き方

`FVD/data/` の直下に、比較する 2 つの動画集合をそれぞれディレクトリとして置きます。
各ディレクトリ以下の動画ファイルは再帰的に読み込まれます。

- 各動画の先頭 **30 フレーム** を使用します（30 フレーム未満の動画は扱えません）
- 動画は **16 本ずつ** のバッチで処理され、16 で割り切れない端数は捨てられます
- フレームは 224×224 にリサイズされます

動画データはサイズが大きいため Git では管理していません。

## セットアップ

環境は [uv](https://docs.astral.sh/uv/) で管理しています（Python 3.12 も uv が用意します）。

```sh
uv sync
```

## 使い方

```sh
cd FVD
uv run python FVD.py
```

初回実行時に I3D モデルを TensorFlow Hub（`https://tfhub.dev/deepmind/i3d-kinetics-400/1`）からダウンロードします。

## 現状と既知の問題

数年前のコードのため、現在の環境ではまだ動きません。現在把握している問題:

- TF1 の `hub.Module` を使っているが、tensorflow-hub 0.16 で削除されている（TF2 の `hub.load` への移行が必要）
- `FVD.py` で平均・共分散をバッチごとに計算しており、ディレクトリ単位の統計になっていない

## 参考

- T. Unterthiner et al., "Towards Accurate Generative Models of Video: A New Metric & Challenges", arXiv:1812.01717, 2018.
- [google-research/frechet_video_distance](https://github.com/google-research/google-research/tree/master/frechet_video_distance)
  — `frechet_video_distance.py` はこの実装を元にしています（Apache License 2.0）。
