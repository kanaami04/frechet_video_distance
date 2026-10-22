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
├── frechet_video_distance.py   # Google Research の FVD 参照実装を PyTorch 向けに修正したもの
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

### 実動画の統計を保存して使い回す

実動画側の平均・共分散は一度計算すれば変わらないため、`.npz` に保存しておけば、生成動画を何度も評価するときに再計算を省けます。

```sh
# 実動画の統計を保存（最初の 1 回だけ）
uv run python FVD.py --save-stats data/real real_stats.npz

# 以降はディレクトリの代わりに .npz を指定
uv run python FVD.py real_stats.npz data/fake
```

`.npz` には計算時の `--num-frames` も保存され、異なる値で使おうとするとエラーになります。

| オプション | 既定値 | 説明 |
|---|---|---|
| `--num-frames` | 30 | 各動画の先頭から使うフレーム数 |
| `--batch-size` | 16 | I3D に一度に入力する動画数 |
| `--device` | 使える中で最速のもの | `mps`（Apple Silicon GPU）/ `cuda` / `cpu` |
| `--threads` | CPU コア数の半分 | `--device cpu` のときに使う CPU スレッド数 |

初回実行時に I3D モデル（約 50MB）をダウンロードし、`~/.cache/torch/hub/checkpoints/` に保存します。

Apple M4 Pro での目安（実動画 64 本 vs 生成動画 64 本）:

| `--device` | CPU 使用率（ピーク / 平均） | 1 本あたり |
|---|---|---|
| `mps`（既定） | 約 70% / 約 25% | 約 0.07 秒 |
| `cpu`（6 スレッド） | 約 530% / 約 260% | 約 0.5 秒 |

### I3D モデルについて

FVD の元実装は TensorFlow Hub の `deepmind/i3d-kinetics-400/1` を使います。
ここでは GPU（MPS / CUDA）で動かすため、[StyleGAN-V](https://github.com/universome/stylegan-v) が公開している同モデルの TorchScript 移植版を使っています。
TensorFlow Hub 版と同じ動画で埋め込みを比較し、差が相対 1.3×10⁻⁶ 以下で FVD の値も一致することを確認しています。

モデルファイルは次の順に取得を試み、SHA-256（`bec6519f…5860a5e2`）が一致することを確認します。
いずれも同一ファイルであることを確認済みです。

1. StyleGAN-V 配布元（Dropbox）: `https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1`
2. Hugging Face [flateon/FVD-I3D-torchscript](https://huggingface.co/flateon/FVD-I3D-torchscript)
3. Hugging Face [kiwhansong/DFoT](https://huggingface.co/kiwhansong/DFoT/tree/main/metrics_models)

すべて取得できなくなった場合は、DeepMind の [kinetics-i3d](https://github.com/deepmind/kinetics-i3d) にある元の重みから変換する必要があります。

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
- I. Skorokhodov et al., "StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2", CVPR 2022.
  — I3D の TorchScript 版の配布元です。
