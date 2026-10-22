"""Computes the FVD between a directory of real videos and one of generated videos.

Either side can be a .npz of precomputed statistics saved with --save-stats,
so the real videos only need to be embedded once.
"""

import argparse
import os
import pathlib
import sys

import cv2
import numpy as np
import torch

import frechet_video_distance as fvd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('real', type=pathlib.Path,
                        help='実動画のディレクトリか統計の .npz（--save-stats 時は入力ディレクトリ）')
    parser.add_argument('fake', type=pathlib.Path,
                        help='生成動画のディレクトリか統計の .npz（--save-stats 時は出力先の .npz）')
    parser.add_argument('--save-stats', action='store_true',
                        help='FVD を計算せず、1 つ目のディレクトリの統計を 2 つ目のパスに保存する')
    parser.add_argument('--no-live', action='store_true',
                        help='計算中にグラフのウィンドウを開かない')
    parser.add_argument('--num-frames', type=int, default=30, help='各動画の先頭から使うフレーム数')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default=fvd.default_device(),
                        help='計算に使うデバイス（mps / cuda / cpu。既定: 使える中で最速のもの）')
    parser.add_argument('--threads', type=int, default=max(1, os.cpu_count() // 2),
                        help='CPU で計算するときのスレッド数（既定: コア数の半分）')
    return parser.parse_args()


def limit_threads(threads):
    """PyTorch と OpenCV が使う CPU スレッド数を制限する。"""
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)


def find_videos(directory):
    return sorted(p for p in directory.glob('**/*') if p.is_file())


def read_video(path, num_frames):
    """先頭 num_frames フレームを RGB の [num_frames, H, W, 3] で返す。足りなければ None。"""
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        # OpenCV は BGR で読み込むが、I3D は RGB で学習されている
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < num_frames:
        return None
    return np.stack(frames)


def embed_directory(i3d, device, directory, num_frames, batch_size, on_batch=None):
    """ディレクトリ内の全動画を I3D で埋め込み、[動画数, 400] で返す。

    on_batch(ここまでの埋め込み, 処理した本数, 全本数) をバッチごとに呼ぶ。
    """
    paths = find_videos(directory)
    embeddings = []
    batch = []
    skipped = 0

    for i, path in enumerate(paths, 1):
        video = read_video(path, num_frames)
        if video is None:
            print(f'スキップ（{num_frames} フレーム未満か読み込み失敗）: {path}', file=sys.stderr)
            skipped += 1
        else:
            batch.append(video)
        if len(batch) == batch_size or (i == len(paths) and batch):
            videos = fvd.preprocess(np.stack(batch), (224, 224), device)
            embeddings.append(fvd.create_id3_embedding(i3d, videos))
            batch = []
            print(f'\r{directory}: {i}/{len(paths)}', end='', file=sys.stderr)
            if on_batch:
                on_batch(np.concatenate(embeddings), i, len(paths))

    print(file=sys.stderr)
    if skipped:
        print(f'{directory}: {skipped} 本をスキップしました', file=sys.stderr)
    if not embeddings:
        sys.exit(f'{directory} に使える動画がありません')
    return np.concatenate(embeddings)


class StatisticsLoader:
    """ディレクトリなら動画を埋め込んで、.npz なら保存済みの統計を読む。

    (mu, sigma, 動画ごとの埋め込み) を返す。古い .npz では埋め込みは None。
    """

    def __init__(self, args):
        self.args = args
        self.i3d = None

    def __call__(self, path, on_batch=None):
        if path.suffix == '.npz':
            return self.load(path)
        if self.i3d is None:
            print(f'デバイス: {self.args.device}', file=sys.stderr)
            self.i3d = fvd.load_i3d(self.args.device)
        embeddings = embed_directory(self.i3d, self.args.device, path,
                                     self.args.num_frames, self.args.batch_size, on_batch)
        return (*fvd.calculate_statistics(embeddings), embeddings)

    def load(self, path):
        stats = np.load(path)
        if int(stats['num_frames']) != self.args.num_frames:
            sys.exit(f'{path} は {int(stats["num_frames"])} フレームで計算された統計です'
                     f'（--num-frames {self.args.num_frames} と一致しません）')
        return stats['mu'], stats['sigma'], stats.get('embeddings')


def save_stats(args, load_statistics):
    if args.fake.suffix != '.npz':
        sys.exit('--save-stats の出力先は .npz にしてください')
    mu, sigma, embeddings = load_statistics(args.real)
    extra = {} if embeddings is None else {'embeddings': embeddings}
    np.savez(args.fake, mu=mu, sigma=sigma, num_frames=args.num_frames, **extra)
    print(f'統計を保存しました: {args.fake}')


def compute_fvd(args, load_statistics):
    mu_real, sigma_real, _ = load_statistics(args.real)
    mu_fake, sigma_fake, _ = load_statistics(args.fake)
    print(f'FVD: {fvd.frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake):.2f}')


def compute_fvd_live(args, load_statistics):
    """生成動画の埋め込みに合わせてグラフを更新しながら FVD を計算する。"""
    from live_view import LiveView  # matplotlib のウィンドウはこのモードでだけ使う

    view = LiveView(args.real.stem, args.fake.stem)
    real = load_statistics(args.real, on_batch=lambda _, done, total: view.progress(
        args.real.stem, done, total))
    view.set_real(*real)
    mu_fake, sigma_fake, fake = load_statistics(args.fake, on_batch=view.update)
    score = fvd.frechet_distance(real[0], real[1], mu_fake, sigma_fake)
    if fake is not None:
        view.update(fake, len(fake), len(fake), final=True)
    view.finish(score)
    print(f'FVD: {score:.2f}')
    view.show()


def main():
    args = parse_args()
    limit_threads(args.threads)
    load_statistics = StatisticsLoader(args)
    if args.save_stats:
        save_stats(args, load_statistics)
    elif args.no_live:
        compute_fvd(args, load_statistics)
    else:
        compute_fvd_live(args, load_statistics)


if __name__ == '__main__':
    main()
