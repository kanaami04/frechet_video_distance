"""Computes the FVD between a directory of real videos and one of generated videos."""

import argparse
import os
import pathlib
import sys

import cv2
import numpy as np
import tensorflow as tf

import frechet_video_distance as fvd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('real_dir', type=pathlib.Path, help='実動画のディレクトリ')
    parser.add_argument('fake_dir', type=pathlib.Path, help='生成動画のディレクトリ')
    parser.add_argument('--num-frames', type=int, default=30, help='各動画の先頭から使うフレーム数')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=max(1, os.cpu_count() // 2),
                        help='計算に使う CPU スレッド数（既定: コア数の半分）')
    return parser.parse_args()


def limit_threads(threads):
    """TensorFlow と OpenCV が使う CPU スレッド数を制限する。TF の演算を実行する前に呼ぶこと。"""
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
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


def embed_directory(i3d, directory, num_frames, batch_size):
    """ディレクトリ内の全動画を I3D で埋め込み、[動画数, 400] で返す。"""
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
            videos = fvd.preprocess(np.stack(batch), (224, 224))
            embeddings.append(fvd.create_id3_embedding(i3d, videos))
            batch = []
            print(f'\r{directory}: {i}/{len(paths)}', end='', file=sys.stderr)

    print(file=sys.stderr)
    if skipped:
        print(f'{directory}: {skipped} 本をスキップしました', file=sys.stderr)
    if not embeddings:
        sys.exit(f'{directory} に使える動画がありません')
    return np.concatenate(embeddings)


def main():
    args = parse_args()
    limit_threads(args.threads)
    i3d = fvd.load_i3d()
    real = embed_directory(i3d, args.real_dir, args.num_frames, args.batch_size)
    fake = embed_directory(i3d, args.fake_dir, args.num_frames, args.batch_size)
    print(f'FVD: {fvd.calculate_fvd(real, fake):.2f}')


if __name__ == '__main__':
    main()
