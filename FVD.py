import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import math
import pathlib
import cv2
import numpy as np
import tensorflow as tf
import frechet_video_distance as fvd
from scipy.linalg import sqrtm


def main(argv):
    del argv
    batch = 16

    mus = []
    sigmas = []

    dirs_path = pathlib.Path('data').glob('*')

    for dir_num, dir_path in enumerate(dirs_path):
        videos_path = pathlib.Path(dir_path.as_posix()).glob('**/*.*')
        videos_posix_path = [videos_path.as_posix() for videos_path in videos_path]

        lists_of_videos = lists_of_batched_videos(videos_posix_path, batch)

        videos_in_dir = []

        for list_num in range(len(lists_of_videos)):
            print(f'\n処理された動画の数: [{list_num}] {list_num*batch}/{len(lists_of_videos*batch)}')
            videos = videos_embedding(array_of_processed_video(lists_of_videos[list_num]), videos_in_dir)
        
        print(videos)

        
        #Calculates average and dispersion of videos in each directory    
        mu, sigma = calc_average_dispersion(videos)

        if dir_num == 0:
            mus.append(mu)
            sigmas.append(sigma)
        else:
            mus.append(mu)
            sigmas.append(sigma)

    print(mus)
    print(sigmas)

    fake_mu = mus[0]
    fake_sigma = sigmas[0]
    real_mu = mus[1]
    real_sigma = sigmas[1]

    cov_mean = sqrtm(real_sigma.dot(fake_sigma)).real
    fid = round(np.sum((real_mu - fake_mu) ** 2.0) +
                np.trace(real_sigma + fake_sigma - 2.0 * cov_mean), 2)
    print(f'FVD is: {fid}.')



def lists_of_batched_videos(videos_path, batch):
    array = []
    length = math.floor((len(videos_path) / batch))
    if len(videos_path) <= batch:
        del videos_path
    else:
        for i in range(length):
            array.append(videos_path[:batch])
            del videos_path[:batch]
    del videos_path
    return array


def array_of_processed_video(videos):
    array = []
    for video in videos:
        print(video)
        input_video = cv2.VideoCapture(video)

        frames = []
        for frame in range(30):
            ret, image = input_video.read()

            frames.append(image)

        video = np.array(frames)
        # print(video.shape)
        array.append(video)

        input_video.release()
    array = np.array(array)
    print(f'{array.shape}:{type(array)}')       # 動画が破損しているものを調べるためにプリントしている。
    return array


def videos_embedding(videos, array_embedded):
    """"""
    with tf.Graph().as_default():
        videos = tf.constant(videos)
        preprocess = fvd.create_id3_embedding(fvd.preprocess(videos, (224, 224)))

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())

            preprocess = preprocess.eval()
    array_embedded.append(preprocess)
    array_embedded = np.array(array_embedded)
    del preprocess
    return array_embedded


def calc_average_dispersion(videos):
    videos = np.reshape(videos, (-1, 400))
    mu = videos.mean(axis=0)
    sigma = np.cov(videos, rowvar=False)
    print(f'mu type {type(mu)}, sigma type {type(sigma)}')
    return mu, sigma


if __name__ == '__main__':
    tf.compat.v1.app.run(main)

