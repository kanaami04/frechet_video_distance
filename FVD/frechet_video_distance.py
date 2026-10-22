# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from the original (google-research/frechet_video_distance) to use
# TensorFlow 2: the I3D model is loaded with hub.load instead of the removed
# hub.Module, and the Fréchet distance is computed with NumPy/SciPy instead of
# tensorflow_gan.

"""Minimal Reference implementation for the Frechet Video Distance (FVD).

FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy import linalg

I3D_MODULE = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"


def preprocess(videos, target_resolution):
  """Runs some preprocessing on the videos for I3D model.

  Args:
    videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
      preprocessed. We don't care about the specific dtype of the videos, it can
      be anything that tf.image.resize accepts. Values are expected to be in
      the range 0-255.
    target_resolution: (height, width): target video resolution

  Returns:
    videos: <float32>[batch_size, num_frames, height, width, depth]
  """
  videos = tf.convert_to_tensor(videos)
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  resized_videos = tf.image.resize(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos


def load_i3d():
  """Loads the Inflated 3D Convolution network from tf.hub."""
  return hub.load(I3D_MODULE).signatures["default"]


def create_id3_embedding(i3d, videos):
  """Embeds the given videos using the Inflated 3D Convolution network.

  Args:
    i3d: the model returned by load_i3d().
    videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3].
      Expected range is [-1, 1].

  Returns:
    embedding: <float32>[batch_size, 400], the kinetics-i3d-400-logits layer.
  """
  tf.debugging.assert_less_equal(
      tf.reduce_max(videos), 1.001, "max value in frame is > 1")
  tf.debugging.assert_greater_equal(
      tf.reduce_min(videos), -1.001, "min value in frame is < -1")
  return i3d(videos)["default"].numpy()


def calculate_fvd(real_activations, generated_activations):
  """Computes the Fréchet distance between two sets of activations.

  Args:
    real_activations: <float32>[num_samples, embedding_size]
    generated_activations: <float32>[num_samples, embedding_size]

  Returns:
    The FVD as a float.
  """
  mu_real = real_activations.mean(axis=0)
  mu_gen = generated_activations.mean(axis=0)
  sigma_real = np.cov(real_activations, rowvar=False)
  sigma_gen = np.cov(generated_activations, rowvar=False)

  cov_mean = linalg.sqrtm(sigma_real.dot(sigma_gen)).real
  return float(np.sum((mu_real - mu_gen) ** 2) +
               np.trace(sigma_real + sigma_gen - 2. * cov_mean))
