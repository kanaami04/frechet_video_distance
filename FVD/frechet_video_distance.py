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
# PyTorch: the I3D model is a TorchScript port of the TF-Hub
# i3d-kinetics-400 module (published by StyleGAN-V), so it can run on Apple
# Silicon GPUs (MPS) and CUDA, and the Fréchet distance is computed with
# NumPy/SciPy instead of tensorflow_gan.

"""Minimal Reference implementation for the Frechet Video Distance (FVD).

FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

# The original StyleGAN-V link first, then Hugging Face mirrors of the same
# file (verified to have the same SHA-256).
I3D_URLS = [
    "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1",
    "https://huggingface.co/flateon/FVD-I3D-torchscript/resolve/main/i3d_torchscript.pt",
    "https://huggingface.co/kiwhansong/DFoT/resolve/main/metrics_models/i3d_torchscript.pt",
]
I3D_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"


def default_device():
  """Returns the fastest available device: mps (Apple Silicon), cuda or cpu."""
  if torch.backends.mps.is_available():
    return "mps"
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


def preprocess(videos, target_resolution, device):
  """Runs some preprocessing on the videos for I3D model.

  Args:
    videos: <uint8>[batch_size, num_frames, height, width, depth] The videos to
      be preprocessed, as a NumPy array in RGB with values in the range 0-255.
    target_resolution: (height, width): target video resolution
    device: the torch device to put the videos on.

  Returns:
    videos: <float32>[batch_size, depth, num_frames, height, width] in [-1, 1]
  """
  videos = torch.from_numpy(videos).to(device)
  batch_size, num_frames = videos.shape[:2]
  all_frames = videos.flatten(0, 1).permute(0, 3, 1, 2).float()
  if tuple(all_frames.shape[-2:]) != tuple(target_resolution):
    all_frames = F.interpolate(
        all_frames, size=target_resolution, mode="bilinear",
        align_corners=False)
  output_videos = all_frames.reshape(
      batch_size, num_frames, 3, *target_resolution).permute(0, 2, 1, 3, 4)
  return 2. * output_videos / 255. - 1


def load_i3d(device):
  """Loads the Inflated 3D Convolution network, downloading it on first use."""
  path = pathlib.Path(torch.hub.get_dir()) / "checkpoints" / "i3d_torchscript.pt"
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    _download_i3d(path)
  return torch.jit.load(str(path), map_location=device).eval()


def _download_i3d(path):
  """Downloads the I3D model from the first reachable URL in I3D_URLS."""
  errors = []
  for url in I3D_URLS:
    try:
      torch.hub.download_url_to_file(url, str(path), hash_prefix=I3D_SHA256)
      return
    except Exception as e:  # pylint: disable=broad-except
      errors.append(f"{url}: {e}")
  raise RuntimeError("Failed to download the I3D model:\n" + "\n".join(errors))


@torch.no_grad()
def create_id3_embedding(i3d, videos):
  """Embeds the given videos using the Inflated 3D Convolution network.

  Args:
    i3d: the model returned by load_i3d().
    videos: <float32>[batch_size, depth=3, num_frames, height=224, width=224].
      Expected range is [-1, 1].

  Returns:
    embedding: <float32>[batch_size, 400], the kinetics-i3d-400-logits layer.
  """
  assert videos.max() <= 1.001, "max value in frame is > 1"
  assert videos.min() >= -1.001, "min value in frame is < -1"
  return i3d(videos, rescale=False, resize=False,
             return_features=True).cpu().numpy()


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
