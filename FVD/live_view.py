"""FVD の計算中に開き、生成動画の埋め込みが進むたびに更新されるウィンドウ。"""

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse

import frechet_video_distance as fvd

# 日本語を表示できるフォントのうち、インストールされているものを使う
_JAPANESE_FONTS = ['Hiragino Sans', 'Noto Sans CJK JP', 'Yu Gothic', 'IPAexGothic']
_installed = {f.name for f in font_manager.fontManager.ttflist}
matplotlib.rcParams['font.family'] = [f for f in _JAPANESE_FONTS if f in _installed] + ['DejaVu Sans']

# 共分散の推定に最低限必要な本数。これ未満では FVD の推移を描かない
MIN_VIDEOS_FOR_FVD = 32
# FVD（行列の平方根）の再計算間隔
FVD_INTERVAL_SECONDS = 1.0


def fvd_terms(mu_real, sigma_real, mu_fake, sigma_fake):
    """FVD を「平均のずれ」と「ばらつきのずれ」の 2 項に分けて返す。"""
    mean_term = float(np.sum((mu_real - mu_fake) ** 2))
    return mean_term, fvd.frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake) - mean_term


class LiveView:
    """実動画の統計を基準に、生成動画の埋め込みの様子を描く。"""

    def __init__(self, real_label, fake_label):
        self.real_label = real_label
        self.fake_label = fake_label
        self.real = None
        self.started = time.time()
        self.last_fvd_time = 0.
        self.history = []  # (本数, 平均の項, ばらつきの項)

        plt.ion()
        self.fig = plt.figure(figsize=(14, 8), layout='constrained')
        self.fig.canvas.manager.set_window_title('FVD')
        grid = self.fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
        self.ax_scatter = self.fig.add_subplot(grid[:, 0])
        self.ax_fvd = self.fig.add_subplot(grid[0, 1])
        self.ax_hist = self.fig.add_subplot(grid[1, 1])
        self.fig.suptitle('実動画の統計を準備中…')
        self.refresh()

    def refresh(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def progress(self, label, done, total):
        """生成動画以外（実動画の埋め込み中など）の進捗をタイトルに表示する。"""
        self.fig.suptitle(f'{label}: {done}/{total} 本を埋め込み中')
        self.refresh()

    def set_real(self, mu, sigma, embeddings):
        """基準となる実動画の統計を設定し、散布図の軸（主成分）を決める。"""
        self.real = (mu, sigma)
        eigvals, eigvecs = np.linalg.eigh(sigma)
        order = np.argsort(eigvals)[::-1][:2]
        self.pc_axes = eigvecs[:, order]
        self.pc_vars = eigvals[order]
        self.explained = self.pc_vars / eigvals.sum()

        ax = self.ax_scatter
        if embeddings is not None:
            pts = self.project(embeddings)
            ax.scatter(pts[:, 0], pts[:, 1], s=4, c='0.7', alpha=0.4, linewidths=0,
                       label=f'{self.real_label}（{len(embeddings)} 本）')
            self.real_distances = self.distance_from_real_mean(embeddings)
        else:
            self.real_distances = None
        for k in (1, 2):
            ax.add_patch(Ellipse((0, 0), 2 * k * np.sqrt(self.pc_vars[0]),
                                 2 * k * np.sqrt(self.pc_vars[1]), fill=False,
                                 ec='0.35', ls='--', lw=1))
        self.fake_scatter = ax.scatter([], [], s=6, c='tab:red', alpha=0.6, linewidths=0,
                                       label=self.fake_label)
        ax.set_xlabel(f'主成分 1（分散の {self.explained[0]:.0%}）')
        ax.set_ylabel(f'主成分 2（分散の {self.explained[1]:.0%}）')
        ax.set_title('I3D 特徴の分布（破線: 実動画の 1σ・2σ）')
        ax.legend(loc='upper right', markerscale=3)
        lim = 3.5 * np.sqrt(self.pc_vars)
        ax.set_xlim(-lim[0], lim[0])
        ax.set_ylim(-lim[1], lim[1])
        self.refresh()

    def project(self, embeddings):
        return (embeddings - self.real[0]) @ self.pc_axes

    def distance_from_real_mean(self, embeddings):
        """各動画の特徴が実動画の平均からどれだけ離れているか。"""
        return np.linalg.norm(embeddings - self.real[0], axis=1)

    def update(self, embeddings, done, total, final=False):
        """生成動画の埋め込みが進むたびに呼ぶ。embeddings はここまでの全件。"""
        pts = self.project(embeddings)
        self.fake_scatter.set_offsets(pts)
        self.fake_scatter.set_label(f'{self.fake_label}（{len(embeddings)} 本）')
        self.ax_scatter.legend(loc='upper right', markerscale=3)

        now = time.time()
        if len(embeddings) >= MIN_VIDEOS_FOR_FVD and (
                final or now - self.last_fvd_time >= FVD_INTERVAL_SECONDS):
            self.last_fvd_time = now
            stats = fvd.calculate_statistics(embeddings)
            self.history.append((len(embeddings), *fvd_terms(*self.real, *stats)))
            self.draw_fvd()
            self.draw_hist(embeddings)

        if not final:
            eta = (now - self.started) / done * (total - done)
            current = sum(self.history[-1][1:]) if self.history else float('nan')
            self.fig.suptitle(f'{self.fake_label}: {done}/{total} 本  '
                              f'暫定 FVD = {current:.1f}  残り約 {eta / 60:.1f} 分')
        self.refresh()

    def finish(self, score):
        """最終的な FVD をタイトルに表示する。"""
        minutes = (time.time() - self.started) / 60
        self.fig.suptitle(f'FVD = {score:.2f}（{minutes:.1f} 分）')
        self.refresh()

    def draw_fvd(self):
        n, mean_term, cov_term = np.array(self.history).T
        ax = self.ax_fvd
        ax.clear()
        ax.stackplot(n, mean_term, cov_term, labels=['平均のずれ', 'ばらつきのずれ'],
                     colors=['tab:blue', 'tab:orange'], alpha=0.7)
        ax.plot(n, mean_term + cov_term, c='k', lw=1.5, label='FVD')
        ax.set_xlabel('処理した生成動画の本数')
        ax.set_ylabel('FVD')
        ax.set_title('FVD の推移（本数が少ないうちは大きめに出る）')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    def draw_hist(self, embeddings):
        ax = self.ax_hist
        ax.clear()
        fake = self.distance_from_real_mean(embeddings)
        upper = np.percentile(fake, 99)
        bins = np.linspace(0, upper * 1.1, 60)
        if self.real_distances is not None:
            ax.hist(self.real_distances, bins=bins, density=True, color='0.6', alpha=0.6,
                    label=self.real_label)
        ax.hist(fake, bins=bins, density=True, color='tab:red', alpha=0.5,
                label=self.fake_label)
        ax.set_xlabel('実動画の平均からの距離')
        ax.set_ylabel('割合')
        ax.set_title('1 本ごとの実動画の平均からの距離')
        ax.legend(loc='upper right')

    def show(self):
        """計算が終わった後、ウィンドウが閉じられるまで待つ。"""
        plt.ioff()
        plt.show()
