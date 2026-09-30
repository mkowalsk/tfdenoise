"""STFT : analyse Φ*, synthèse Φ, inverse.

Signaux réels float32, coefficients complex64 des fréquences positives (rfft),
rangés en (..., N_τ, N_ν).

Pour la trame τ (fenêtre commençant à l'échantillon τ H du signal bourré),

    Y[τ, ν] = c[ν] sum_{j=0}^{N_w - 1} x_p[τ H + j] w[j] exp(-2iπ ν j / N_fft),

avec c[ν] = sqrt(2 / N_fft) pour 0 < ν < N_ν-1 et c[ν] = sqrt(1 / N_fft) aux
bins réels ν = 0 et ν = N_ν-1. La synthèse Φ est l'adjointe de l'analyse Φ*.

Comme N_w <= N_fft, l'opérateur de trame est diagonal en temps :

    (Φ Φ* x)[n] = E[n] x[n],    E[n] = sum_τ w[n - τ H]²,

et l'inverse (à gauche, de norme minimale) de l'analyse est Φ̃ = (Φ Φ*)^{-1} Φ,
soit la synthèse suivie d'une division par E. Avec une fenêtre ajustée
(tight=True), E = 1 : Φ Φ* = Id et l'inverse est la synthèse.
"""

import math

import torch
import torch.nn.functional as F

__all__ = ["STFT", "base_window", "tight_window"]


def base_window(name, win_length):
    """Fenêtre de base, float32.

    name : "sine", w[n] = sin(π (n + 1/2) / N_w), ou "hann", fenêtre de Hann
    périodique, w[n] = (1 - cos(2π n / N_w)) / 2.
    """
    n = torch.arange(win_length, dtype=torch.float32)
    if name == "sine":
        return torch.sin(math.pi * (n + 0.5) / win_length)
    if name == "hann":
        return 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / win_length)
    raise ValueError(f"fenêtre inconnue {name!r} : 'sine' ou 'hann'")


def _envelope(w, hop_length):
    """Enveloppe H-périodique E[q] = sum_j w[q + j H]², q in [0, H)."""
    envelope = torch.zeros(hop_length, dtype=torch.float32)
    for start in range(0, w.shape[0], hop_length):
        chunk = w[start:start + hop_length] ** 2
        envelope[:chunk.shape[0]] += chunk
    if envelope.min() <= 1e-6 * envelope.max():
        raise ValueError(f"l'enveloppe s'annule pour win_length={w.shape[0]}, "
                         f"hop_length={hop_length} : ce n'est pas une trame")
    return envelope


def tight_window(name, win_length, hop_length):
    """Fenêtre ajustée : sum_τ w[n - τ H]² = 1 pour tout n.

    La fenêtre de base est divisée par la racine de son enveloppe E.
    """
    w = base_window(name, win_length)
    envelope = _envelope(w, hop_length)
    return w / torch.sqrt(envelope[torch.arange(win_length) % hop_length])


class STFT:
    """Trame de Gabor, ajustée (Φ Φ* = Id) par défaut.

    Paramètres
    ----------
    signal_length : N, longueur des signaux.
    win_length : N_w, longueur de fenêtre.
    hop_length : H, pas d'avance ; win_length // 2 par défaut.
    n_fft : N_fft >= N_w, pair ; win_length par défaut.
    window : "sine" ou "hann".
    tight : True (défaut) pour la fenêtre ajustée, Φ Φ* = Id ; False pour la
        fenêtre de base, Φ Φ* = diag(E).

    L'opérateur suit le périphérique de son entrée : CPU, MPS ou CUDA.
    """

    def __init__(self, signal_length, win_length=1024, hop_length=None, n_fft=None,
                 window="sine", tight=True):
        hop_length = win_length // 2 if hop_length is None else hop_length
        n_fft = win_length if n_fft is None else n_fft
        if not 0 < hop_length <= win_length:
            raise ValueError(f"il faut 0 < hop_length <= win_length, reçu {hop_length}")
        if n_fft < win_length or n_fft % 2:
            raise ValueError(f"n_fft doit être pair et >= win_length, reçu {n_fft}")
        self.signal_length = signal_length
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.tight = tight
        if tight:
            self.window = tight_window(window, win_length, hop_length)
        else:
            self.window = base_window(window, win_length)
            _envelope(self.window, hop_length)  # vérifie que c'est une trame

        # Bords : N_w - H zéros à gauche, et à droite de quoi couvrir la fin.
        self.pad_left = win_length - hop_length
        self.n_frames = (self.pad_left + signal_length - 1) // hop_length + 1
        self.padded_length = (self.n_frames - 1) * hop_length + win_length
        self.pad_right = self.padded_length - self.pad_left - signal_length
        self.n_freqs = n_fft // 2 + 1
        # R = P_R / N, P_R = (2 N_ν - 2) N_τ coefficients réels.
        self.redundancy = (2 * self.n_freqs - 2) * self.n_frames / signal_length

        scale = torch.full((self.n_freqs,), math.sqrt(2.0 / n_fft), dtype=torch.float32)
        scale[0] = scale[-1] = math.sqrt(1.0 / n_fft)
        self._scale = scale

        # Diagonale E[n] de Φ Φ* : addition-recouvrement des w², restreinte au signal.
        frames = (self.window ** 2).expand(1, self.n_frames, win_length)
        self.frame_diag = self._overlap_add(frames)[0, self.pad_left:self.pad_left + signal_length]

    def __repr__(self):
        return (f"STFT(N={self.signal_length}, N_w={self.win_length}, H={self.hop_length}, "
                f"N_fft={self.n_fft}, tight={self.tight}) -> (N_tau={self.n_frames}, N_nu={self.n_freqs}), "
                f"R={self.redundancy:.3f}")

    def analysis(self, x):
        """Φ* : (..., N) float32 -> (..., N_τ, N_ν) complex64."""
        batch_shape = x.shape[:-1]
        window, scale = self.window.to(x.device), self._scale.to(x.device)
        x = F.pad(x.reshape(-1, self.signal_length), (self.pad_left, self.pad_right))
        frames = x.unfold(-1, self.win_length, self.hop_length) * window   # (B, N_τ, N_w)
        Y = torch.fft.rfft(frames, n=self.n_fft, dim=-1) * scale           # (B, N_τ, N_ν)
        return Y.reshape(*batch_shape, self.n_frames, self.n_freqs)

    def synthesis(self, Y):
        """Φ : (..., N_τ, N_ν) complex64 -> (..., N) float32, adjoint exact de analysis.

        Transposée de chaque étape de l'analyse, dans l'ordre inverse : échelle,
        irfft, troncature de la trame, fenêtre, addition-recouvrement,
        troncature à signal_length.
        """
        batch_shape = Y.shape[:-2]
        window, scale = self.window.to(Y.device), self._scale.to(Y.device)
        Z = Y.reshape(-1, self.n_frames, self.n_freqs) / scale
        frames = torch.fft.irfft(Z, n=self.n_fft, dim=-1)[..., :self.win_length] * window
        x = self._overlap_add(frames)[:, self.pad_left:self.pad_left + self.signal_length]
        return x.reshape(*batch_shape, self.signal_length)

    def inverse(self, Y):
        """Inverse de analysis, Φ̃ = (Φ Φ*)^{-1} Φ : synthèse, puis division par E[n].

        Avec une fenêtre ajustée, E = 1 à l'arrondi float32 près : l'inverse est
        la synthèse.
        """
        x = self.synthesis(Y)
        return x / self.frame_diag.to(x.device)

    def _overlap_add(self, frames):
        """Addition-recouvrement : (B, N_τ, N_w) -> (B, >= padded_length)."""
        batch, n_frames, _ = frames.shape
        hop = self.hop_length
        r = -(-self.win_length // hop)
        blocks = F.pad(frames, (0, r * hop - self.win_length)).reshape(batch, n_frames, r, hop)
        out = frames.new_zeros(batch, n_frames + r - 1, hop)
        for i in range(r):
            out[:, i:i + n_frames] += blocks[:, :, i]
        return out.reshape(batch, -1)
