"""Débruitage par seuillage dans une STFT ajustée, x̂ = Φ g(Φ* y).

Opérateurs ponctuels, Ŷ = Y h_θ(r), r = |Y| :

    soft      h_θ(r) = (1 - θ/r)_+
    hard      h_θ(r) = 1{r > θ}
    wiener    h_θ(r) = (1 - θ²/r²)_+                  (Wiener empirique)
    spectral  h_θ(r) = (1 - (θ/r)^α)_+^β              (soustraction spectrale)

Opérateurs sociaux [2, 3], Ŷ = Y h_θ(ρ), avec l'énergie voisine

    ρ²[τ,ν] = Σ_{Δτ,Δν} W[Δτ,Δν] |Y[τ+Δτ, ν+Δν]|²,   W ≥ 0, Σ W = 1 :

    pew       gain du Wiener empirique en ρ           (Persistent Empirical Wiener)
    wgl       gain du seuillage doux en ρ             (Windowed Group Lasso)

Niveau de débruitage : θ = λ θ_max(Y), θ_max = max_{τ,ν} r (ou max ρ),
calculé par signal. λ ∈ [0, 1] : λ = 0 ne touche à rien, λ = 1 annule tout.
σ n'intervient pas dans ces opérateurs.

Block thresholding de Yu, Mallat et Bacry [1] : gain constant par
bloc, a_i = (1 - κ(N_i)/u_i)_+, forme des blocs choisie par macrobloc 8 × 16 en
minimisant l'estimateur de risque de l'article (éq. 27). Aucun paramètre libre ;
demande la variance du bruit σ²[τ,ν].

Post-traitement de Wiener non diagonal (éq. 28 de [1]),
ouvert à toutes les méthodes : x̃ = Φ(ã ⊙ Φ* y), ã = |Φ* x̂|² / (|Φ* x̂|² + σ²[τ,ν]).

Coefficients complex64 de forme (..., N_τ, N_ν), trame puis fréquence. lam est
un scalaire.

Références
----------
[1] G. Yu, S. Mallat, E. Bacry, « Audio denoising by time-frequency block
    thresholding », IEEE Trans. Signal Processing 56(5), 2008.
[2] K. Siedenburg, M. Dörfler, « Persistent time-frequency shrinkage for audio
    denoising », J. Audio Eng. Soc. 61(1/2), 2013.
[3] M. Kowalski, K. Siedenburg, M. Dörfler, « Social sparsity! Neighborhood
    systems enrich structured shrinkage operators », IEEE Trans. Signal
    Processing 61(10), 2013.
"""

import math

import torch
import torch.nn.functional as F

__all__ = ["soft_gain", "hard_gain", "wiener_gain", "spectral_gain", "neighbour_energy",
           "soft", "hard", "wiener", "spectral_subtraction", "pew", "wgl",
           "block_thresholding", "noise_variance_map", "wiener_post_filter", "denoise",
           "METHODS"]


# ---------------------------------------------------------------------------
# Gains h_θ(r) à seuil absolu θ fixé
# ---------------------------------------------------------------------------
# Les branches écartées par torch.where sont évaluées sur r = 1 : aucune
# division par zéro, aucun NaN.

def soft_gain(r, threshold):
    """Seuillage doux, h_θ(r) = (1 - θ/r)_+."""
    support = r > threshold
    r_safe = torch.where(support, r, torch.ones_like(r))
    return torch.where(support, 1.0 - threshold / r_safe, torch.zeros_like(r))


def hard_gain(r, threshold):
    """Seuillage dur, h_θ(r) = 1{r > θ}."""
    return (r > threshold).to(r.dtype)


def spectral_gain(r, threshold, alpha=2.0, beta=1.0):
    """Soustraction spectrale généralisée, h_θ(r) = (1 - (θ/r)^α)_+^β, α, β > 0."""
    support = r > threshold
    r_safe = torch.where(support, r, torch.ones_like(r))
    u = torch.where(support, (threshold / r_safe) ** alpha, torch.zeros_like(r))
    return torch.where(support, (1.0 - u) ** beta, torch.zeros_like(r))


def wiener_gain(r, threshold):
    """Wiener empirique, h_θ(r) = (1 - θ²/r²)_+ : soustraction spectrale α = 2, β = 1."""
    return spectral_gain(r, threshold, alpha=2.0, beta=1.0)


# ---------------------------------------------------------------------------
# Opérateurs ponctuels, θ = λ max |Y|
# ---------------------------------------------------------------------------

def _check_lam(lam):
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam doit être dans [0, 1], reçu {lam}")
    return lam


def _threshold(amplitude, lam):
    """θ = λ max_{τ,ν} amplitude, par signal, de forme (..., 1, 1)."""
    return _check_lam(lam) * amplitude.amax(dim=(-2, -1), keepdim=True)


def soft(Y, lam):
    """Ŷ = Y (1 - θ/|Y|)_+, θ = λ max |Y|."""
    r = Y.abs()
    return Y * soft_gain(r, _threshold(r, lam))


def hard(Y, lam):
    """Ŷ = Y 1{|Y| > θ}, θ = λ max |Y|."""
    r = Y.abs()
    return Y * hard_gain(r, _threshold(r, lam))


def wiener(Y, lam):
    """Ŷ = Y (1 - θ²/|Y|²)_+, θ = λ max |Y|."""
    r = Y.abs()
    return Y * wiener_gain(r, _threshold(r, lam))


def spectral_subtraction(Y, lam, alpha=2.0, beta=1.0):
    """Ŷ = Y (1 - (θ/|Y|)^α)_+^β, θ = λ max |Y|."""
    r = Y.abs()
    return Y * spectral_gain(r, _threshold(r, lam), alpha, beta)


# ---------------------------------------------------------------------------
# Opérateurs sociaux, θ = λ max ρ
# ---------------------------------------------------------------------------

def _symmetric_pad(x, q, dim):
    """Prolonge x de q échantillons de chaque côté de dim, réflexion à bord
    répété : -1 ↦ 0, N ↦ N-1."""
    if q == 0:
        return x
    n = x.shape[dim]
    head = x.narrow(dim, 0, q).flip(dim)
    tail = x.narrow(dim, n - q, q).flip(dim)
    return torch.cat([head, x, tail], dim=dim)


def _check_kernel(kernel):
    """Noyau W float32 de tailles impaires, positif, poids central > 0, normalisé
    à somme 1. kernel[Q_τ + Δτ, Q_ν + Δν] = W[Δτ, Δν]."""
    kernel = torch.as_tensor(kernel, dtype=torch.float32)
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(f"noyau de forme (2Q_tau+1, 2Q_nu+1) attendu, reçu {tuple(kernel.shape)}")
    if bool((kernel < 0).any()) or not kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] > 0:
        raise ValueError("le noyau doit être positif, de poids central > 0")
    return kernel / kernel.sum()


def neighbour_energy(Y, kernel):
    """ρ[τ,ν] = sqrt(Σ W[Δτ,Δν] |Y[τ+Δτ, ν+Δν]|²), de même forme que Y."""
    kernel = _check_kernel(kernel)
    q_tau, q_nu = kernel.shape[0] // 2, kernel.shape[1] // 2
    n_tau, n_nu = Y.shape[-2:]
    energy = (Y.abs() ** 2).reshape(-1, 1, n_tau, n_nu)
    padded = _symmetric_pad(_symmetric_pad(energy, q_tau, -2), q_nu, -1)
    # conv2d est une corrélation : kernel[Q_τ+Δτ, Q_ν+Δν] pondère (τ+Δτ, ν+Δν).
    rho2 = F.conv2d(padded, kernel.to(Y.device)[None, None]).reshape(Y.shape)
    return torch.sqrt(rho2.clamp_min(0.0))


def pew(Y, lam, kernel):
    """Persistent Empirical Wiener, Ŷ = Y (1 - θ²/ρ²)_+, θ = λ max ρ."""
    rho = neighbour_energy(Y, kernel)
    return Y * wiener_gain(rho, _threshold(rho, lam))


def wgl(Y, lam, kernel):
    """Windowed Group Lasso, Ŷ = Y (1 - θ/ρ)_+, θ = λ max ρ."""
    rho = neighbour_energy(Y, kernel)
    return Y * soft_gain(rho, _threshold(rho, lam))


# ---------------------------------------------------------------------------
# Block thresholding de Yu, Mallat et Bacry [1]
# ---------------------------------------------------------------------------

# Table I de [1], p_res = 0,1 % : N_i -> κ.
OVERSUBTRACTION_TABLE = {4: 4.7, 8: 3.5, 16: 2.5, 32: 2.0, 64: 1.8, 128: 1.5}
MACROBLOCK_SHAPE = (8, 16)  # (P_τ^max, P_ν^max) : trames, bins
BLOCK_SHAPES = tuple((p_tau, p_nu) for p_tau in (8, 4, 2) for p_nu in (16, 8, 4, 2, 1))


def oversubtraction(n_coefs, width):
    """κ(N_i), Table I de [1]. Largeur 1 bin : table lue en 2 N_i ([1], §III-B). Hors
    des points de la table : interpolation linéaire en log2 N, bornée."""
    log2_n = torch.tensor([math.log2(n) for n in OVERSUBTRACTION_TABLE], device=n_coefs.device)
    table = torch.tensor(list(OVERSUBTRACTION_TABLE.values()), device=n_coefs.device)
    n_eff = torch.where(width == 1, 2 * n_coefs, n_coefs).to(torch.float32)
    x = torch.log2(n_eff).clamp(log2_n[0], log2_n[-1])
    k = (x - log2_n[0]).floor().long().clamp(max=len(table) - 2)
    t = x - log2_n[k]
    return (1 - t) * table[k] + t * table[k + 1]


def block_risk(u, kappa, n_coefs, noise_var_mean):
    """R̂_i, estimateur de risque du bloc (article, éq. 27) :

    R̂_i = σ̄²_i ( N_i + (κ² N_i - 2κ (N_i - 2)) / u_i · 1{u_i >= κ}
                      + N_i (u_i - 2) · 1{u_i < κ} ).
    """
    kept = u >= kappa
    u_safe = torch.where(kept, u, torch.ones_like(u))
    risk_kept = n_coefs + (kappa ** 2 * n_coefs - 2 * kappa * (n_coefs - 2)) / u_safe
    risk_killed = n_coefs + n_coefs * (u - 2)
    return noise_var_mean * torch.where(kept, risk_kept, risk_killed)


def _block_sums(z, p_tau, p_nu):
    """Sommes de z (..., T, F) sur des blocs disjoints P_τ × P_ν."""
    *batch, t, f = z.shape
    return z.reshape(*batch, t // p_tau, p_tau, f // p_nu, p_nu).sum(dim=(-3, -1))


def _expand(z, p_tau, p_nu):
    """Chaque valeur recopiée sur son bloc P_τ × P_ν."""
    return z.repeat_interleave(p_tau, dim=-2).repeat_interleave(p_nu, dim=-1)


def block_gains(Y, noise_var):
    """Gains a[τ,ν] de Yu et al., de même forme que Y.

    Pour chaque macrobloc 8 × 16, les 15 pavages P_τ ∈ {8, 4, 2}, P_ν ∈ {16, 8,
    4, 2, 1} sont essayés ; on garde celui dont Σ R̂_i est minimale. Sur un bloc
    B_i de N_i coefficients, u_i = Σ |Y|² / Σ σ² et a_i = (1 - κ(N_i)/u_i)_+.
    Pavage aligné sur l'origine ; blocs tronqués au bord de la grille.
    """
    energy = Y.abs() ** 2
    noise_var = torch.broadcast_to(noise_var, energy.shape).to(energy.dtype)
    n_tau, n_nu = energy.shape[-2:]
    p_tau_max, p_nu_max = MACROBLOCK_SHAPE
    pad_tau, pad_nu = -n_tau % p_tau_max, -n_nu % p_nu_max
    energy = F.pad(energy, (0, pad_nu, 0, pad_tau))
    noise_var = F.pad(noise_var, (0, pad_nu, 0, pad_tau))
    on_tau = F.pad(energy.new_ones(n_tau), (0, pad_tau))   # trames présentes
    on_nu = F.pad(energy.new_ones(n_nu), (0, pad_nu))      # bins présents

    risks, gain_maps = [], []
    for p_tau, p_nu in BLOCK_SHAPES:
        frames = on_tau.reshape(-1, p_tau).sum(-1)
        width = on_nu.reshape(-1, p_nu).sum(-1)
        n_coefs = frames[:, None] * width[None, :]          # N_i
        present = n_coefs > 0
        n_safe = torch.where(present, n_coefs, torch.ones_like(n_coefs))
        kappa = oversubtraction(n_safe, width.expand_as(n_safe))

        sum_energy = _block_sums(energy, p_tau, p_nu)
        sum_var = _block_sums(noise_var, p_tau, p_nu)
        sum_var_safe = torch.where(present, sum_var, torch.ones_like(sum_var))
        u = sum_energy / sum_var_safe
        gain = (1 - kappa / u.clamp_min(torch.finfo(u.dtype).tiny)).clamp_min(0)
        risk = torch.where(present, block_risk(u, kappa, n_safe, sum_var_safe / n_safe),
                           torch.zeros_like(u))
        risks.append(_block_sums(risk, p_tau_max // p_tau, p_nu_max // p_nu))
        gain_maps.append(_expand(gain, p_tau, p_nu))

    choice = torch.stack(risks).argmin(dim=0)               # un pavage par macrobloc
    index = _expand(choice, p_tau_max, p_nu_max)[None]
    gain = torch.gather(torch.stack(gain_maps), 0, index)[0]
    return gain[..., :n_tau, :n_nu]


def block_thresholding(Y, noise_var):
    """Ŷ = a ⊙ Y, block thresholding adaptatif de Yu et al. ([1], éq. 17, §III-C)."""
    return Y * block_gains(Y, noise_var)


# ---------------------------------------------------------------------------
# Variance du bruit et post-traitement de Wiener
# ---------------------------------------------------------------------------

def noise_variance_map(stft, sigma):
    """σ²[τ,ν] = E|(Φ* n)[τ,ν]|² pour un bruit blanc d'écart-type sigma.

    Vaut σ² c[ν]² Σ_j w[j]², somme restreinte aux échantillons de la trame τ qui
    tombent dans le signal. Tenseur (N_τ, N_ν) float32, CPU.
    """
    inside = F.pad(torch.ones(stft.signal_length), (stft.pad_left, stft.pad_right))
    frames = inside.unfold(-1, stft.win_length, stft.hop_length)          # (N_τ, N_w)
    window_energy = (frames * stft.window ** 2).sum(-1)                   # (N_τ,)
    return float(sigma) ** 2 * window_energy[:, None] * stft._scale[None, :] ** 2


def wiener_post_filter(stft, Y, x_hat, noise_var):
    """x̃ = Φ(ã ⊙ Y), ã = |Φ* x̂|² / (|Φ* x̂|² + σ²[τ,ν]) (éq. 28 de Yu et al.).

    Y : coefficients de l'observation, Φ* y. x_hat : estimée de la première étape.
    """
    energy = stft.analysis(x_hat).abs() ** 2
    return stft.synthesis(Y * (energy / (energy + noise_var.to(Y.device))))


# ---------------------------------------------------------------------------
# Débruiteur complet
# ---------------------------------------------------------------------------

METHODS = ("soft", "hard", "wiener", "spectral", "pew", "wgl", "yu")


def denoise(y, stft, method, lam=None, sigma=None, kernel=None, alpha=2.0, beta=1.0,
            post=False):
    """x̂ = Φ g(Φ* y), puis x̃ = Φ(ã ⊙ Φ* y) si post.

    method : l'un de METHODS. "yu" n'a pas de lam et demande sigma.
    lam : niveau dans [0, 1], pour toutes les méthodes sauf "yu".
    sigma : écart-type du bruit blanc ; requis pour "yu" et pour post=True.
    kernel : noyau W des méthodes sociales "pew" et "wgl".
    alpha, beta : exposants de "spectral".
    """
    Y = stft.analysis(y)
    if method == "yu":
        if sigma is None:
            raise ValueError("method='yu' demande sigma")
        Y_hat = block_thresholding(Y, noise_variance_map(stft, sigma).to(y.device))
    elif method in ("pew", "wgl"):
        if kernel is None:
            raise ValueError(f"method={method!r} demande un noyau kernel")
        Y_hat = (pew if method == "pew" else wgl)(Y, lam, kernel)
    elif method == "spectral":
        Y_hat = spectral_subtraction(Y, lam, alpha, beta)
    elif method in ("soft", "hard", "wiener"):
        Y_hat = {"soft": soft, "hard": hard, "wiener": wiener}[method](Y, lam)
    else:
        raise ValueError(f"method doit valoir l'un de {METHODS}, reçu {method!r}")
    x_hat = stft.synthesis(Y_hat)
    if not post:
        return x_hat
    if sigma is None:
        raise ValueError("post=True demande sigma")
    return wiener_post_filter(stft, Y, x_hat, noise_variance_map(stft, sigma))
