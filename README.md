# Débruitage audio par seuillage temps-fréquence

Débruiteurs de la forme x̂ = Φ g(Φ* y) : analyse par une STFT ajustée,
seuillage des coefficients, synthèse. Bruit blanc gaussien. Écrit en PyTorch,
float32 ; tourne sur CPU, CUDA ou MPS (Apple).

## Contenu

| fichier | contenu |
|---|---|
| `stft.py` | classe `STFT` : `analysis` (Φ*), `synthesis` (Φ, adjointe), `inverse` ((Φ Φ*)⁻¹ Φ) ; trame ajustée par défaut (`tight=True`), où Φ Φ* = Id et l'inverse est la synthèse |
| `tfdenoise.py` | seuillages doux, dur, Wiener empirique, soustraction spectrale ; seuillages sociaux PEW et WGL ; block thresholding de Yu et al. ; post-traitement de Wiener ; fonction `denoise` |
| `denoising.ipynb` | notebook didactique : iSNR réglable en dB, niveau λ ∈ [0, 1], comparaison des méthodes, écoute |

## Installation

```bash
pip install -r requirements.txt
jupyter lab denoising.ipynb
```

Le notebook choisit le périphérique seul : CUDA, puis MPS, puis CPU.

## Utilisation

```python
import torch
from stft import STFT
import tfdenoise as tfd

N = 44_100
x = torch.randn(N)                       # signal propre, float32
sigma = 0.1
y = x + sigma * torch.randn(N)           # observation

stft = STFT(N, win_length=1024, hop_length=512)
x_hat = tfd.denoise(y, stft, "wiener", lam=0.02)
x_pew = tfd.denoise(y, stft, "pew", lam=0.02, kernel=torch.ones(3, 3))
x_yu = tfd.denoise(y, stft, "yu", sigma=sigma, post=True)
```

## Niveau de débruitage

Le seuil vaut θ = λ max |Φ* y| (ou λ max ρ pour PEW et WGL, ρ étant l'énergie
voisine). λ ∈ [0, 1] : λ = 0 ne touche à rien, λ = 1 annule tout. Le block
thresholding de Yu et al. n'a pas de paramètre libre et demande σ. Le
post-traitement de Wiener (`post=True`) s'applique à toutes les méthodes et
demande σ.

## Références

1. G. Yu, S. Mallat, E. Bacry, « Audio denoising by time-frequency block
   thresholding », *IEEE Trans. Signal Processing* 56(5), 2008.
2. K. Siedenburg, M. Dörfler, « Persistent time-frequency shrinkage for audio
   denoising », *J. Audio Eng. Soc.* 61(1/2), 2013.
3. M. Kowalski, K. Siedenburg, M. Dörfler, « Social sparsity! Neighborhood
   systems enrich structured shrinkage operators », *IEEE Trans. Signal
   Processing* 61(10), 2013.

## Licence

BSD 3 clauses, voir `LICENSE`.
