"""Augmentations d'éclairage réalistes pour les photos de supports vierges.

Appliquées à la photo AVANT la pose de l'écriture simulée : l'encre suit ensuite
la luminosité locale (poser_ecriture.poser_mot), donc ombre et teinte restent
cohérentes entre le papier et l'encre.

Tirage (une photo sur trois reste brute) :
  temperature   balance des blancs ramenée vers 2700 K (lampe) à 7500 K (ciel couvert),
                par gains RVB d'un corps noir, appliqués en linéaire ;
  exposition    gain 0,55 à 1,45 en linéaire, écrêtage des hautes lumières ;
  ombre         ombre portée douce : polygone aléatoire (main, téléphone) entrant par
                un bord, flou gaussien large, assombrissement 25 à 60 % ;
  gradient      éclairage rasant : gradient linéaire d'une direction aléatoire, 0,55 à 1 ;
  faible        faible lumière : gain 0,2 à 0,45, bruit photonique (Poisson) et de
                lecture, bruit de chrominance, léger lissage de débruitage.
Plusieurs effets peuvent se cumuler ; les paramètres tirés sont renvoyés.
"""
import math

import numpy as np
from scipy import ndimage


def srgb_lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_srgb(x):
    x = np.clip(x, 0, 1)
    return 255.0 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def rvb_kelvin(t):
    """Couleur RVB (0-1) d'un corps noir à t kelvins (approximation de Tanner Helland)."""
    t = t / 100.0
    r = 1.0 if t <= 66 else min(1.0, 1.292936 * (t - 60) ** -0.1332047)
    g = (0.3900816 * math.log(t) - 0.6318414) if t <= 66 else 1.1298909 * (t - 60) ** -0.0755148
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else 0.5432068 * math.log(t - 10) - 1.1962541)
    return np.clip(np.array([r, g, b]), 0.05, 1.0)


def temperature(lin, t, ref=5500.0):
    g = rvb_kelvin(t) / rvb_kelvin(ref)
    g = g / g.mean()
    return lin * g


def ombre(shape, rng):
    h, w = shape
    bord = rng.integers(4)
    n = int(rng.integers(4, 8))
    prof = rng.uniform(0.2, 0.6)
    pts = []
    for _ in range(n):
        u, v = rng.uniform(0, 1), rng.uniform(-0.2, prof)
        pts.append({0: (u, v), 1: (1 - v, u), 2: (u, 1 - v), 3: (v, u)}[int(bord)])
    from PIL import Image, ImageDraw
    m = Image.new("L", (w, h), 0)
    hull = np.array(pts) * [w, h]
    c = hull.mean(0)
    hull = hull[np.argsort(np.arctan2(hull[:, 1] - c[1], hull[:, 0] - c[0]))]
    ImageDraw.Draw(m).polygon([tuple(p) for p in hull], fill=255)
    s = rng.uniform(0.02, 0.07) * max(h, w)
    m = ndimage.gaussian_filter(np.asarray(m, np.float32) / 255, s)
    force = rng.uniform(0.25, 0.6)
    return 1 - force * m, dict(bord=int(bord), force=round(float(force), 2), flou=round(float(s), 1))


def gradient(shape, rng):
    h, w = shape
    a = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    p = (xx / w - 0.5) * math.cos(a) + (yy / h - 0.5) * math.sin(a)
    p = (p - p.min()) / max(1e-6, p.max() - p.min())
    bas = rng.uniform(0.55, 0.9)
    return bas + (1 - bas) * p, dict(angle=round(math.degrees(a)), min=round(float(bas), 2))


def augmenter(rgb, rng, p_brut=1 / 3, t_min=2700.0, t_max=7500.0):
    """rgb : float32 HxWx3 (0-255). Renvoie (rgb augmenté, paramètres). t_min, t_max : bornes de la balance des
    blancs tirée (log-uniforme)."""
    if rng.random() < p_brut:
        return rgb, dict(brut=True)
    lin = srgb_lin(rgb)
    par = {}
    if rng.random() < 0.7:
        t = float(np.exp(rng.uniform(np.log(t_min), np.log(t_max))))
        lin = temperature(lin, t)
        par["temperature"] = round(t)
    faible = rng.random() < 0.2
    if faible:
        g = rng.uniform(0.2, 0.45)
    else:
        g = float(np.exp(rng.normal(0, 0.2))) if rng.random() < 0.7 else 1.0
        g = float(np.clip(g, 0.55, 1.45))
    lin = lin * g
    par["exposition"] = round(g, 2)
    if rng.random() < 0.35:
        m, po = ombre(lin.shape[:2], rng)
        lin = lin * m[..., None]
        par["ombre"] = po
    if rng.random() < 0.35:
        m, pg = gradient(lin.shape[:2], rng)
        lin = lin * m[..., None]
        par["gradient"] = pg
    if faible:
        k = rng.uniform(150, 600)  # photons par unité de linéaire : plus bas = plus bruité
        lin = rng.poisson(np.clip(lin, 0, 1) * k) / k
        lin = lin + rng.normal(0, rng.uniform(0.002, 0.01), lin.shape)
        chroma = ndimage.gaussian_filter(rng.normal(0, rng.uniform(0.004, 0.015), lin.shape), (2, 2, 0))
        lin = lin + chroma
        lin = ndimage.gaussian_filter(lin, (0.6, 0.6, 0))
        par["faible"] = dict(photons=round(float(k)))
    return lin_srgb(lin).astype(np.float32), par


def mesures(rgb):
    """Luminance moyenne (0-255) et température apparente (rapport R/B du papier clair, en kelvins approchés)."""
    lum = rgb.mean(2)
    clair = lum > np.percentile(lum, 70)
    r, b = rgb[..., 0][clair].mean(), rgb[..., 2][clair].mean()
    ratio = r / max(b, 1e-3)
    ts = np.exp(np.linspace(np.log(2000), np.log(10000), 200))
    rr = np.array([(rvb_kelvin(t)[0] / rvb_kelvin(t)[2]) / (rvb_kelvin(5500)[0] / rvb_kelvin(5500)[2]) for t in ts])
    t = float(ts[np.argmin(np.abs(rr - ratio))])
    return dict(luminance=round(float(lum.mean()), 1), temperature=round(t), r_b=round(float(ratio), 3))
