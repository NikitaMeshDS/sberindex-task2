"""Render mathematical indices consistently for the research deck."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/formulas"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    formulas = {
        "forecast": r"$\widehat{y}_{i,t+h}=y_{i,t}\,\frac{p_{r,m(t+h)}}{p_{r,m(t)}}\,(1+g)^k$",
        "profile": r"$p_r=0.5\,p_{\mathrm{Russia}}+0.5\,p_{\mathrm{region}}$",
        "peer_residual": r"$z_{i,t}=\log\!\left(\frac{y_{i,t}}{\widehat{y}_{i,t}}\right)-\mathrm{median}_{j\ne i,\,r(j)=r(i)}\log\!\left(\frac{y_{j,t}}{\widehat{y}_{j,t}}\right)$",
    }
    for name, formula in formulas.items():
        fig = plt.figure(figsize=(12, 1.3))
        fig.text(
            0.02,
            0.5,
            formula,
            fontsize=31 if name != "peer_residual" else 26,
            va="center",
            color="#0C825D",
        )
        fig.savefig(
            OUT / f"{name}.png",
            dpi=220,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
