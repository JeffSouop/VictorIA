#!/usr/bin/env python3
"""
train.py – Entraîne les modèles VictorIA une fois, hors interface Streamlit.

Usage:
    python train.py                    # entraînement réel (défaut)
    python train.py --force --refresh  # re-télécharger l'historique API
    python train.py --synthetic        # fallback debug (données factices)
    python train.py --seasons 2021 2022 2023 2024
    python train.py --competitions PL PD FL1 SA BL1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from training.pipeline import train_models


def main():
    parser = argparse.ArgumentParser(description="Entraîner les modèles VictorIA")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Réentraîner même si le cache existe",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-télécharger les matchs historiques depuis l'API",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Utiliser des données synthétiques (debug uniquement)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Nombre d'échantillons synthétiques (--synthetic)",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Saisons à récupérer (année de début, ex: 2021 2022 2023 2024)",
    )
    parser.add_argument(
        "--competitions",
        type=str,
        nargs="+",
        default=None,
        help="Codes compétition (PL PD FL1 SA BL1)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Sauter l'évaluation hold-out",
    )
    args = parser.parse_args()

    metrics = train_models(
        force=args.force,
        use_real=not args.synthetic,
        n_samples=args.samples,
        evaluate=not args.no_eval,
        competitions=args.competitions,
        seasons=args.seasons,
        refresh_data=args.refresh,
    )

    print("\n── Résultats ──")
    print(f"  Source              : {metrics.get('data_source', '—')}")
    if metrics.get("competitions"):
        print(f"  Compétitions        : {', '.join(metrics['competitions'])}")
    if metrics.get("seasons_used"):
        print(f"  Saisons             : {metrics['seasons_used']}")
    print(f"  Échantillons        : {metrics.get('n_samples', '—')}")
    if "accuracy" in metrics:
        print(f"  Accuracy (test)     : {metrics['accuracy']:.1%}")
        print(f"  Log loss            : {metrics['log_loss']:.4f}")
        print(f"  Baseline (majorité) : {metrics['baseline_accuracy']:.1%}")
        print(f"  Gain vs baseline    : {metrics['improvement_vs_baseline']:+.1%}")
        if metrics.get("eval_mode"):
            print(f"  Mode évaluation     : {metrics['eval_mode']}")
    print(f"  Modèles actifs      : {', '.join(metrics.get('active_models', []))}")
    print("\nLancez l'app : streamlit run app.py")


if __name__ == "__main__":
    main()
