<div align="center">

  <h1>⚽ Football AI</h1>

  <p><b>Analyse automatique de matchs de football à partir d'une simple vidéo</b> — détection,
  suivi, classification d'équipes, radar tactique, statistiques et cartographie, propulsés par
  la vision par ordinateur.</p>

  <p>
    <a href="https://football-ai-x.streamlit.app/"><b>🚀 Voir la démo en ligne</b></a>
    ·
    <a href="examples/soccer/README.md">📖 Documentation technique</a>
  </p>

</div>

## 👋 à propos

Ce projet transforme une vidéo de match de football en analyse tactique : détection des
joueurs, du ballon et de l'arbitre, suivi dans le temps, reconstruction de la position de
chaque joueur sur le terrain, et génération automatique de statistiques d'équipe et
individuelles — le tout visualisable en direct dans une interface Streamlit.

Il s'appuie sur le toolkit open-source [roboflow/sports](https://github.com/roboflow/sports)
(licence MIT), complété par un pipeline d'analyse par joueur, des statistiques d'équipe et
une application web complète.

## 🚀 démo en ligne

**[football-ai-x.streamlit.app](https://football-ai-x.streamlit.app/)**

La démo propose deux modes :
- **🎬 Exemples pré-calculés** — résultats déjà générés (vidéo annotée, stats, heatmaps,
  réseau de passes) sur plusieurs extraits, affichés instantanément.
- **📤 Analyser ma vidéo** — upload et traitement en direct de ta propre vidéo (plus lent
  sans GPU, l'hébergement gratuit n'en fournit pas).

## ✨ fonctionnalités

| catégorie | fonctionnalités |
|:----------|:-----------------|
| 🎯 détection & suivi | joueurs, gardiens, arbitres, ballon — avec ID de suivi persistant |
| 👕 équipes | classification automatique par équipe (SigLIP + clustering), sans annotation manuelle |
| 🗺️ tactique | radar terrain, heatmaps individuelles et collectives, réseau de passes, possession par équipe |
| 📊 par joueur | touches de balle, passes faites/reçues, distance parcourue, vitesse moyenne |
| 🔢 identification | lecture des numéros de maillot par OCR (mode GPU) |
| ☁️ déploiement | modèles téléchargés automatiquement, fonctionne sans GPU (en dégradé) |

## 🖥️ lancer en local

```bash
git clone https://github.com/ai-mohammed/Football-AI.git
cd Football-AI
pip install git+https://github.com/roboflow/sports.git
cd examples/soccer && pip install -r requirements.txt && cd ../..
streamlit run examples/soccer/streamlit_app.py
```

Les poids des modèles sont téléchargés automatiquement au premier lancement de chaque mode.
Détails complets (modes CLI, déploiement Streamlit Cloud, entraînement des modèles) dans
[`examples/soccer/README.md`](examples/soccer/README.md).

## 🧠 sous le capot

- [YOLO](https://docs.ultralytics.com/) — détection joueurs, ballon, terrain
- [Supervision](https://github.com/roboflow/supervision) — tracking, annotation, calcul de trajectoires
- [SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) + UMAP + KMeans — classification d'équipe non supervisée
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — lecture des numéros de maillot
- [Streamlit](https://streamlit.io/) — interface web

## 🗺️ pistes à venir

- Détection d'événements (tirs, corners, fautes)
- Tableau de bord avec vidéo synchronisée, timeline et graphiques en direct
- Hauteur du bloc défensif et reconnaissance de formation

## 🏆 crédits

Basé sur le toolkit [roboflow/sports](https://github.com/roboflow/sports) de Piotr Skalski /
Roboflow, sous licence [MIT](LICENSE).
