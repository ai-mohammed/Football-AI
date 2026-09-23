# Football AI

**Analyse vidéo de football, suivi des joueurs et cartographie du terrain.**

**[Ouvrir la démonstration Streamlit](https://football-ai-x.streamlit.app/)** · [Entraîner YOLO11](docs/TRAINING.md) · [Résultats et limites](reports/README.md)

Projet développé par **Mohammed ADDI**, à partir du toolkit open source Roboflow Sports.

Football AI extrait des pistes de joueurs, reconnaît leurs équipes, projette leurs positions sur le terrain et produit des statistiques exploratoires. L'application propose cinq extraits précalculés et l'analyse d'une vidéo importée.

## Ce qui fonctionne

| Fonction | Mise en œuvre |
| --- | --- |
| Détection | Modèles football pour les joueurs, gardiens, arbitres et ballon |
| Points du terrain | Modèle à 32 points ; filtrage par confiance et calibration RANSAC |
| Suivi individuel | ByteTrack ou BoT-SORT avec compensation du mouvement caméra et caractéristiques d'apparence, dans le mode Analyse par joueur |
| Équipes | Couleur du torse, clustering reproductible et vote temporel ; une attribution incertaine reste inconnue |
| Numéro de maillot | OCR optionnel, lectures répétées et contrôle des conflits |
| Ré-identification | Rapprochement équipe + numéro ; refus de fusionner des pistes dont les périodes de visibilité se chevauchent |
| Analyse | Trajectoires horodatées, heatmaps, distance observée, vitesse sur segments mesurables |
| Ballon | Temps de contrôle estimé, possession indéterminée, passes probables et réseau de passes |
| Export | Vidéo annotée et JSON contenant mesures, trajectoires et diagnostics |
| Entraînement | Audit des annotations, affinage YOLO11, comparaison avec un modèle de référence |

**Un ID de suivi n'est pas un numéro de maillot.** Sans lectures suffisamment cohérentes, le joueur conserve un libellé `ID…`. Deux équipes peuvent avoir un même numéro. Aucune limite artificielle à onze pistes ne masque les erreurs de suivi.

## Utiliser l'application

La [version en ligne](https://football-ai-x.streamlit.app/) affiche instantanément les exemples déjà calculés. Ces exemples proviennent du pipeline historique : ils ne démontrent pas les performances des nouvelles expériences YOLO11/BoT-SORT.

Pour utiliser ta carte graphique, depuis la racine du dépôt, dans un environnement Python 3.11 avec PyTorch adapté à ton GPU :

```bash
python -m pip install -r examples/soccer/requirements.txt
python -m pip install -e . --no-deps
python -m pip install easyocr
python -m streamlit run examples/soccer/streamlit_app.py
```

Dans **Analyser ma vidéo → Analyse par joueur**, choisir ByteTrack ou BoT-SORT. Les modèles football historiques manquants sont téléchargés au lancement ; BoT-SORT télécharge également son modèle d'apparence au premier usage.

Le mode complet est proposé lorsque CUDA est disponible. `FORCE_ENABLE_PLAYER_ANALYSIS=1` permet de l'essayer sur CPU. Le GPU du PC n'est pas accessible automatiquement depuis l'application hébergée.

Pour un traitement reproductible, avec les poids déjà téléchargés :

```bash
python scripts/analyze_match.py --video chemin/match.mp4 --output runs/mon-match --tracker botsort --device auto
```

Le dossier de sortie contient `annotated.mp4`, `preview.jpg` et `analysis.json`. Ajouter `--frames 100` pour un essai court, `--stride 2` pour échantillonner et `--no-ocr` pour désactiver la lecture des maillots.

## YOLO11 : entraîné, mesuré, encore expérimental

Un **YOLO11n-pose à 32 points** a été affiné localement : 20 époques initiales, puis 80 à partir du meilleur checkpoint. Sur les 30 images de validation disponibles, sa mAP50–95 des points atteint **0,743**, contre **0,925** pour le modèle historique.

Le modèle historique reste donc sélectionné par défaut. Le candidat YOLO11 n'a pas satisfait le contrôle géométrique sur le court extrait testé : une bonne métrique globale ne suffit pas pour obtenir des distances fiables. Les détails, résultats exportés et conditions de mesure sont dans [le rapport](reports/README.md).

L'entraînement est reproductible pour le terrain, le ballon et les joueurs. Seul le terrain a été entraîné dans cette expérience ; les jeux d'annotations et les poids ne sont pas versionnés. Voir [le guide](docs/TRAINING.md).

## Interpréter les résultats

- Les « contrôles » et « passes probables » sont des estimations de proximité entre ballon et pieds. Ce ne sont pas toutes les touches physiques ni des statistiques officielles.
- La possession est calculée sur le temps attribuable ; le temps inconnu est exposé séparément.
- Les distances sont limitées aux segments observés, calibrés et plausibles. Le terrain de référence mesure **120 × 70 m** ; il faut adapter ses dimensions au stade pour une interprétation métrique.
- Une vidéo télévisée ne montre pas tous les joueurs. Les occultations, ralentis, maillots similaires et coupures peuvent fragmenter les identités. La ré-identification reste imparfaite.
- BoT-SORT utilise ici des caractéristiques d'un classifieur générique. Ce n'est pas un modèle d'identité entraîné spécifiquement sur des footballeurs.
- Tirs, fautes, corners, formation tactique et reconnaissance des remplacements **ne sont pas encore implémentés**.

## Vérifier le projet

```bash
python -m unittest discover -s tests -v
```

Les tests couvrent notamment les collisions de maillots, la ré-identification, les coupures, les distances après fusion, la possession, la calibration, la validation des données et l'affichage Streamlit. Des essais GPU sur une vraie vidéo complètent ces tests ; ils ne constituent pas un benchmark de précision du tracking.

## Déploiement

Dans Streamlit Community Cloud, utiliser `examples/soccer/streamlit_app.py` comme fichier principal et choisir Python 3.11 dans les paramètres de déploiement. Les dépendances sont dans `examples/soccer/requirements.txt` et les paquets système dans `packages.txt`.

Le projet est accessible à **[football-ai-x.streamlit.app](https://football-ai-x.streamlit.app/)**. Les nouveaux checkpoints locaux ne sont ni téléversés ni activés automatiquement.

## Crédits

Base : [Roboflow Sports](https://github.com/roboflow/sports), par Piotr Skalski / Roboflow, sous [licence MIT](LICENSE). Les bibliothèques, poids et jeux de données conservent leurs licences respectives. Les modèles Ultralytics utilisent leur propre licence ; les annotations de terrain utilisées indiquent CC BY 4.0.
