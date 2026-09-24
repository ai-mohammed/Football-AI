# Football AI

**Analyse vidéo de football, suivi des joueurs et cartographie du terrain.**

**[Ouvrir la démonstration Streamlit](https://football-ai-x.streamlit.app/)** · [Entraîner YOLO11](docs/TRAINING.md) · [Résultats et limites](reports/README.md)

Projet développé par **Mohammed ADDI**, à partir du toolkit open source Roboflow Sports.

Football AI transforme un **extrait vidéo** en un espace d'analyse tactique : vidéo annotée, carte du terrain synchronisée, déplacements et transitions de contrôle. Cinq extraits analysés sur GPU sont disponibles immédiatement ; une vidéo personnelle peut aussi être importée.

## Un atelier tactique, centré sur l'extrait

- **Revoir une action** : lecture, ralenti, avance image par image et accès direct aux événements. La carte suit le temps de la vidéo, sans relancer les modèles.
- **Choisir une période** : les indicateurs, cartes, réseaux et exports sont recalculés sur la fenêtre sélectionnée.
- **Explorer le jeu** : chronologie du contrôle du ballon, occupation du terrain en 24 zones, réseau dirigé de passes probables, largeur et profondeur des joueurs visibles.
- **Étudier une piste** : mise en évidence sur le terrain, traces des deux dernières secondes, heatmap individuelle, comparaison de deux déplacements et courbes de vitesse estimée.
- **Vérifier les mesures** : temps indéterminé, couverture de calibration, ballon localisé et maillots confirmés. Export CSV des pistes/événements, JSON de la période et vidéo annotée de l'extrait complet.

La bibliothèque s'ouvre sans charger PyTorch ni télécharger de modèles. Les vidéos de démonstration sont encodées en H.264 720p et pèsent environ **2,4 à 2,8 Mo** chacune.

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
| Export | Vidéo annotée, CSV et JSON contenant mesures, positions horodatées et diagnostics |
| Entraînement | Audit des annotations, affinage YOLO11, comparaison avec un modèle de référence |

**Un ID de suivi n'est pas un numéro de maillot.** Sans lectures suffisamment cohérentes, le joueur conserve un libellé `ID…`. Deux équipes peuvent avoir un même numéro. Aucune limite artificielle à onze pistes ne masque les erreurs de suivi.

## Utiliser l'application

Dans la [version en ligne](https://football-ai-x.streamlit.app/), choisir **Extraits analysés**, puis un des cinq passages de douze secondes. Ils ont été recalculés avec **BoT-SORT, OCR et les détecteurs football historiques**. Ce ne sont pas des démonstrations du candidat YOLO11 expérimental.

Pour utiliser ta carte graphique, depuis la racine du dépôt, dans un environnement Python 3.11 avec PyTorch adapté à ton GPU :

```bash
python -m pip install -r examples/soccer/requirements.txt
python -m pip install -e . --no-deps
python -m pip install easyocr
python -m streamlit run examples/soccer/streamlit_app.py
```

Dans **Importer une vidéo**, choisir le début et la durée du passage. Les réglages permettent de sélectionner ByteTrack ou BoT-SORT, l'échantillonnage et l'OCR lorsqu'il est installé. Les modèles football manquants sont téléchargés uniquement au lancement de l'analyse ; BoT-SORT télécharge également son modèle d'apparence au premier usage.

L'interface limite les traitements à **30 secondes avec CUDA**, **8 secondes sur CPU**. Le CPU est automatiquement utilisé si aucun GPU compatible n'est présent. L'analyse complète peut prendre plusieurs minutes et n'est pas temps réel ; utiliser les démos pour une présentation fluide. Le GPU du PC n'est pas accessible automatiquement depuis l'application hébergée. Les fichiers temporaires d'import sont nettoyés après traitement ; le résultat reste dans la session Streamlit.

Pour un traitement reproductible, avec les poids déjà téléchargés :

```bash
python scripts/analyze_match.py --video chemin/match.mp4 --output runs/mon-match --tracker botsort --device auto
```

Le dossier de sortie contient `annotated.mp4`, `preview.jpg` et `analysis.json`. Ajouter `--frames 100` pour un essai court, `--stride 2` pour échantillonner et `--no-ocr` pour désactiver la lecture des maillots.

Pour reconstruire les cinq démos à partir des vidéos présentes dans `examples/soccer/notebooks` :

```bash
python scripts/build_demos.py --output-dir runs/mes-demos --device cuda --seconds 12 --stride 2
```

La commande écrit dans un nouveau dossier pour préserver les exemples publiés. Après vérification, remplacer les fichiers de chaque extrait dans `examples/soccer/demo_data`. Le format de données et les règles de calcul sont décrits dans [le guide du tableau de bord](docs/TACTICAL_WORKSPACE.md).

## YOLO11 : entraîné, mesuré, encore expérimental

Un **YOLO11n-pose à 32 points** a été affiné localement : 20 époques initiales, puis 80 à partir du meilleur checkpoint. Sur les 30 images de validation disponibles, sa mAP50–95 des points atteint **0,743**, contre **0,925** pour le modèle historique.

Le modèle historique reste donc sélectionné par défaut. Le candidat YOLO11 n'a pas satisfait le contrôle géométrique sur le court extrait testé : une bonne métrique globale ne suffit pas pour obtenir des distances fiables. Les détails, résultats exportés et conditions de mesure sont dans [le rapport](reports/README.md).

L'entraînement est reproductible pour le terrain, le ballon et les joueurs. Seul le terrain a été entraîné dans cette expérience ; les jeux d'annotations et les poids ne sont pas versionnés. Voir [le guide](docs/TRAINING.md).

## Interpréter les résultats

- Les « contrôles » et « passes probables » sont des estimations de proximité entre ballon et pieds. Ce ne sont pas toutes les touches physiques ni des statistiques officielles.
- Le contrôle A/B est calculé sur le temps attribuable ; le temps inconnu est exposé séparément. Les changements de contrôle ne sont pas assimilés à des interceptions.
- Les distances sont limitées aux segments observés, calibrés et plausibles. Le terrain de référence mesure **120 × 70 m** ; il faut adapter ses dimensions au stade pour une interprétation métrique.
- Une vidéo télévisée ne montre pas tous les joueurs. Les occultations, ralentis, maillots similaires et coupures peuvent fragmenter les identités. La ré-identification reste imparfaite.
- BoT-SORT utilise ici des caractéristiques d'un classifieur générique. Ce n'est pas un modèle d'identité entraîné spécifiquement sur des footballeurs.
- La largeur/profondeur et les cartes d'occupation ne décrivent que les joueurs visibles. Elles ne prouvent ni une formation complète ni la domination d'une équipe.
- Tirs, xG, fautes, corners, formation tactique et reconnaissance des remplacements **ne sont pas encore implémentés**.

## Vérifier le projet

```bash
python -m unittest discover -s tests -v
```

Les tests couvrent notamment les collisions de maillots, la ré-identification, les coupures, les distances après fusion, le contrôle du ballon, la calibration, les fenêtres temporelles, les unités et les cinq vues de démonstration Streamlit. Des essais d'import sur GPU et CPU complètent ces tests ; ils ne constituent pas un benchmark de précision du tracking.

## Déploiement

Dans Streamlit Community Cloud, utiliser `examples/soccer/streamlit_app.py` comme fichier principal et choisir Python 3.11 dans les paramètres de déploiement. Les dépendances sont dans `examples/soccer/requirements.txt` et les paquets système dans `packages.txt`.

Le projet est accessible à **[football-ai-x.streamlit.app](https://football-ai-x.streamlit.app/)**. Les nouveaux checkpoints locaux ne sont ni téléversés ni activés automatiquement.

## Crédits

Base : [Roboflow Sports](https://github.com/roboflow/sports), par Piotr Skalski / Roboflow, sous [licence MIT](LICENSE). Les bibliothèques, poids et jeux de données conservent leurs licences respectives. Les modèles Ultralytics utilisent leur propre licence ; les annotations de terrain utilisées indiquent CC BY 4.0.
