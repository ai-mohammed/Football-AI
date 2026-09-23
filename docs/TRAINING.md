# Entraîner et évaluer les modèles football

Le pipeline distingue trois problèmes : détection des personnes, détection d'un petit ballon, localisation de 32 points du terrain. Un YOLO11-pose préentraîné sur le corps humain prédit 17 articulations : ses poids doivent être affinés sur les annotations de terrain avant de pouvoir calibrer une vue de football.

## Données attendues

Exporter les annotations au format YOLO, avec `train/images`, `train/labels`, `valid/images`, `valid/labels` et éventuellement `test/…`.

| Tâche | Classes, dans cet ordre | Points |
| --- | --- | --- |
| `pitch` | `pitch` | `kpt_shape: [32, 3]` ; ordre exact de `SoccerPitchConfiguration.vertices` |
| `ball` | `ball` | Aucun |
| `players` | `ball`, `goalkeeper`, `player`, `referee` | Aucun |

Les coordonnées sont normalisées entre 0 et 1. Chaque ligne terrain contient une classe, une boîte et 32 triplets x/y/visibilité. Le script vérifie les formes, classes, valeurs finies, coordonnées visibles, présence des labels et doublons exacts entre ensembles. Il écrit un YAML avec chemins absolus dans le dossier d'expérience sans modifier les annotations originales.

**Séparer les matchs et caméras avant l'export.** L'audit détecte les fichiers identiques, pas toutes les images voisines ou augmentées d'une même séquence. Une séparation aléatoire image par image peut rendre les résultats artificiellement élevés. L'audit ne peut pas vérifier automatiquement la signification des 32 indices.

## Audit sans entraînement

Depuis la racine du dépôt :

```bash
python scripts/train_football.py --task pitch --data examples/soccer/notebooks/football-field-detection-12/data.yaml --output runs/pitch-audit --audit-only
```

Ces exports locaux ne sont pas inclus dans Git. Les sources Roboflow et notebooks historiques restent référencés dans la documentation de `examples/soccer`.

## Affinage et comparaison

```bash
python scripts/train_football.py --task pitch --data chemin/data.yaml --output runs/yolo11-pitch --model yolo11n-pose.pt --epochs 100 --batch 8 --imgsz 640 --device 0 --baseline examples/soccer/data/football-pitch-detection.pt
```

Pour le ballon : `--task ball --model yolo11n.pt`. Pour les joueurs : `--task players --model yolo11s.pt`, avec un jeu de données aux quatre classes attendues. Ne pas employer directement les poids COCO dans l'analyse de match : leur classe 2 ne signifie pas « joueur ».

Le script fixe la graine à 42, consigne les versions, conserve les poids sous `train/weights/`, et écrit `comparison.json`. La comparaison utilise le même jeu de validation et la même résolution. Les ensembles de test restent réservés à une évaluation finale indépendante. Le script ne remplace jamais les poids de l'application.

Pour le terrain, les retournements et la mosaïque sont désactivés : la correspondance sémantique des points doit rester cohérente. Pour une expérience plus poussée, comparer les résolutions 640/960/1280, YOLO11n/s/m et les augmentations, avec un budget et des partitions documentés.

## Utiliser un candidat explicitement

```bash
python scripts/analyze_match.py --video chemin/match.mp4 --output runs/essai-candidat --pitch-model runs/yolo11-pitch/train/weights/best.pt --tracker botsort --device auto
```

Dans Streamlit, les variables d'environnement `FOOTBALL_PITCH_MODEL`, `FOOTBALL_PLAYER_MODEL` et `FOOTBALL_BALL_MODEL` sélectionnent des poids personnalisés avant le démarrage. Exemple PowerShell :

```powershell
$env:FOOTBALL_PITCH_MODEL = (Resolve-Path 'runs/yolo11-pitch/train/weights/best.pt').Path
python -m streamlit run examples/soccer/streamlit_app.py
```

Un chemin personnalisé invalide déclenche une erreur ; il n'est pas remplacé silencieusement par un téléchargement historique. Les poids locaux ne sont pas disponibles sur Streamlit Cloud sans transfert explicite.

## Mesures nécessaires avant promotion

1. Détection : précision/rappel et mAP, en particulier sur les petits ballons.
2. Terrain : erreur de reprojection et proportion d'images effectivement calibrées sur des vidéos indépendantes. La mAP seule ne suffit pas.
3. Suivi : IDF1, HOTA et changements d'identité sur des séquences **annotées avec des identités réelles**. Compter les IDs produits ne mesure pas la qualité du suivi.
4. Numéros : précision des numéros confirmés, couverture et fausses fusions, par équipe.
5. Événements : validation manuelle horodatée des transitions de contrôle ; distinguer une interception d'une passe ratée demande davantage d'information.
6. Performance : latence après préchauffage, mémoire et images/seconde du pipeline complet, sur le matériel cible.

Les exemples actuels n'ont pas de vérité terrain d'identité/événements : aucune valeur IDF1/HOTA ou précision de passes n'est annoncée. Voir les résultats réellement obtenus dans [reports](../reports/README.md).

## Sources techniques

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Format d'annotations pose](https://docs.ultralytics.com/datasets/pose/)
- [ByteTrack, BoT-SORT et ReID](https://docs.ultralytics.com/modes/track/)
