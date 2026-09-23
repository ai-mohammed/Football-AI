# Résultats mesurés — 24 septembre 2026

## Affinage YOLO11 du terrain

Machine : NVIDIA GeForce RTX 3080. Ultralytics 8.4.103, PyTorch 2.13.0+cu130.
Export local `football-field-detection-12`, 32 points, 222 images d'entraînement,
30 de validation, 24 de test. Graine 42, taille 640, lot de 8.

| Modèle | Entraînement de cette expérience | mAP50–95 des points, validation |
| --- | --- | ---: |
| YOLO11n-pose, premier essai | 20 époques | 0,0594 |
| YOLO11n-pose, affinage prolongé | 80 époques supplémentaires depuis le meilleur checkpoint | 0,7433 |
| Terrain historique du projet | Évaluation uniquement | 0,9254 |

Le candidat affiné pèse **6,81 Mo**. Son checkpoint reste local dans
`runs/football/yolo11n-pitch-refined/train/weights/best.pt`.

**Décision : conserver le modèle historique par défaut.** Un modèle plus petit
ou plus récent ne justifie pas de dégrader la projection des joueurs sur le terrain.
Les résultats bruts, la taille des poids et l'empreinte du dataset sont dans
[yolo11_pitch_experiment.json](yolo11_pitch_experiment.json).

Cette comparaison porte sur un petit ensemble de validation, pas sur des matchs
indépendants. La provenance d'entraînement du modèle historique n'est pas
vérifiée contre cette partition. Les latences enregistrées sont des observations
de validation, sans protocole de benchmark dédié. Le jeu de test n'a pas été
utilisé pour choisir les réglages.

## Essais du pipeline sur vidéo

Source locale : `08fd33_0.mp4`, 1920 × 1080, 25 images/s. Pas d'annotations
d'identité ou d'événements disponibles sur cette vidéo.

| Essai | Images traitées / pas | Calibration acceptée | OCR |
| --- | --- | --- | --- |
| ByteTrack, terrain historique (premier test) | 30 / 2 | 28 sur 30 | Désactivé |
| BoT-SORT + apparence, terrain historique, équipes par couleur | 75 / 2 | 73 sur 75 | Activé |
| ByteTrack, candidat YOLO11 affiné, équipes par couleur | 30 / 2 | 0 sur 30 | Désactivé |

Le contrôle géométrique exige au moins quatre correspondances fiables, 60 %
d'inliers et une tolérance RANSAC de six pixels. Le candidat YOLO11 ne passe pas
ce contrôle sur l'extrait testé, malgré sa mAP élevée : il ne produit donc pas
de mesures métriques pour cet extrait. Le système conserve la valeur « inconnue ».

Le test BoT-SORT a terminé en 35,67 s, préchauffage et OCR inclus, pour six
secondes de vidéo échantillonnée. **Ce pipeline complet n'est pas temps réel.**
Les 24 identités restaient non résolues par le maillot au terme de ce test court.
Une transition de contrôle a été exportée comme passe probable, sans validation
manuelle de l'événement. Aucune amélioration IDF1/HOTA ou précision OCR n'est
déduite de ces essais.

La variante initiale SigLIP/PCA sans blanchiment regroupait presque tous les
joueurs dans la même équipe sur la première image. Elle a été remplacée par
une référence simple utilisant la couleur du torse, avec faible poids de la
luminance et possibilité de rester indéterminé. Une inspection de l'aperçu a
confirmé une séparation cohérente des maillots verts/blancs sur cette image ;
ce n'est pas une validation sur toutes les couleurs, tous les stades ou tous les matchs.

Les vidéos annotées et JSON complets sont conservés localement sous
`runs/football/validated-botsort` et `runs/football/validated-yolo11`.
Les cinq démonstrations versionnées n'ont pas été régénérées avec cette version.

## Vérification logicielle

Les tests automatisés couvrent les collisions OCR, les identités des deux équipes,
la reprise des distances après fusion, les coupures, les images sans joueur,
le temps de possession, les points aberrants, le repli CPU, le suivi après
réinitialisation, la classification de couleurs et la validation des annotations.
Un test Streamlit vérifie aussi l'affichage d'une piste sans vitesse calculable.

```bash
python -m unittest discover -s tests -v
```
