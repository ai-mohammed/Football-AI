# Continuité des passes et des extraits — 24 septembre 2026

[Démonstration Streamlit](https://football-ai-x.streamlit.app/) · [Données avant/après](continuity_demos.json)

La première passe de l’extrait 1 manquait parce que le lien avec le passeur expirait
après une seconde, alors que le ballon restait visible pendant le trajet.
Le nouveau calcul retrouve **ID22 → ID3, départ 0,48 s, réception confirmée 2,32 s**.
Ce sont des IDs de suivi, pas des numéros de maillot.

## Modifications

- Association du ballon à son mouvement récent, à partir des détections sélectionnées.
  Les candidats rejetés ne polluent plus le centre de l’historique. Aucune position
  prédite n’est exportée comme une observation du ballon.
- Confirmation d’un nouveau contrôleur par deux observations espacées d’au moins
  0,08 s, à moins de 0,24 s l’une de l’autre. Une observation isolée d’un voisin
  ne suffit plus à créer un transfert.
- Conservation du passeur jusqu’à 5 s pendant un trajet observé, sans absence du
  ballon supérieure à 0,4 s. Le déplacement doit atteindre 2 m ; un saut excédant
  45 m/s plus 0,75 m de tolérance invalide le trajet. Le temps sans contrôle reste
  indéterminé. Ces seuils sont des heuristiques, pas un modèle entraîné d’événements.
- Maintien bref de la calibration par suivi optique des repères : au moins six
  correspondances cohérentes aller-retour, erreur de suivi bornée, même validation
  RANSAC, au plus 0,32 s après une calibration directe. Une coupure remet cet état
  à zéro. Référence technique : [OpenCV, Lucas–Kanade pyramidal](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).
- Interpolation réservée au rendu entre observations d’une même piste, à identité
  et équipe identiques, sans coupure, dans une fenêtre de 0,32 s. Les positions
  intermédiaires de la carte sont signalées et n’alimentent aucune statistique.
- Vidéos H.264 rendues avec toutes les images source à 25 images/s. L’analyse
  conserve son pas de deux images. Avance image par image à 0,04 s et accès aux
  passes depuis leur départ, au lieu de leur réception uniquement.

## Recalcul des cinq démos

Même GPU RTX 3080, mêmes détecteurs football historiques, même BoT-SORT et OCR.
Douze secondes par extrait ; 150 observations et 300 images de rendu. Référence
avant correction : commit `7141a62`. Les 1 500 images des cinq MP4 ont été décodées
avec succès après réencodage.

| Extrait | Passes probables avant → après | Calibration avant → après, sur 150 | Positions ajoutées pour l’affichage uniquement |
| --- | ---: | ---: | ---: |
| 01 · `08fd33_0` | 1 → 3 | 150 → 150 | 10 |
| 02 · `0bfacc_0` | 3 → 5 | 150 → 150 | 4 |
| 03 · `121364_0` | 1 → 2 | 144 → 150 | 3 |
| 04 · `2e57b9_0` | 5 → 0 | 149 → 150 | 7 |
| 05 · `573e61_0` | 3 → 1 | 142 → 149 | 8 |

Le suivi optique récupère 14 des 15 calibrations manquantes. Cela mesure la
disponibilité d’une projection, **pas son erreur métrique réelle**.
Les vidéos pèsent maintenant 3,46 à 3,86 Mo, contre une cadence précédente de
12,5 images/s. Les données source de statistiques restent séparées du rendu lissé.

Les baisses de passes dans les extraits 4 et 5 résultent des critères plus stricts
et du nouveau choix des détections du ballon. Certaines anciennes transitions
étaient des allers-retours entre voisins en 0,08 s. **Le zéro de l’extrait 4 ne
prouve pas l’absence de passes réelles** : aucun transfert n’y est confirmé avec
ces détections et ces règles. Un jeu d’annotations humaines reste nécessaire pour
mesurer les faux positifs et les passes manquées, notamment les remises en une touche
et les ballons aériens, dont la projection au sol est inexacte.

## Vérifications

- 55 tests Python réussis, dont trajets longs, voisin temporaire, perte de ballon,
  coupure, rejet de géométrie, séparation interpolation/statistiques et conservation
  de la cadence et des images source.
- Tests JavaScript du lecteur : mouvement intermédiaire, conservation des données,
  refus des sauts entre pistes, coupures et absence de calibration.
- Navigateur local : première passe dans la timeline et le réseau ; bouton de
  relecture à 0,23 s ; avance de 0,04 s ; deux positions signalées comme interpolées
  à 2,00 s ; tableau de bord et carte rendus sans erreur.

Pour reproduire le calcul, avec les vidéos source et les poids déjà présents :

```bash
python scripts/build_demos.py --output-dir runs/continuity-check --device cuda --seconds 12 --stride 2
python -m unittest discover -s tests -v
node tests/test_replay_component.cjs
```

Le nombre de passes exportées est un résultat du système, pas une vérité terrain.
La continuité d’identité hors champ, les occultations longues et les contacts très
rapides restent imparfaits. Ces changements n’entraînent pas de nouveaux poids YOLO.
