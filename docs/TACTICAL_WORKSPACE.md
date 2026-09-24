# Atelier tactique des extraits

[Ouvrir l'application](https://football-ai-x.streamlit.app/)

## Parcours

1. Ouvrir **Extraits analysés** et sélectionner un passage dans la bibliothèque.
2. Choisir la période étudiée et, éventuellement, une piste à suivre.
3. Lire la vidéo, avancer image par image ou sélectionner une action sous le lecteur.
4. Explorer **Tactique**, **Joueurs**, **Événements** et **Fiabilité & exports**.

Le curseur du lecteur contrôle l'instant affiché sur la carte. Le filtre « Période étudiée »
contrôle les statistiques agrégées et borne la lecture. Les exports CSV/JSON respectent
cette fenêtre ; l'export vidéo contient toujours l'extrait complet. L'instant zéro d'une
vidéo importée correspond au début du passage choisi, indiqué au-dessus du résultat.

Les contrôles natifs de la vidéo permettent aussi la lecture en plein écran. Le terrain
se place sous la vidéo sur les petits écrans. La carte n'extrapole pas les joueurs hors champ.

## Lecture des visualisations

| Vue | Calcul | Limite |
| --- | --- | --- |
| Contrôle A / B | Durée attribuée à l'équipe divisée par la durée attribuable | Le temps inconnu est exclu du ratio et affiché séparément |
| Chronologie du contrôle | Équipe du joueur proche du ballon à chaque échantillon | La proximité ne prouve pas une touche physique |
| Occupation | Secondes de présence cumulées par zone, normalisées à 100 % | Influence du cadrage ; ce n'est pas une mesure de domination |
| Réseau de passes | Transitions probables ; positions moyennes pondérées par le temps | Uniquement les pistes reliées ; flèche dans le sens de la transition |
| Largeur / profondeur | Max moins min sur les axes transversal / longitudinal | Au moins trois joueurs visibles ; pas une formation complète |
| Distance / vitesse | Mouvements consécutifs calibrés, plausibles et sans coupure | Distance partielle et sensible à la calibration |
| Épisodes de contrôle | Entrées observées en contrôle dans la période | Une occultation peut fragmenter un épisode ; ce ne sont pas les touches physiques |

Une passe probable est retenue lorsque le contrôle passe à une autre piste de la même
équipe dans un délai d'une seconde. Une transition entre équipes est seulement nommée
« changement de contrôle ». Les événements dont le départ précède la fenêtre choisie
sont exclus de ses comptes. Aucune précision événementielle n'est revendiquée sans annotations.

## Données version 3

`analysis.json` contient :

- `schema_version`, `source_video`, `duration_s`, `source_fps`, `stride` ;
- `source_start_s` pour un extrait découpé depuis une vidéo importée ;
- `pitch` : dimensions du terrain de référence, unité mètres ;
- `frames` : `time_s`, `dt`, `calibrated`, positions de `players`, `ball`, `possessor` ;
- champs facultatifs d'audit : `source_resolution`, `frames[].image_detections` (boîtes source en pixels), `calibration_method`, `pitch_boundary_px` ;
- `players` : identité, pistes associées, hypothèse de maillot, trajectoire et `motion_samples` ;
- `events` : transitions horodatées, coupures et diagnostics d'identité ;
- `diagnostics` : méthodes utilisées, couverture et paramètres.

Les positions dans `frames` et les distances dans `motion_samples` sont en **mètres**.
Le champ historique `players[].trajectory` reste en **centimètres** pour compatibilité
avec les anciens annotateurs. Le nouveau tableau de bord utilise les positions de `frames`.

Un échantillon couvre `[time_s, time_s + dt)`. Un intervalle de mouvement couvre
`[time_s - dt, time_s]`. Les durées sont coupées à l'intersection avec la fenêtre choisie,
et la distance d'un intervalle partiellement sélectionné est pondérée par sa durée.
Les images sans géométrie valide et les trous de données restent indéterminés.

Les nouvelles analyses utilisent une référence de **105 × 68 m**, modifiable dans l'import
ou par `--pitch-length` / `--pitch-width`. Les surfaces de réparation mesurent 16,5 × 40,32 m.
Le lecteur SVG, les graphiques et les exports prennent les dimensions du JSON ; les anciens
exports à 120 × 70 m restent lisibles. La couverture de calibration mesure la disponibilité
d'une projection, pas sa précision absolue.

Le profil `aerial` détecte les petits joueurs dans des zones recouvrantes, élimine les doublons
avant le suivi et rejette les centres hors terrain lorsque ses limites sont reconnues.
Il estime la géométrie à partir des quatre lignes extérieures et vérifie la ligne médiane.
L'OCR et les événements de ballon sont indisponibles dans ce profil : `ball_events_available=false`.
Le profil ne recycle jamais une ancienne géométrie lors d'un échec ; une caméra partielle,
un terrain trop incliné dans l'image ou des lignes invisibles demandent un autre protocole.

Les IDs de pistes et d'identités restent distincts. Les identités fusionnées après
consensus équipe/maillot sont résolues dans les positions et les événements exportés.
La couleur d'équipe d'un échantillon conserve l'attribution observée à cet instant.
Le numéro affiché dans le tableau de bord est le consensus obtenu sur l'ensemble de l'extrait,
alors que la vidéo annotée montre ce qui était connu au moment du traitement de chaque image.

## Architecture et performances

`streamlit_app.py` orchestre la bibliothèque et l'import. `tactical_dashboard.py` affiche
les figures et appelle le lecteur dans `components/tactical_player`. Celui-ci utilise
un composant Streamlit v1 sans dépendance JavaScript externe. Vidéo et SVG partagent
la même horloge de lecture dans le navigateur : aucun aller-retour Python par image.

`sports/common/segments.py` calcule les agrégats sans importer PyTorch. `analysis_service.py`
est chargé uniquement dans le parcours d'import. Les exemples ne déclenchent ni inférence
ni téléchargement de poids. Les vidéos H.264 720p sont intégrées au lecteur depuis les
fichiers locaux : elles n'exigent aucun hébergement vidéo tiers.

Les détecteurs historiques restent utilisés, avec ByteTrack ou BoT-SORT. Le modèle
YOLO11 affiné localement est expérimental ; voir [les mesures](../reports/README.md).

Le pipeline d'import choisit automatiquement le matériel disponible, limite la durée,
nettoie ses fichiers temporaires et conserve le résultat dans la session. Sur un hébergement
CPU, l'analyse reste lente. La bibliothèque précalculée est le parcours prévu pour les démonstrations.
