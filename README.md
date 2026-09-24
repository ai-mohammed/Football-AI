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
- **Associer un maillot à un ID** : galerie d’images source, lectures concordantes, candidats à vérifier, correction manuelle et export des associations par équipe.
- **Vérifier les mesures** : temps indéterminé, couverture de calibration, ballon localisé et maillots confirmés. Export CSV des pistes/événements, JSON de la période et vidéo annotée de l'extrait complet.

La bibliothèque s'ouvre sans charger PyTorch ni télécharger de modèles. Les vidéos de démonstration sont encodées en H.264 720p et pèsent environ **3,5 à 3,9 Mo** chacune, à **25 images/s**. Le rendu conserve toutes les images de la source, même lorsque l’analyse en traite une sur deux.

## Ce qui fonctionne

| Fonction | Mise en œuvre |
| --- | --- |
| Détection | Modèles football pour les joueurs, gardiens, arbitres et ballon ; suppression des boîtes presque identiques avant et après suivi |
| Points du terrain | Modèle à 32 points ; RANSAC et suivi optique bref des repères lors d’une détection manquante |
| Vue drone expérimentale | Petits joueurs détectés par zones, exclusion du hors-terrain, calibration par les lignes blanches |
| Suivi individuel | ByteTrack ou BoT-SORT avec compensation du mouvement caméra et caractéristiques d'apparence, dans le mode Analyse par joueur |
| Équipes | Couleur du torse, centres robustes et vote récent pondéré ; rejet des couleurs ambiguës ou éloignées |
| Numéro de maillot | Lecteur spécialisé football facultatif, consensus sur plusieurs images, contrôle des conflits et validation visuelle |
| Ré-identification | Rapprochement équipe + numéro ; refus de fusionner des pistes dont les périodes de visibilité se chevauchent |
| Analyse | Trajectoires horodatées, heatmaps, distance observée, vitesse sur segments mesurables |
| Ballon | Suivi du mouvement, contrôle confirmé sur plusieurs observations, passes probables et réseau |
| Export | Vidéo annotée, CSV et JSON contenant mesures, positions horodatées et diagnostics |
| Entraînement | Audit des annotations, affinage YOLO11, comparaison avec un modèle de référence |

**Un ID de suivi n'est pas un numéro de maillot.** Sans lectures suffisamment cohérentes, le joueur conserve un libellé `ID…`. Deux équipes peuvent avoir un même numéro. Aucune limite artificielle à onze pistes ne masque les erreurs de suivi.

## Utiliser l'application

Dans la [version en ligne](https://football-ai-x.streamlit.app/), choisir **Extraits analysés**, puis un des cinq passages de douze secondes. Ils ont été recalculés avec **BoT-SORT, le lecteur de maillots spécialisé ViT-S et les détecteurs football historiques**. Ce ne sont pas des démonstrations du candidat YOLO11 expérimental.

Pour utiliser ta carte graphique, depuis la racine du dépôt, dans un environnement Python 3.11 avec PyTorch adapté à ton GPU :

```bash
python -m pip install -r examples/soccer/requirements.txt
python -m pip install -e . --no-deps
python -m pip install -r requirements-jersey.txt
python scripts/setup_jersey_model.py
python -m streamlit run examples/soccer/streamlit_app.py
```

Dans **Importer une vidéo**, choisir le début et la durée du passage. Les réglages permettent de sélectionner ByteTrack ou BoT-SORT, l'échantillonnage et l'OCR lorsqu'il est installé. Les modèles football manquants sont téléchargés uniquement au lancement de l'analyse ; BoT-SORT télécharge également son modèle d'apparence au premier usage.

Le lecteur spécialisé est facultatif ; EasyOCR reste un repli possible. Les nouvelles démos contiennent **27 associations par consensus, contre 2 auparavant**, dont 9 sur le premier extrait. Ce sont des sorties à vérifier, pas un taux de précision. L’onglet **Maillots** expose les images et permet de corriger les propositions. Le quatrième extrait reste sans numéro confirmé. Voir [l’installation, la provenance et les règles de lecture](docs/JERSEY_NUMBERS.md) et [la comparaison avant/après](reports/JERSEY_IDENTITY.md). Les nouveaux imports sur Streamlit Cloud n’activent pas automatiquement ce modèle local.

Le choix **Drone · terrain entier** active l'analyse par zones de 1280 pixels avec recouvrement, puis une suppression globale des doublons. Les quatre côtés du terrain et la ligne médiane doivent être visibles, avec le grand axe à l'horizontale. La géométrie est recalculée sur chaque image : en cas d'échec, les positions restent indisponibles. Ce mode conserve des IDs de suivi ; l'OCR, le ballon et les passes y sont désactivés. Il ne charge que le détecteur de joueurs.

Les dimensions du terrain sont réglables dans l'import et la carte reprend ces mêmes dimensions. La référence par défaut est **105 × 68 m**, avec une surface de réparation corrigée de **16,5 × 40,32 m**. Les mesures restent estimées tant que les dimensions réelles du stade ne sont pas connues. Voir [les essais de calibration et de vue drone](reports/AERIAL_CALIBRATION.md).

L'interface limite les traitements à **30 secondes avec CUDA**, **8 secondes sur CPU**. Le CPU est automatiquement utilisé si aucun GPU compatible n'est présent. L'analyse complète peut prendre plusieurs minutes et n'est pas temps réel ; utiliser les démos pour une présentation fluide. Le GPU du PC n'est pas accessible automatiquement depuis l'application hébergée. Les fichiers temporaires d'import sont nettoyés après traitement ; le résultat reste dans la session Streamlit.

Pour un traitement reproductible, avec les poids déjà téléchargés :

```bash
python scripts/analyze_match.py --video chemin/match.mp4 --output runs/mon-match --tracker botsort --device auto
```

Le dossier de sortie contient `annotated.mp4`, `preview.jpg` et `analysis.json`. Ajouter `--frames 100` pour un essai court, `--stride 2` pour échantillonner et `--no-ocr` pour désactiver la lecture des maillots.

Pour une vue aérienne complète :

```bash
python scripts/analyze_match.py --video chemin/drone.mp4 --output runs/drone --profile aerial --tracker botsort --stride 3 --pitch-length 105 --pitch-width 68
```

Les dimensions de cet exemple sont une hypothèse à adapter au stade. Le profil standard `broadcast` reste sélectionné par défaut pour les plans de tribune et les vidéos télévisées.

Pour reconstruire les cinq démos à partir des vidéos présentes dans `examples/soccer/notebooks` :

```bash
python scripts/build_demos.py --output-dir runs/mes-demos --device cuda --seconds 12 --stride 2
```

La commande écrit dans un nouveau dossier pour préserver les exemples publiés. Après vérification, remplacer les fichiers de chaque extrait dans `examples/soccer/demo_data`. Le format de données et les règles de calcul sont décrits dans [le guide du tableau de bord](docs/TACTICAL_WORKSPACE.md).

## Continuité des actions

La première passe de l’extrait 1 est désormais détectée : **ID22 → ID3, de 0,48 à 2,32 s**. Le passeur reste mémorisé pendant un trajet de ballon observé, même lorsqu’il dure plus d’une seconde. Les alternances isolées entre voisins ne suffisent plus à produire une passe.

Les brèves disparitions de positions sont interpolées **pour l’affichage uniquement**, entre deux observations du même joueur distantes d’au plus 0,32 s. Les repères concernés sont en pointillés sur la carte. Les statistiques utilisent les observations ; elles ne gagnent ni distance ni contrôle artificiels. La calibration peut suivre ses repères par mouvement optique pendant une courte défaillance du détecteur, puis redevient indisponible si les contrôles échouent.

Les cinq démos ont été recalculées. Les résultats avant/après, le cas restant sans passe confirmée et les limites sont détaillés dans [le rapport de continuité](reports/CONTINUITY.md).

## YOLO11 : entraîné, mesuré, encore expérimental

Un **YOLO11n-pose à 32 points** a été affiné localement : 20 époques initiales, puis 80 à partir du meilleur checkpoint. Sur les 30 images de validation disponibles, sa mAP50–95 des points atteint **0,743**, contre **0,925** pour le modèle historique.

Le modèle historique reste donc sélectionné par défaut. Le candidat YOLO11 n'a pas satisfait le contrôle géométrique sur le court extrait testé : une bonne métrique globale ne suffit pas pour obtenir des distances fiables. Les détails, résultats exportés et conditions de mesure sont dans [le rapport](reports/README.md).

L'entraînement est reproductible pour le terrain, le ballon et les joueurs. Seul le terrain a été entraîné dans cette expérience ; les jeux d'annotations et les poids ne sont pas versionnés. Voir [le guide](docs/TRAINING.md).

## Interpréter les résultats

- Les « contrôles » et « passes probables » reposent sur la proximité ballon/pieds, la confirmation temporelle et la continuité observée du ballon. Ce ne sont pas toutes les touches physiques ni des statistiques officielles.
- Le contrôle A/B est calculé sur le temps attribuable ; le temps inconnu est exposé séparément. Les changements de contrôle ne sont pas assimilés à des interceptions.
- Les distances sont limitées aux segments observés, calibrés et plausibles. Le terrain de référence mesure par défaut **105 × 68 m** ; il faut adapter ses dimensions au stade pour une interprétation métrique. Les exports enregistrent les dimensions utilisées.
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

Lecteur de maillots externe : [Łukasz Grad, CVPRW 2025](https://github.com/lukaszgrad/uncertainty-jnr). Son fichier LICENSE indique **CC-BY-NC-SA-4.0** ; les détails et la divergence avec son README sont documentés dans [le guide](docs/JERSEY_NUMBERS.md). Ses poids ne sont pas inclus dans ce dépôt.
