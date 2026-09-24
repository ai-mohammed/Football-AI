# Calibration corrigée et analyse aérienne — 24 septembre 2026

[Application Streamlit](https://football-ai-x.streamlit.app/) · [Mesures drone](aerial_evaluation.json) · [Mesures de géométrie](calibration_geometry_comparison.json)

## Correction de la géométrie

Le gabarit historique utilisait une surface de réparation de **20,15 × 41 m**.
La profondeur est corrigée à **16,5 m** et la largeur à **40,32 m** (deux fois
16,5 m plus l'ouverture de but de 7,32 m), conformément à la
[Loi 1 de l'IFAB](https://www.theifab.com/laws/latest/the-field-of-play/).
La référence par défaut devient **105 × 68 m**, avec dimensions réglables à l'import
et en ligne de commande. Ce choix n'est pas une mesure automatique du stade.
Le lecteur, les graphiques, les positions et les exports utilisent la même référence.

Les cinq vidéos originales ont été échantillonnées aux images 0, 10, …, 290,
soit **150 images 1920 × 1080**. Même détecteur de terrain historique, mêmes
prédictions et seuil de confiance 0,5 pour les deux géométries. La tolérance
RANSAC reste de 6 pixels, avec au moins 60 % de points retenus.

| Mesure | Ancien gabarit 120 × 70 m | Gabarit corrigé 105 × 68 m |
| --- | ---: | ---: |
| Images dont la calibration est acceptée | 83 / 150 | 147 / 150 |
| Erreur médiane sur un point exclu du calcul | 22,29 px | 7,42 px |
| 90e percentile de cette erreur | 109,88 px | 31,67 px |

Pour chaque point suffisamment confiant, l'homographie est ajustée sur les autres
points, puis le point exclu est projeté dans l'image. **1 625 prédictions exclues**
sont comparées dans chaque condition, sur les images ayant au moins six points.
Cette mesure est plus informative que l'erreur des seuls points retenus par RANSAC.
Elle vérifie la **cohérence entre prédictions du modèle**, pas une précision absolue
mesurée avec des annotations manuelles ou des positions GPS. Les dimensions et
les marquages ont changé ensemble ; le gain ne peut pas être attribué à un seul paramètre.

Reproduire la mesure sans relancer le détecteur :

```bash
python scripts/evaluate_calibration.py --observations reports/calibration_observations.json --output runs/calibration-check.json
```

Les observations publiées incluent l'empreinte SHA-256 du modèle. Aucun seuil de
validation n'a été relâché pour augmenter la couverture.

## Cinq démonstrations recalculées

Les cinq passages de 12 secondes ont été entièrement retraités sur RTX 3080,
avec BoT-SORT, OCR et un pas de 2 images. Leurs cartes, distances et événements
proviennent du nouveau calcul ; les anciens JSON n'ont pas simplement été remis à l'échelle.

| Extrait | Ancienne couverture | Nouvelle couverture |
| --- | ---: | ---: |
| 08fd33_0 | 98,7 % | 100 % |
| 0bfacc_0 | 81,3 % | 100 % |
| 121364_0 | 38,7 % | 96 % |
| 2e57b9_0 | 48,7 % | 99,3 % |
| 573e61_0 | 10,7 % | 94,7 % |

Les résultats complets sont dans [recalibrated_demos.json](recalibrated_demos.json).
Une couverture élevée signifie qu'une projection a été acceptée ; elle ne garantit
ni une identité correcte ni des distances exactes. Les matchs restent partiellement visibles.

## Solution pour les joueurs très petits en vue drone

Le profil **Drone · terrain entier** est distinct du profil de diffusion :

1. Détection dans des zones de 1280 pixels avec 20 % de recouvrement. La suppression
   globale des doublons intervient avant le suivi, y compris entre classes de personnes.
2. Recherche d'un quadrilatère vert entouré de lignes blanches, ajustement sur leurs
   longues portions droites et vérification indépendante de la ligne médiane.
3. Rejet des centres de détection hors terrain, avec une petite marge de ligne de touche.
4. Support spatial d'association d'au moins 1/64 de la largeur d'image pour ByteTrack
   et BoT-SORT. Cela tolère le déplacement d'un petit joueur entre deux images échantillonnées.
   Les boîtes exportées et dessinées restent les boîtes réellement détectées.
5. Projection des centres des joueurs depuis le dessus. La géométrie est mesurée
   de nouveau à chaque image ; une calibration absente n'est pas remplacée par l'ancienne.

Le profil exige un terrain entier, le grand axe proche de l'horizontale, des lignes
visibles et un contraste suffisant. C'est une méthode expérimentale pour cette prise
de vue, pas une calibration universelle. Il utilise les poids existants, sans nouvel
entraînement. Aucun nombre maximum de pistes ne cache les erreurs de suivi.

**OCR, ballon et événements de ballon sont désactivés en vue aérienne.** Les dos ne
sont pas suffisamment lisibles et le modèle de ballon n'est pas validé pour cette vue.
Les résultats indisponibles sont signalés dans l'interface et les diagnostics.
Les modèles de ballon et de terrain ne sont pas chargés pour ce profil.

## Évaluation annotée sur trois passages 4K

Source : [TeamTrack, Atom Scott et collaborateurs](https://github.com/AtomScott/TeamTrack),
[archive publique Zenodo 8302872](https://zenodo.org/records/8302872), licence **CC BY 4.0**.
Vidéos `D_20220220_1_0060_0090`, `0300_0330` et `0600_0630`, chacune analysée
sur les secondes 5–11 : 3840 × 2160, 29,97 images/s, **60 observations**, pas de 3.
Les deux derniers décalages ont été contrôlés par comparaison avec les images source
voisines ; la première image correspond à l'index source 150, donc à l'annotation 151.

Le premier passage a servi au développement, le deuxième au contrôle secondaire.
Le troisième n'a été utilisé qu'après fixation des paramètres de la version finale.
**Il s'agit du même match et de la même caméra**, pas de trois matchs indépendants.
Les annotations n'ont pas été réauditées manuellement. Leurs IDs ne sont pas des numéros de maillot.

Une localisation concordante exige que le centre de la boîte prédite soit dans une
boîte annotée, avec appariement un-à-un. La précision est la part des prédictions
ainsi appariées ; le rappel est la part des joueurs annotés retrouvés.

| Passage / méthode | Joueurs localisés par image | Rappel des centres | Précision des centres | Pistes cumulées | Calibration acceptée |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0060 — nouveau profil | 13,83 / 21,87 | 63,26 % | 92,84 % | 28 | 100 % |
| 0300 — nouveau profil | 12,47 / 22 | 56,67 % | 95,90 % | 36 | 100 % |
| 0600 — plein cadre, entrée 2560 | 2,00 / 22 | 9,09 % | 18,46 % | 58 | 0 % |
| 0600 — nouveau profil | 11,67 / 22 | 53,03 % | 95,50 % | 44 | 100 % |

Le critère standard **IoU ≥ 0,5** est également conservé :

| Passage / méthode | Rappel IoU | Précision IoU |
| --- | ---: | ---: |
| 0060 — nouveau profil | 4,50 % | 6,60 % |
| 0300 — nouveau profil | 4,85 % | 8,21 % |
| 0600 — plein cadre | 8,11 % | 16,46 % |
| 0600 — nouveau profil | 41,67 % | 75,03 % |

La forte différence entre critères, notamment sur les deux premiers passages,
montre que les boîtes du détecteur ne correspondent pas toujours à l'étendue des
boîtes annotées. La mesure des centres ne doit pas être présentée comme une mAP.
Le gain concerne surtout la localisation des joueurs et le rejet des faux positifs.

Sur le dernier passage, dix joueurs annotés sont encore répartis sur plusieurs
pistes et sept changements d'ID sont observés entre deux appariements consécutifs.
Une meilleure association ne résout donc pas toute la ré-identification. Ces
comptages ne remplacent pas une mesure MOT/HOTA/IDF1.

Les analyses aériennes prennent **65,92 à 70,10 secondes pour six secondes de vidéo**
sur RTX 3080, initialisation et écriture temporaire incluses, encodage H.264 final exclu.
Le plein cadre du dernier passage prend 57,22 secondes. Une seule exécution par
condition : ce ne sont ni des latences garanties ni une capacité temps réel.

Le script de mesure peut être réutilisé sur une analyse et le CSV TeamTrack correspondant :

```bash
python scripts/evaluate_teamtrack.py --analysis runs/drone/analysis.json --annotations chemin/annotations.csv --source-frame-offset 150 --output runs/evaluation.json
```

Les coordonnées doivent rester dans la résolution originale. Le décalage 150 vaut
pour le découpage de cette expérience uniquement. Les empreintes des sources et
toutes les mesures sont publiées dans [aerial_evaluation.json](aerial_evaluation.json).

## Limites et vérification

Les améliorations ne permettent pas encore de retrouver tous les joueurs, leurs
maillots ou les actions sur n'importe quelle vidéo. Le prochain travail de précision
doit porter sur un détecteur annoté pour les vues aériennes, une évaluation sur des
matchs indépendants et un modèle d'identité adapté aux petits joueurs. Les mesures
métriques demandent aussi les dimensions réelles du terrain et une validation spatiale.

Les tests couvrent les doublons entre zones, les images vides, les lignes incomplètes,
les faux positifs hors terrain, le déplacement de caméra synthétique, le maintien
des IDs de petits joueurs et l'absence de boîtes artificiellement agrandies dans les
exports. Les cinq démonstrations et l'état « non analysé » du profil drone ont été
vérifiés avec Streamlit AppTest. Le JavaScript du lecteur passe le contrôle de syntaxe.
Le navigateur pilotable n'était pas disponible pour une nouvelle inspection visuelle
du site hébergé ; la mise en ligne doit être distinguée de ces validations locales.

Les vidéos comparatives restent locales dans `runs/football/quality-study/` ; les
cinq démonstrations recalculées et le code sont versionnés. Aucun nouveau checkpoint
YOLO n'est revendiqué ni activé par cette correction.
