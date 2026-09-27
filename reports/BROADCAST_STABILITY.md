# Stabilité sur cinq minutes de diffusion — 27 septembre 2026

Source : les 300 premières secondes du fichier Barcelone–Real Madrid déjà
utilisé dans la démonstration. 720p, 50 images/s, 3 750 observations avec un pas
de quatre images. Les versions avant et après utilisent les mêmes modèles
football et les mêmes prototypes de couleur. Les résultats bruts sont dans
[broadcast_stability_comparison.json](broadcast_stability_comparison.json).

## Problèmes observés et corrections

Le seuil de différence d’image précédent ne signalait qu’un changement de
plan. Les nouvelles vérifications de couleur et de correspondances visuelles
retrouvent les principaux changements entre gros plans, plans larges et vues
de bord de terrain. Ce détecteur reste une heuristique : il signale également
les transitions graphiques et peut produire de fausses alertes, notamment
lors d’un mouvement rapide dans un gros plan.

Les naissances de pistes sont désormais conditionnées à une calibration, avec
une marge de 1,5 m autour du terrain pour exclure les personnes hors terrain.
Les pistes déjà actives peuvent survivre brièvement en image à une perte de
calibration, mais ne produisent alors pas de positions métriques.

La géométrie des nouveaux repères est comparée à leur déplacement mesuré par
suivi optique aller-retour. Les désaccords importants sont rejetés ; lorsque
les estimations concordent, 25 % de correction par le modèle et 75 % de
projection issue du mouvement observé limitent le bruit entre deux images.
Le relais sans nouvelle mesure directe expire après 0,32 s.

Une inspection des images a révélé une **double vue encadrée** vers 57–60 s.
Une seule homographie était appliquée aux deux scènes. La nouvelle version
suspend la carte, les annotations et les mesures pendant **3,44 s**, sans
retirer ces images de la vidéo. Le critère de bandes sombres est conservateur :
il ne reconnaît pas tous les montages et peut rejeter une vidéo simple avec
de larges bandes noires. Ce n’est pas un détecteur général de ralentis.

## Comparaison mesurée

| Indicateur | Avant | Après |
|---|---:|---:|
| Pistes brutes de personnes | 780 | 617 |
| Pistes visibles moins de 0,5 s | 339 | 243 |
| Identités/pistes après consensus maillot | 714 | 555 |
| Identités avec numéro par consensus | 18 | 16 |
| Secondes avec calibration acceptée | 197,28 | 195,92 |
| Temps de présence cumulé des joueurs cartographiés | 3 387,36 s | 3 349,12 s |
| Passes probables | 21 | 21 |
| Secondes de contrôle indéterminé | 250,96 | 251,60 |

Le total de 555 **n’est pas un effectif de joueurs uniques**. Les occlusions,
retours après changements de caméra et maillots illisibles fragmentent encore
les identités. La réduction des pistes brèves est de 28,3 %, mais ne constitue
pas une mesure de précision d’identité. La baisse des numéros par consensus
ne doit pas être présentée comme une amélioration OCR.

Pour les paires d’images ayant au moins trois pistes communes, le déplacement
médian collectif dépasse un mètre dans 259 cas avant et 198 après. Le 95e
percentile passe de 1,8598 à 1,4460 m. Il s’agit d’un indicateur de discontinuité
à 12,5 observations/s ; ces déplacements ne sont pas des annotations de vérité
terrain et les ensembles de pistes visibles diffèrent légèrement.

La calibration accepte un peu moins de temps après exclusion du montage.
Les 21 passes restent des estimations et 83,9 % du temps reste sans contrôle
du ballon attribuable : cet essai n’établit donc pas des statistiques officielles
du match ni une précision comparable à un système professionnel validé.

## Reproduction et vérification

Le premier recalcul (`continuous-v2`, local) reprend la source avec les nouveaux
contrôles de plans et de pistes. La géométrie a ensuite été affinée à partir des
repères détectés sur exactement les mêmes images, sans relancer les détecteurs.
La version finale (`continuous-v4`, locale) recalcule toutes les positions,
distances, contrôles et passes ; elle régénère aussi le rendu annoté pour retirer
les annotations sur la double vue. Les preuves de maillots et les associations
issues du premier passage sont conservées et restent à vérifier.

```powershell
python scripts/audit_match_geometry.py --video "C:\chemin\match.mp4" --analysis runs/football/matches/barcelona-real-2025-26/continuous-v1/analysis.json.gz --output runs/football/geometry-audit
python scripts/refine_match_geometry.py --video "C:\chemin\match.mp4" --analysis runs/football/matches/barcelona-real-2025-26/continuous-v2/analysis.json.gz --keypoints runs/football/geometry-audit/keypoints.npz --output runs/football/refined
python scripts/compare_match_stability.py --before runs/football/matches/barcelona-real-2025-26/continuous-v1/analysis.json.gz --after runs/football/refined/analysis.json.gz --output reports/broadcast_stability_comparison.json
```

Les scripts refusent un cache provenant d’une autre source ou d’un autre modèle.
Les tests couvrent mouvements de caméra, bruit de repères, montage encadré,
personnes hors terrain, pertes de calibration et recalcul des statistiques :
87 tests Python et les tests du lecteur JavaScript passent.

La vidéo finale a été décodée intégralement : 15 000 images à 50 images/s,
soit 300 secondes. Les 3 750 observations sont contiguës, sans trou ni
chevauchement. Les 43 observations de la vue encadrée ne contiennent ni
positions projetées, ni annotations de joueurs, ni attribution de contrôle.
