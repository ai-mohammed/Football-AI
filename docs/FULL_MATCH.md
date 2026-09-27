# Cinq minutes, une vidéo et une analyse continues

Ouvrir **Match complet** dans
[Football AI](https://football-ai-x.streamlit.app/).
Les cinq premières minutes se lisent dans **un seul fichier vidéo**, avec une
carte synchronisée et les graphiques de toute la période. Le contrôle **Aller à
un moment** place la lecture sur un des 25 repères de 12 secondes. Il ne change
pas de fichier et ne provoque pas d’arrêt au repère suivant. Les repères sont
gérés dans le navigateur, sans relancer Streamlit.

Le filtre **Période étudiée** sélectionne les mesures des graphiques et exports.
Il ne découpe pas la vidéo ; naviguer dans le lecteur ne change ni les graphiques
ni la période étudiée. Les résultats sont préparés sur le GPU local ; la
consultation sur Streamlit Cloud ne lance pas les modèles. Le navigateur lit le
MP4 directement, sans recevoir une copie encodée de toute la vidéo via Streamlit.

## Période et repères

Le fichier fourni contient 321 813 images à 50 images/s, en 1280 × 720, soit
1 h 47 min 16,26 s. La période autorisée ici reste **00:00–05:00**. Le fichier
original complet reste local et la suite du match n’est pas traitée.

| Repère | Début dans le fichier | Fin exclue |
|---|---|---|
| 001 | 00:00:00 | 00:00:12 |
| 002 | 00:00:12 | 00:00:24 |
| 011 | 00:02:00 | 00:02:12 |
| 025 | 00:04:48 | 00:05:00 |

Le rendu vérifié contient les **15 000 images source**, à 50 images/s. Avec le
pas de quatre images du manifeste, l’analyse contient 3 750 observations à
12,5 observations/s. La vidéo finale a été décodée intégralement pour vérifier ce nombre. Les
repères ne retirent ni ne répètent d’images. Les horaires sont ceux du fichier,
pas nécessairement le chronomètre officiel du match.

## Continuité du suivi et limites

Un nouvel appel d’analyse BoT-SORT parcourt les 300 secondes depuis la source,
sans réutiliser les analyses des anciens segments. Le tracker, l’historique de
calibration, les votes de couleur et l’état du contrôle du ballon persistent
au-delà de 12, 24 ou 36 secondes. Une action n’est plus interrompue simplement
parce qu’elle traverse un repère. Les IDs sont communs à cette analyse continue.

Les **vrais changements de caméra**, ralentis et pauses de la source restent
visibles. Le détecteur de changement de plan peut réinitialiser le suivi visuel
et la calibration pour ne pas projeter les anciennes positions sur un autre
plan. Une occlusion ou une sortie du champ peut encore produire une nouvelle
piste. La continuité entre deux repères ne garantit pas une ré-identification
parfaite après tous les changements de caméra.

Le contrôle des plans combine les variations d’image, de couleur et les
correspondances visuelles. Cela détecte aussi des transitions de réalisation ;
ce n’est pas un classement sémantique des ralentis. Les mouvements de caméra
cohérents sont vérifiés pour limiter les fausses ruptures.

Avant d’ouvrir une nouvelle piste dans le profil télévisé, l’analyse exige une
calibration du terrain. Les points d’appui projetés à plus de 1,5 m des limites
sont exclus du suivi : spectateurs et bancs ne doivent pas gonfler l’effectif.
Une piste existante peut continuer en image pendant une courte perte de repères,
mais elle ne produit alors aucune position métrique. Cette règle peut retarder
l’apparition d’un vrai joueur lorsque la calibration échoue ; elle ne force
jamais le total à onze.

Une calibration nouvelle est aussi comparée aux repères suivis dans l’image.
Un désaccord médian dépassant deux mètres est rejeté tant que le déplacement
mesuré des repères reste fiable. Le relais par suivi optique expire après
0,32 seconde et ne prolonge pas une géométrie simplement mémorisée.
Lorsque les deux estimations concordent, la correction des repères du modèle
est progressive, tandis que le mouvement de caméra mesuré est conservé.

Les compositions vidéo fortement encadrées par des bandes noires, dont la
double vue présente dans cet exemple, suspendent la carte et les mesures.
La vidéo continue ; aucun joueur de la vue secondaire n’est placé artificiellement
sur le terrain. Ce filtre de mise en page est conservateur : il peut aussi
écarter une vidéo simple avec de larges bandes noires. Il ne détecte pas tous
les montages et ne constitue pas une détection générale des ralentis.

Les couleurs A/B sont apprises sur des plans larges et conservées pour le calcul.
Le petit groupe de couleur jaune correspondant à des arbitres avait faussé le
premier regroupement ; il est écarté lorsque les critères de groupe rare et
éloigné sont réunis. Les noms de clubs ne sont pas déduits des couleurs.
Les numéros de maillot restent des lectures automatiques à vérifier.

Les cartes et distances utilisent uniquement les périodes calibrées. La calibration
reste indisponible pendant un gros plan non exploitable ; aucun point n’y est
inventé sur la carte. Les interruptions restantes
des graphiques peuvent donc représenter des données inconnues. Les pistes ne
sont pas un effectif de joueurs uniques et le contrôle mesuré n’est pas une
possession officielle. L’exclusion automatique des ralentis n’est pas implémentée.

## Reconstruire la version continue

Partir du manifeste borné aux cinq premières minutes, qui contient l’empreinte
du fichier et les références de couleur. Utiliser un nouveau dossier de sortie :

```powershell
python scripts/build_continuous_match.py --video "C:\chemin\match.mp4" --directory runs/football/matches/barcelona-real-2025-26 --output runs/football/matches/barcelona-real-2025-26/continuous-v2 --device cuda
```

Le script vérifie la source et refuse une période dépassant cinq minutes. Il
sauvegarde `analysis.json.gz`, puis rend `annotated.mp4` à la cadence source.
Le `progress.json` du dossier de sortie suit l’inférence, pas le rendu vidéo.
Garder le PC allumé pendant le calcul. Si seul le rendu a échoué et que
`annotated.mp4` n’existe pas, `--render-only` réutilise l’analyse terminée.
Le script refuse tout `annotated.mp4` existant, même incomplet.
Un calcul d’inférence interrompu doit être relancé ; il n’y a
pas de reprise de l’état interne du tracker au milieu de cette version continue.

La publication est séparée et exige que la diffusion de ces résultats soit
autorisée. Le script valide la chronologie et le nombre d’images avant d’activer
les deux fichiers complets dans l’index public :

```powershell
python scripts/publish_continuous_match.py --directory runs/football/matches/barcelona-real-2025-26 --continuous-directory runs/football/matches/barcelona-real-2025-26/continuous-v2 --tag match-barcelona-real-2025-26 --catalog examples/soccer/match_data/barcelona-real-2025-26/manifest.json
```

L’analyse compressée et le MP4 sont publiés comme fichiers immuables nommés par
leur empreinte. Le `progress.json` public de la release désigne la version active et le
catalogue Git sert d’index de secours. Les données d’analyse sont vérifiées par
SHA-256 au chargement. Aucun secret ni fichier original complet n’est publié.

## Compatibilité avec les anciens segments

Les anciens bundles restent disponibles dans la
[release du match](https://github.com/ai-mohammed/Football-AI/releases/tag/match-barcelona-real-2025-26).
Leur traitement indépendant, assuré par `process_full_match.py`, et leur cumul
via `build_match_overview.py` sont conservés pour les anciens catalogues et le
secours en cas d’échec du chargement de l’analyse continue. Dans ce mode historique, les IDs
restent préfixés par segment et ne sont pas fusionnés. Ces anciennes analyses
ne sont jamais mélangées aux résultats de la nouvelle analyse continue.

## Vérification reproductible

`scripts/audit_match_geometry.py` examine les mêmes images source que l’analyse,
conserve les points détectés dans un cache associé à la source et au modèle,
et exporte les changements de plan et les désaccords géométriques rejetés.
`scripts/compare_match_stability.py` compare deux analyses de la même période
avec le même échantillonnage : pistes brèves, personnes observées, calibration
et événements. Une baisse du nombre de pistes n’est pas une mesure IDF1/HOTA ;
il faut des identités annotées pour mesurer la justesse de la ré-identification.

`scripts/refine_match_geometry.py` permet de refaire la calibration à partir
des repères enregistrés sans relancer les détecteurs. Il conserve les détections
et les preuves de maillots, puis recalcule positions, distances, contrôle et
événements. Les anciennes mesures ne sont pas réutilisées. Le rendu annoté doit
être régénéré lorsque des vues sont exclues. Résultats et limites de l’essai :
[stabilité de la diffusion](../reports/BROADCAST_STABILITY.md).
