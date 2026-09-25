# Explorer un match complet par segments

Dans [Football AI](https://football-ai-x.streamlit.app/), ouvrir **Match complet**,
puis choisir un segment vidéo. **Les cinq premières minutes sont réunies dans
un seul tableau de bord** : le filtre temporel couvre 0 à 300 secondes, et tous
les graphiques utilisent les données de la période choisie, quel que soit le
segment affiché dans le lecteur. Passer à la vidéo suivante ne remet pas les
statistiques à zéro. La carte synchronisée suit la vidéo du segment courant.
Les résultats sont préparés sur le GPU local et publiés sur GitHub. La
consultation sur Streamlit Cloud ne lance aucune inférence.

## Découpage du fichier fourni

Le fichier `FC Barcelona 2 - 0 Real Madrid 2025_26 PARTIDO COMPLETO_720p.mp4`
contient 321 813 images, à 50 images/s, en 1280 × 720 : **1 h 47 min 16,26 s**.
L'analyse demandée est limitée à **00:00–05:00**, soit **25 segments de 12
secondes**, sans trou ni recouvrement. Le traitement de la suite est arrêté.
Le fichier entier représenterait 537 segments ; cette totalité n'est pas la
période présentée dans le tableau de bord.

| Segment | Début dans le fichier | Fin exclue | Durée |
|---|---|---|---|
| 001 | 00:00:00 | 00:00:12 | 12 s |
| 002 | 00:00:12 | 00:00:24 | 12 s |
| 011 | 00:02:00 | 00:02:12 | 12 s |
| 025 | 00:04:48 | 00:05:00 | 12 s |

Ces horaires désignent le **fichier source**, pas le chronomètre officiel du
match. Le lecteur affiche cette position et le temps relatif du segment.
Le plan de la période analysée est exportable en CSV depuis l'application.

## Tous les graphiques sur une chronologie commune

Les observations, mouvements et événements sont replacés à leur temps dans le
fichier : une action à 1,76 s du segment 002 apparaît à **13,76 s** dans les
graphiques. Contrôle du ballon, heatmaps, réseau de passes, largeur/profondeur,
vitesses, déplacements, tableau des pistes et exports utilisent les 25 segments.
Une fenêtre choisie, par exemple 60–120 s, filtre toutes ces mesures et reste
sélectionnée lorsque la vidéo change.

Les IDs sont réindexés sans collision pour le stockage et restent affichés sous
la forme `S0002 · ID8`. Un même ID8, ou une lecture automatique N°9, ne suffit
pas à fusionner deux pistes de segments différents. Les comptes de pistes et
lectures de maillot ne sont donc **pas des nombres de joueurs uniques**.
Les graphiques individuels concernent une piste située dans un segment tant
qu'une continuité d'identité entre segments n'est pas établie. Les couleurs
communes permettent en revanche de cumuler directement les mesures par équipe.

## Actions aux limites et identités

Jusqu'à deux secondes précédant chaque segment sont analysées pour retrouver
le début d'une action. Elles ne sont pas rejouées dans le segment exporté.
Positions, temps de contrôle et déplacements sont bornés au segment. Une passe
commencée dans ce contexte appartient au segment de sa réception, avec le
marqueur `origin_in_context`. Elle n'est pas recomptée dans le segment précédent.
Ce contexte améliore la continuité sans garantir de détecter toutes les passes.

Les modèles sont chargés une fois pour toute la file. Les références de couleur
A/B sont apprises sur des plans larges répartis dans le fichier, puis conservées.
Un petit groupe de couleur très éloigné des deux groupes principaux peut être
écarté de cet apprentissage : sur ce fichier, des arbitres jaunes détectés comme
joueurs avaient faussé le premier regroupement. Cela ne remplace pas un modèle
de reconnaissance des rôles et ne garantit pas chaque affectation individuelle.
Les noms des clubs ne sont pas déduits de ces couleurs. Les pistes et IDs sont
**propres à chaque segment** : ID7 dans deux segments ne suffit pas à identifier
la même personne. Les numéros de maillot restent des lectures à vérifier.

Ralentis, pauses, plans serrés, bancs et célébrations sont conservés. Leur
exclusion automatique et la réconciliation des identités sur tout le match
ne sont pas implémentées. Ne pas additionner tous les segments pour annoncer
des statistiques officielles, un nombre de joueurs uniques ou une durée de jeu.
Les mesures métriques dépendent de la visibilité et de la calibration du terrain.

## Préparer ou reprendre une analyse locale

Avec les dépendances et modèles du projet installés :

```powershell
python scripts/process_full_match.py --video "C:/chemin/match.mp4" --output runs/football/matches/mon-match --title "Mon match" --device cuda
```

Le plan est écrit avant le traitement. Ajouter `--plan-only` pour inspecter le
découpage sans inférence, ou `--segments 1 11` pour traiter deux segments précis.
Pour limiter dès le départ le calcul aux cinq premières minutes, passer
`--segments` suivi des nombres 1 à 25 (en PowerShell : `--segments (1..25)`).
Le pas par défaut est de quatre images : 12,5 observations/s pour cette vidéo,
avec une vidéo annotée exportée à 50 images/s. Le GPU accélère le traitement,
mais le calcul n'est pas en temps réel. Prévoir plusieurs heures pour 537 segments.

La même commande reprend une analyse interrompue : les segments terminés sont
conservés et les segments incomplets ou en échec sont repris. Le fichier source
est vérifié par empreinte SHA-256 ; le pas d'analyse doit rester identique. Ne pas
modifier les modèles ou le code d'analyse pendant une reprise destinée à un même
jeu de résultats. Utiliser un nouveau dossier pour une autre configuration.

Chaque dossier de segment contient `analysis.json`, `annotated.mp4`, `preview.jpg`
et `bundle.zip`. Le fichier `manifest.json` conserve l'état de toute la file.
`worker.lock` empêche deux traitements concurrents dans le même dossier. Après un
arrêt brutal, vérifier que le processus identifié dans le verrou est terminé
avant de retirer ce verrou et relancer. Le PC doit rester allumé et sans veille
pendant le calcul ; un arrêt exige une reprise manuelle.

## Publication sur GitHub et consultation distante

La publication est une action séparée, à utiliser uniquement pour des résultats
dont la diffusion publique est autorisée. Le script utilise l'accès GitHub déjà
configuré pour ce dépôt ; aucun secret n'est écrit dans les résultats.

Une fois les 25 segments terminés et les processus arrêtés, créer la vue globale
et publier la période de cinq minutes :

```powershell
python scripts/build_match_overview.py --directory runs/football/matches/mon-match --seconds 300 --tag match-mon-match --publish
```

Cette commande limite le catalogue à la période demandée et ajoute un fichier
compressé contenant les données cumulées, sans recopier les vidéos. Les sorties
locales éventuellement calculées au-delà de cinq minutes sont conservées mais
exclues de la vue. Chaque vidéo reste téléchargée à la demande ; la vue globale
ne charge pas 25 vidéos en mémoire.

Pour publier progressivement des segments pendant un autre traitement :

```powershell
python scripts/publish_match.py --directory runs/football/matches/mon-match --tag match-mon-match --watch
```

La commande crée une release dédiée dans `ai-mohammed/Football-AI`, puis publie
les segments terminés et l'index `progress.json`. Chaque archive contient seulement
la vidéo annotée et ses résultats ; le fichier original n'est pas publié.
La publication peut reprendre sans recalculer les vidéos. `publication.json`
mémorise les archives déjà envoyées ; `publisher.lock` évite les doublons de
processus. Les erreurs temporaires d'envoi sont retentées pendant la surveillance.

Pour ajouter ce match au catalogue Cloud, enregistrer un instantané de l'index
public dans `examples/soccer/match_data/<identifiant>/manifest.json`. Son champ
`remote_manifest_url` pointe vers la release. Cet instantané sert de secours si
l'index distant est indisponible. Les archives restent dans la release, hors Git.
L'application vérifie leur empreinte SHA-256 et ne charge que le segment choisi.

Le match fourni est publié ici :
[segments et progression](https://github.com/ai-mohammed/Football-AI/releases/tag/match-barcelona-real-2025-26).
L'index public décrit ici les 25 segments des cinq premières minutes et la vue
globale correspondante. Le fichier original complet reste local.
