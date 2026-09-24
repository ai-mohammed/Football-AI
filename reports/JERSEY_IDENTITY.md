# Détection, équipes et maillots — 24 septembre 2026

Les cinq extraits ont été recalculés sur la RTX 3080. Les associations de numéro
ayant atteint le consensus passent de **2 à 27** ; le premier extrait passe de **1 à 9**.
Il s’agit de propositions du système, **pas d’une précision de 27/27**. Les extraits
ne disposent pas d’annotations exhaustives indépendantes d’identité, d’équipe ou de maillot.

## Protocole et résultats

Référence : commit `b140cf4cab4799941c80f4d4caf2ee3f5b10e715`. Même source 1080p,
mêmes douze premières secondes, 150 observations espacées de 0,08 s, BoT-SORT et
détecteurs football historiques. Aucune modification ni nouvel entraînement des poids YOLO
n’est revendiqué dans cette comparaison. Le lecteur spécialisé est un modèle externe
ViT-S de Łukasz Grad ; [provenance, installation et licence](../docs/JERSEY_NUMBERS.md).

| Extrait | Identités avant → après | Maillots par consensus avant → après | Paires de boîtes IoU > 0,85 avant → après | Passes avant → après |
| --- | ---: | ---: | ---: | ---: |
| 1 · 08fd33_0 | 26 → 23 | 1 → 9 | 3 → 0 | 3 → 3 |
| 2 · 0bfacc_0 | 27 → 24 | 1 → 5 | 3 → 0 | 5 → 5 |
| 3 · 121364_0 | 30 → 30 | 0 → 3 | 0 → 0 | 2 → 2 |
| 4 · 2e57b9_0 | 26 → 23 | 0 → 0 | 4 → 0 | 0 → 0 |
| 5 · 573e61_0 | 34 → 30 | 0 → 10 | 5 → 0 | 1 → 1 |

Une paire est comptée pour chaque observation où deux boîtes joueur/gardien se
recouvrent au-delà du seuil. Cela mesure un symptôme de duplication, pas tous les faux
positifs. Les pistes restent cumulées sur l’extrait : le total ne représente pas
l’effectif simultanément présent, et n’est pas plafonné à onze par équipe.

La calibration reste disponible sur 150/150 observations des quatre premiers extraits
et 149/150 du cinquième. Cela mesure la disponibilité, pas l’erreur de projection.
Chaque vidéo garde **300 images à 25 images/s**. Le nombre de passes émises et reçues
est cohérent avec les événements exportés sur les cinq extraits. La première passe de
l’extrait 1 reste **ID22 → ID3, départ à 0,48 s, arrivée à 2,32 s**.

Les empreintes des analyses, les associations proposées et les événements complets
avant/après sont dans [jersey_identity_demos.json](jersey_identity_demos.json).
Les paramètres de consensus et les seuils sont décrits dans [le guide](../docs/JERSEY_NUMBERS.md).

## Vérifications visuelles ciblées

- Extrait 1, ID11 : l’ancien OCR proposait **4** ; les images du dos montrent **6**,
  valeur proposée par le lecteur spécialisé. Les images de #9 et #18 sont également
  lisibles dans la galerie. ID18 rejoint l’équipe blanche et ID20 l’équipe verte,
  corrigeant les deux erreurs de couleur repérées dans la version précédente.
- Extrait 2, ID24 : l’ancienne proposition **1** devient **7**, cohérente avec les
  images retenues. Les galeries des extraits 2, 3 et 5 ont également été examinées.
- Extrait 4 : aucune association ne passe les critères. Les petites silhouettes et
  les vues sans dos lisible ne justifient pas d’attribuer des numéros.

Ces contrôles sont guidés par les propositions du modèle et ne constituent donc pas
un jeu de test annoté à l’aveugle. Un consensus peut répéter une erreur, et une piste
peut encore changer de personne lors d’une occultation. Le numéro aide à rapprocher
les pistes ; il ne garantit pas à lui seul une identité stable sur un match entier.

## Modifications vérifiées

Le filtrage spatial agit avant l’allocation des pistes puis après leur mise à jour :
sur l’extrait 4, un dernier doublon provenait du rapprochement des boîtes par le suivi.
Le clustering emploie des centres médians, un rejet des observations éloignées et
des votes récents pondérés. Cela permet de corriger une erreur initiale sans changer
d’équipe à chaque image ambiguë.

La lecture des maillots conserve les preuves horodatées et jusqu’à trois recadrages
par identité. Les validations de l’onglet **Maillots** restent explicites, réversibles
et séparées des observations. Elles ne créent ni mouvement ni passe.

Vérification automatique : tests de consensus, répétition d’une même image, couleur
hors distribution, correction temporelle, suppression de doublons avant/après suivi,
conflits de maillots et correction réversible dans Streamlit. La suite complète compte
66 tests Python ; le contrôle du lecteur JavaScript vérifie aussi la continuité.
Les cinq démos sont ouvertes par les tests Streamlit après remplacement de leurs données.
La galerie a été examinée dans le navigateur sur ordinateur et à 390 pixels de largeur.
Un essai CPU de l’adaptateur retrouve les mêmes cinq numéros que l’implémentation source
sur cinq recadrages de contrôle, avec moins de 0,002 d’écart entre les scores. Ce contrôle
vérifie la conversion et l’inférence ; il ne vérifie pas l’identité réelle de ces joueurs.

```bash
python -m unittest discover -s tests
node tests/test_replay_component.cjs
```

Les démos hébergées n’exécutent pas le lecteur de maillots : elles affichent ses résultats
précalculés. Les nouveaux imports nécessitent l’installation du lecteur et des poids
sur la machine qui réalise l’analyse. La précision globale du suivi et de la lecture
reste à mesurer avec des annotations indépendantes ; aucun gain HOTA, IDF1 ou mAP
n’est déduit des comptes de ce rapport.
