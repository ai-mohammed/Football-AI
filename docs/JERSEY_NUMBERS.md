# Associer les maillots aux identités

Dans [Streamlit](https://football-ai-x.streamlit.app/), l’onglet **Maillots** rassemble
les IDs, leurs pistes associées, leur équipe et les lectures de numéro. Trois images
source au maximum permettent de vérifier chaque proposition. Les images sont recadrées
et enregistrées en JPEG, sans génération de détails. Leur résolution native est indiquée.

Un numéro n’est pas un ID : deux adversaires peuvent porter le 9. Le rapprochement
automatique utilise **équipe + numéro + absence de visibilité simultanée**. Une image
où le dos est caché, trop petit ou flou ne permet pas de retrouver tous les numéros.

## Lecteur spécialisé, facultatif

Depuis la racine du dépôt, après avoir installé les dépendances de l’application :

```bash
python -m pip install -r requirements-jersey.txt
python scripts/setup_jersey_model.py
```

Le script télécharge le checkpoint ViT-S publié par Łukasz Grad, puis conserve uniquement
les paramètres nécessaires à l’inférence dans
`examples/soccer/data/jersey-small16.safetensors` (environ 87 Mo). Il ne réentraîne pas
le réseau et n’écrase pas un fichier existant. Le téléchargement initial est d’environ
294 Mo. Une copie déjà téléchargée peut être convertie avec `--checkpoint chemin/small16.pth`.
`FOOTBALL_JERSEY_MODEL` permet de sélectionner un autre emplacement du fichier converti.

Le lecteur est sélectionné automatiquement lorsque les dépendances et les poids sont
présents. Sinon, EasyOCR est utilisé s’il est installé ; sans lecteur, les IDs restent
disponibles. La reconnaissance est désactivée dans le profil drone. Le CPU est compatible,
mais un GPU est préférable pour recalculer les extraits.

**Les cinq démos en ligne sont précalculées avec ce modèle.** Leur consultation ne charge
aucun réseau. L’installation Streamlit Cloud par défaut n’installe pas ce lecteur et ne
télécharge pas ces poids : les nouveaux imports sur cet hébergement ne bénéficient donc
pas automatiquement de cette reconnaissance. Les validations manuelles restent accessibles.

## Comment une lecture devient une association

1. Suppression des boîtes presque identiques avant le suivi et à sa sortie, sans limiter
   artificiellement l’effectif à onze. Le seuil d’intersection sur union est de 0,85.
2. Séparation des équipes par couleur du torse en espace Lab : centres médians,
   rejet des couleurs trop éloignées ou ambiguës, vote pondéré sur les observations récentes.
   L’écart entre clusters n’est pas une probabilité de classification correcte.
3. Échantillonnage décalé des maillots toutes les cinq images traitées. Les petites
   images, les images peu nettes et les recouvrements importants sont écartés.
4. Lecture spécialisée du torse avec le ViT-S : probabilité par numéro et incertitude
   de Dirichlet. Une lecture retenue exige un score d’au moins 0,65 et une incertitude
   d’au plus 0,2. Le score reste celui du modèle, pas une exactitude vérifiée.
5. Consensus : au moins trois lectures, espacées d’au moins 0,16 s sur une même piste,
   accord d’au moins 80 % et score moyen d’au moins 0,75. Des images proches dans le temps
   peuvent néanmoins répéter la même erreur.
6. Fusion prudente des pistes : équipe suffisamment étayée, même numéro, périodes de
   visibilité non chevauchantes. Un conflit entre joueurs visibles simultanément masque
   le numéro et bloque la fusion. Les IDs restent distincts en cas de doute.

Les preuves exportées incluent l’instant, la piste, le lecteur, le numéro et le score.
Les statuts distinguent lecture concordante, candidat, conflit, non lu et validation humaine.
La couleur observée à chaque instant reste conservée dans les données ; le rendu vidéo
utilise l’attribution finale pour stabiliser les couleurs.

## Vérifier et corriger dans Streamlit

Sélectionner une piste dans **Maillots**, examiner les images et saisir le numéro visible.
Une attribution à deux joueurs de la même équipe présents dans une même observation est
refusée. **Rétablir la lecture automatique** annule une validation.

La correction ne fusionne pas les pistes et ne modifie pas les observations ou les passes.
Elle change les libellés de la carte, des tableaux, du réseau et des exports. Elle reste
dans la session courante ; le CSV des associations et le JSON permettent de l’enregistrer.
La vidéo déjà encodée conserve ses annotations initiales.

## Modèle externe et attribution

Modèle : **Łukasz Grad, “Single-Stage Uncertainty-Aware Jersey Number Recognition in Soccer”,
CVPR Workshops 2025**, variante `small16_reid` entraînée sur SoccerNet.
[Dépôt et checkpoint officiel](https://github.com/lukaszgrad/uncertainty-jnr) ·
[Guide d’inférence](https://github.com/lukaszgrad/uncertainty-jnr/blob/main/docs/INFERENCE.md).

Attention à une divergence dans la source : son README annonce CC-BY-SA-4.0, mais son
[fichier LICENSE](https://github.com/lukaszgrad/uncertainty-jnr/blob/main/LICENSE) contient
**CC-BY-NC-SA-4.0**, avec une restriction non commerciale. Cette intégration indique la
licence du fichier LICENSE ; elle ne présente pas les poids comme librement utilisables
commercialement. Les poids ne sont pas redistribués dans ce dépôt.

L’adaptateur utilise timm et la tête de classification du checkpoint publié. La conversion
retire l’optimiseur et la tête de préentraînement inutilisée ; elle conserve l’auteur,
la source, la licence et l’empreinte du checkpoint en métadonnées. Football AI n’est pas
l’auteur ni l’entraîneur de ce modèle. Les résultats sur nos extraits et leurs limites
sont dans [le rapport de vérification](../reports/JERSEY_IDENTITY.md).
