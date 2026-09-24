# Football AI

Plateforme Streamlit d'analyse de segments vidéo de football, développée par Mohammed ADDI.
Public prioritaire : un analyste qui explore tactiquement un extrait, puis étudie les joueurs.
Ce choix a été confirmé par l'utilisateur le 24 septembre 2026.

La tâche principale consiste à relier ce qui se passe dans la vidéo aux positions projetées,
aux déplacements et aux transitions de contrôle. Les exemples précalculés doivent être
explorables immédiatement sur Streamlit Cloud, sans GPU. L'utilisateur peut également
importer et analyser son extrait. Les calculs lourds s'exécutent sur l'hôte de l'application.

Les statistiques sont limitées à l'extrait et à la visibilité de la caméra. Les pistes
non identifiées restent des pistes ; les passes et changements de contrôle sont des
estimations. Ne pas annoncer de tirs, fautes, score, xG ou formation non mesurés.

Direction existante conservée : interface sombre, accent vert, équipes rose et cyan,
vidéo réelle au centre. Priorité à la lisibilité et à l'exploration plutôt qu'aux paramètres
des modèles. Réglages techniques regroupés dans l'import, pas dans la galerie.

Le travail demandé doit être intégré à main après validation. Déploiement existant :
https://football-ai-x.streamlit.app/
