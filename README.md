# 📈 Fis SECVision — Customer Movement Analytics





[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io) [![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-blueviolet.svg)](https://ultralytics.com/)

Plateforme d'analyse vidéo par IA conçue pour fournir aux exploitants de points de vente des métriques sur le comportement et la circulation des clients : comptage, temps de présence (dwell time), suivi multi-personnes et exports analytiques.

---


## Aperçu
Fis Vision analyse des vidéos (ou flux) pour détecter et suivre des personnes, mesurer les entrées/sorties d'une Zone d'Intérêt (ROI) définie et calculer des métriques exploitables (ex : fréquentation horaire, temps moyen passé sur un rayon). L'interface Streamlit permet de définir la ROI, uploader une vidéo et visualiser la vidéo annotée, le journal d'événements et des graphiques d'affluence.
<img width="1713" height="909" alt="582440185_820038594075683_782456656528039243_n" src="https://github.com/user-attachments/assets/b4b22990-ef2c-4604-b95c-de04d8326268" />
<img width="1667" height="835" alt="581747989_908513228167064_2117190142543821954_n" src="https://github.com/user-attachments/assets/c368863d-a105-42d4-a1b5-30299158ec7d" />

Les captures ci‑dessus montrent :
- l'interface utilisateur (définition de la GREEN ZONE, upload),
- la vidéo annotée avec bounding boxes et trajectoires,
- un graphique d'affluence temporelle.

---

## Fonctionnalités clés
- Détection et suivi multi-personnes (YOLOv8 + tracker).
- Zone d'Intérêt (ROI) dynamique et personnalisable.
- Enregistrement d'événements : ZONE_ENTER, ZONE_EXIT (horodatage, person_id, durée).
- Calcul du dwell time par visiteur et agrégation de statistiques.
- Tableau de bord Streamlit : vidéo annotée, journal, graphiques.
- Exports : events.csv, persons.json, vidéo annotée (optionnel).
- Module d'alerte (détection de comportements anormaux) — optionnel.

---

## Prérequis
- Python 3.8+
- Git
- Optional GPU: CUDA 11.x / 12.x (pour accélérer inference avec torch)
- Espace disque pour outputs (vidéos annotées, logs)

---

## Installation rapide

1. Cloner le dépôt
```bash
git clone [VOTRE_URL]/FIS-secvision.git
cd FIS-secvision
```

2. Créer et activer un environnement virtuel
- Windows
```bash
python -m venv venv
venv\Scripts\activate
```
- macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```
Remarque : si vous utilisez GPU, installez torch compatible CUDA avant d'installer les autres paquets (voir docs PyTorch).

4. Lancer l'application Streamlit
```bash
streamlit run app.py
```
Par défaut l'UI est disponible sur http://localhost:8501

---

## Utilisation (pas à pas)
1. Ouvrir l'UI Streamlit.
2. Définir la GREEN ZONE (ROI) : entrer les coordonnées (format expliqué plus bas) ou utiliser l'outil de dessin si présent.
3. Uploader une vidéo MP4 (ou sélectionner un flux caméra si implémenté).
4. Cliquer sur "RUN DETECTION".
5. Visualiser :
   - la vidéo annotée (bounding boxes, IDs, trajectoires),
   - le journal d'événements (events.csv),
   - les graphiques d'affluence / dwell time.
6. Télécharger les exports pour analyses externes.

---

## Format de la GREEN ZONE (ROI)
Le champ attend une liste de points qui dessinent un polygone. Accepted formats (exemples) :
- Paire séparée par espace : "x1,y1 x2,y2 x3,y3 x4,y4"
- Exemple utilisé dans l'UI :  
  850,350 10,550 10,1400 2700,1400 2700,700

Conseils :
- Les coordonnées sont en pixels par rapport à la résolution de la vidéo.
- Vérifiez l'ordre des points (sens horaire/anti-horaire) si la détection d'entrée/sortie semble inversée.

---

## Sorties générées
Par défaut les résultats sont stockés dans le dossier `outputs/` (configurable).

- outputs/events.csv  
  Colonnes typiques : timestamp, person_id, event_type, zone_id, duration_seconds, x, y, frame_number  
  Exemple de ligne :  
  2025-11-17T07:39:15.123Z, 3, ZONE_ENTER, green_zone, , 850,350, 2245

- outputs/persons.json  
  Structure JSON : statistiques agrégées par personne (id, total_time_in_zone, num_entries, first_seen, last_seen)

- outputs/annotated_<input_name>.mp4 (si activé)  
  Vidéo d'entrée avec boîtes, IDs, trajectoires et annotations de la ROI.

---

## Configuration & variables d'environnement
Variables utiles :
- PORT — Port Streamlit (défaut 8501)
- MODEL_WEIGHTS — Chemin vers les poids YOLOv8 (ex: weights/yolov8n.pt)
- OUTPUT_DIR — Dossier des sorties (défaut : outputs/)
- TRACKER_CONF — Paramètres du tracker (IOU, distance, etc.)

Exemple `.env` :
```env
PORT=8501
MODEL_WEIGHTS=weights/yolov8n.pt
OUTPUT_DIR=outputs
```

---

## Bonnes pratiques & optimisation
- Pour traitement temps réel, exécutez sur GPU (CUDA) et utilisez un modèle YOLOv8 léger (ex : yolov8n).
- Réduisez la résolution d'entrée si la précision reste suffisante.
- Ajustez les seuils de confiance (confidence) et l'IOU du tracker pour limiter le re‑assignement de IDs.
- Filtrez objets par taille/min area pour éviter faux positifs (sacs, petites zones).

---


