# Sign2Speak — Brief Projet pour Claude Code

## 🎯 Vision du projet

**Sign2Speak** est une app Android de **traduction temps réel de la langue des signes américaine (ASL) vers voix**, fonctionnant **100% offline** sur smartphone, conçue pour un usage en extérieur avec un harnais pectoral mains-libres.

**Soumission pour le Gemma 4 Good Hackathon** (Kaggle × Google DeepMind), deadline 18 mai 2026.

### Pitch
- 70M de personnes utilisent une langue des signes dans le monde
- Les interprètes humains coûtent 50-150€/h et ne sont pas disponibles 24/7
- Aucune solution actuelle n'est complète, portable, et offline
- Notre solution : **un smartphone + un harnais à 25€**

### Piste hackathon
**Digital Equity** (accessibilité) + mention secondaire **Health** (communication médicale).

### Bonus visés ($50K au total)
- **LiteRT** (10K) — classifieur ASL au format LiteRT pour AI Edge
- **Cactus** (10K) — SDK pour faire tourner Gemma 4 sur mobile
- **Ollama** (10K) — déjà utilisé en démo PC
- **Unsloth** (10K) — fine-tuning de Gemma 4 sur paires gloss→texte

---

## 📐 Architecture technique complète

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE (sur smartphone)                     │
└─────────────────────────────────────────────────────────────────┘

[Caméra avant smartphone (harnais pectoral)]
        ↓ flux vidéo 30 FPS
[MediaPipe Tasks Android — 3 modèles séparés]
   - HandLandmarker (mains, 21 keypoints chacune)
   - PoseLandmarker (pose, 33 keypoints)
   - FaceLandmarker (visage, 478 keypoints)
        ↓ keypoints par frame
[Assemblage en format Holistic 543 landmarks]
        ↓ (n_frames, 543, 3)
[Preprocessing — normalisation nez, 118 landmarks utiles]
   - Sélection : lèvres (41) + mains (42) + nez (4) + yeux (32)
   - Normalisation : centre sur nez, scale par std
   - Vélocité (dx) + Accélération (dx²)
        ↓ (n_frames, 708) features
[Classifieur LiteRT — Conv1D + Transformer, ~6M params]
   - Architecture : 6× Conv1DBlock + 2× TransformerBlock
   - 50 signes en sortie (vocabulaire MVP)
   - 80% d'accuracy sur signeurs jamais vus
        ↓ gloss prédit ("HELLO", "MY", "NAME")
[Buffer de glosses — accumulation jusqu'à 2s d'inactivité]
        ↓ liste de glosses
[Gemma 4 E2B via Cactus SDK — Gloss2Text on-device]
   - Prompt : "Translate ASL glosses to English: {glosses}"
   - Style sélectionnable : professional / friendly / simple
        ↓ phrase anglaise naturelle
[Android TTS natif (TextToSpeech API)]
        ↓ audio
[Haut-parleur du téléphone]
```

---

## ✅ État actuel (PC, fonctionnel)

### Fichiers existants
```
Sign2Speak-GEMMA-4-GOOD-HACKATHON/
├── keypoint_extractor.py       # MediaPipe Tasks → 298 features (deprecated, gardé pour compat)
├── train_winner.py             # Entraînement TF du modèle gagnant Kaggle (sur WSL)
├── preconvert.py               # Conversion Parquet → npy
├── live_winner.py              # Live test PC (webcam + classifieur + Gemma + TTS)
├── load_google_asl.py          # Loader Google ASL Signs dataset
├── model_winner/
│   ├── best_model.keras        # Modèle entraîné Keras
│   ├── weights.weights.h5      # Poids seuls
│   └── model_meta.json         # Métadonnées (labels, dim, etc.)
├── asl_converted/              # Données pré-converties (sur WSL)
├── piper_voices/               # Voix TTS (en_GB-alba-medium)
└── asl-signs/                  # Dataset Google brut
```

### Ce qui marche déjà
1. **Classifieur** : 82% d'accuracy sur test set du dataset Google (signeurs jamais vus), 80% en live webcam
2. **Live demo PC** (`live_winner.py`) :
   - Webcam → MediaPipe Tasks → Réassemblage Holistic 543
   - Preprocessing identique au gagnant Kaggle (118 landmarks, normalisation nez, vélocité, accélération)
   - Inférence TensorFlow Keras
   - Top 3 prédictions affichées
   - Buffer de glosses avec timeout 2s
   - Appel Ollama → Gemma 4 E2B pour Gloss2Text
   - 3 styles (professional/friendly/simple) sélectionnables via touches 1/2/3
   - TTS Piper (voix `en_GB-alba-medium`)
   - Mute/unmute via touche 'm'

### Modèle de classification (architecture gagnant Kaggle)
- **Source** : Solution 1ère place Kaggle ASL Signs Recognition (hoyso48, MIT License)
- **Input** : (64 frames, 708 features)
- **Architecture** :
  ```
  Masking(-100) → Dense(192) → BatchNorm
  → 3× Conv1DBlock(192, ks=17) → TransformerBlock(192, expand=2)
  → 3× Conv1DBlock(192, ks=17) → TransformerBlock(192, expand=2)
  → Dense(384) → GlobalAvgPool → Dropout(0.5) → Dense(50)
  ```
- **Classes custom** : ECA, CausalDWConv1D, MultiHeadSelfAttention

### Vocabulaire MVP (50 signes)
Signes basés sur le dataset Google ASL Signs (PopSign), vocabulaire enfant/quotidien :
```
hello, bye, thankyou, please, yes, no,
happy, sad, mad, like, hate, fine, sick, hungry, thirsty,
go, drink, give, wait, finish, look, see, talk,
think, listen, sleep, wake, stay, open, close, find,
where, who, why,
mom, dad, boy, girl, child, person,
water, food, milk,
now, time, morning, night, tomorrow, yesterday, later,
home, there, up, down, outside,
have, haveto, can, not
```

### Dataset utilisé
**Google Isolated Sign Language Recognition** (Kaggle competition, 2023)
- 94 477 vidéos de 21 signeurs sourds
- 250 signes (on en garde 50)
- Landmarks MediaPipe Holistic pré-extraits (format Parquet long)
- Split par participant (15 train, 3 val, 3 test signeurs) pour éviter le data leakage

---

## 🎯 OBJECTIF MAINTENANT : App Android

### Matériel cible
**Google Pixel 7a** (Tensor G2, 8 Go RAM, Android 13+)
- NPU Tensor G2 → accélération AI Edge
- 8 Go RAM = peut faire tourner Gemma 4 E2B + MediaPipe + classifieur

### Stack technique recommandé
- **Langage** : Kotlin (Android Studio)
- **MediaPipe** : MediaPipe Tasks Android SDK (`com.google.mediapipe:tasks-vision`)
- **Classifieur** : LiteRT (TensorFlow Lite) via `com.google.ai.edge.litert`
- **LLM Gemma 4 E2B** : **Cactus SDK** (bonus 10K) ou MediaPipe LLM Inference API (Google AI Edge)
- **TTS** : Android TextToSpeech natif

### Étapes de l'app Android

#### 1. Export du classifieur vers LiteRT (bonus 10K LiteRT)
Le modèle actuel est en `.keras` (TensorFlow Keras). Il faut le convertir en `.tflite` :
```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()
with open('classifier.tflite', 'wb') as f:
    f.write(tflite_model)
```
Le `.tflite` sera mis dans `app/src/main/assets/`.

#### 2. Setup projet Android Studio
- Min SDK : 26 (Android 8.0)
- Target SDK : 34 (Android 14)
- Permissions : `CAMERA`, `INTERNET` (juste pour debug), `RECORD_AUDIO`
- Architecture MVVM ou simple (selon préférence)

#### 3. Capture caméra + MediaPipe
- `CameraX` pour la capture (plus simple que Camera2)
- `HandLandmarker`, `PoseLandmarker`, `FaceLandmarker` en mode VIDEO
- Réassemblage en format Holistic 543 landmarks (logique identique à `live_winner.py`)

#### 4. Inférence LiteRT
- Buffer glissant de 64 frames de keypoints
- Preprocessing identique au PC (sélection 118 landmarks, normalisation nez, vélocité, accélération)
- Inférence du `.tflite` via LiteRT
- Lissage temporel (vote majoritaire sur les dernières prédictions)

#### 5. Intégration Cactus SDK pour Gemma 4 (bonus 10K Cactus)
Cactus est un SDK qui simplifie le déploiement de LLMs sur mobile :
- Site : https://github.com/cactus-compute/cactus
- Permet de charger Gemma 4 E2B en GGUF quantifié
- Routing intelligent CPU/GPU/NPU selon la complexité

#### 6. TTS Android
```kotlin
val tts = TextToSpeech(context) { status ->
    if (status == TextToSpeech.SUCCESS) {
        tts.language = Locale.US
        tts.setSpeechRate(1.0f)
    }
}
tts.speak(sentence, TextToSpeech.QUEUE_FLUSH, null, "asl_translation")
```

#### 7. UI
Minimaliste — focus sur la démo :
- Preview caméra plein écran
- Overlay : gloss courant (gros), phrase traduite (en bas), top 3 (côté)
- Boutons : mute/unmute TTS, sélection de style

---

## 📋 Détails techniques importants

### Format des keypoints (Holistic 543)
```
Indices 0-467     : face (468 landmarks)
Indices 468-488   : left_hand (21 landmarks)
Indices 489-521   : pose (33 landmarks)
Indices 522-542   : right_hand (21 landmarks)
```

### Inversion des mains (PIÈGE)
- **Holistic** : "left_hand" = main GAUCHE du sujet
- **MediaPipe Tasks** : la handedness "Right"/"Left" est du point de vue de la **caméra**

⚠️ **NE PAS inverser** : si `handedness == "Left"` → offset 468 (left_hand). Si `handedness == "Right"` → offset 522 (right_hand). C'est le mapping correct vérifié en live.

### Liste des 118 landmarks utilisés
```python
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE  # 118 total
```

### Preprocessing (708 features par frame)
```python
# 1. Centre sur le nez (landmark 17 dans le 543)
mean_nose = mean(coords[:, 17, :])  # sur toutes les frames

# 2. Sélectionne 118 landmarks
x = coords[:, POINT_LANDMARKS, :]  # (T, 118, 3)

# 3. Normalise par std
std = sqrt(mean((x - mean_nose)²))
x = (x - mean_nose) / std

# 4. Garde x, y seulement
x = x[..., :2]  # (T, 118, 2)

# 5. Vélocité
dx = x[1:] - x[:-1]  # (T-1, 118, 2)

# 6. Accélération  
dx2 = x[2:] - x[:-2]  # (T-2, 118, 2)

# 7. Concat → (T, 118 × 2 × 3) = (T, 708)
features = concat([x, dx, dx2], axis=-1).reshape(T, 708)

# 8. NaN → 0
features = nan_to_num(features, 0)
```

### Padding séquence à 64 frames
```python
PAD = -100.0  # valeur de masking
if T > 64:
    features = features[:64]
elif T < 64:
    features = pad(features, ((0, 64-T), (0, 0)), constant_values=PAD)
```

### Prompt Gemma 4 (gardé identique au PC)
```
SYSTEM: You are an ASL (American Sign Language) interpreter. You receive 
a sequence of ASL glosses and must convert them into natural English sentences.

ASL grammar is different from English:
- ASL uses topic-comment structure (topic first)
- Time indicators come first
- No articles or "to be" verbs in ASL

Rules:
- Output ONLY the English sentence, nothing else
- Keep it concise and natural
- Style: {style}

USER: Translate these ASL glosses to English: {glosses}
```

---

## 🚨 Priorités absolues

D'après le brief du hackathon :
1. **Impact + Vidéo = 70% des points** (40 + 30)
2. **Tech = 30% des points**

Donc **NE PAS** sur-ingénier l'app Android. Un MVP qui marche en démo > un produit complet qui foire le tournage.

### Roadmap suggérée
- **J1-J2** : Setup projet Android, intégration MediaPipe + LiteRT, classifier qui marche
- **J3** : Intégration Cactus + Gemma 4 E2B
- **J4** : UI propre + TTS + harnais
- **J5-J6** : **TOURNAGE VIDÉO** (priorité absolue)
- **J7** : Write-up + soumission Kaggle

### Différenciateurs à mettre en avant
- 100% offline (pas de cloud)
- < 50€ de matériel (téléphone + harnais)
- Privacy by design (rien ne quitte le téléphone)
- 4 styles de traduction
- Architecture du gagnant Kaggle (94% sur 250 signes en compétition)

---

## ❓ Questions ouvertes pour Claude Code

1. **Quel SDK pour Gemma 4** ? Cactus (bonus) vs MediaPipe LLM Inference vs llama.cpp Android ?
2. **Threading** : un thread pour chaque modèle (Hand/Pose/Face/Classifier/Gemma/TTS) ou pipeline async ?
3. **Mode bidirectionnel (Phase 2)** : ajouter STT (speech-to-text) Android natif pour que l'entendant puisse répondre, affichage en texte/avatar pour le signeur ? À mentionner dans le write-up comme roadmap si pas le temps.
4. **Format d'export final** : .tflite via TFLite Converter directement, ou passer par ONNX puis ai_edge_torch ?

---

## 🔗 Ressources clés

- **Solution gagnante Kaggle** : https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution
- **MediaPipe Tasks Android** : https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/android
- **Cactus SDK** : https://github.com/cactus-compute/cactus
- **MediaPipe LLM Inference (AI Edge)** : https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android
- **Ollama (utilisé en PC, pas mobile)** : déjà installé localement avec `gemma4:e2b`
- **Dataset Google ASL Signs** : https://www.kaggle.com/competitions/asl-signs
