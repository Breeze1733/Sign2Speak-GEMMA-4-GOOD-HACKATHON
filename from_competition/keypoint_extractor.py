"""
Sign2Speak — Étape 1 : Extraction de keypoints MediaPipe Tasks
===============================================================
Utilise les 3 modèles Tasks séparés (Hand, Pose, Face Landmarker)
au lieu de Holistic (deprecated).

Avantages vs Holistic:
- Modèles plus récents et plus précis (surtout mains)
- Activement maintenus par Google
- Même API que le SDK MediaPipe Android → zéro friction pour la démo mobile

Extrait par frame:
- 2 × 21 landmarks mains (x, y, z) = 126 features
- 18 landmarks pose upper-body (x, y, z, visibility) = 72 features  
- 32 landmarks visage clés: lèvres + sourcils (x, y, z) = 96 features
- Total: 294 features/frame (configurable)

Usage:
    # Télécharger les modèles (première fois)
    python keypoint_extractor.py --download-models

    # Webcam live
    python keypoint_extractor.py --mode webcam

    # Fichier vidéo unique
    python keypoint_extractor.py --mode video --input path/to/video.mp4

    # Batch sur dossier de vidéos (ex: WLASL)
    python keypoint_extractor.py --mode batch --input path/to/videos/ --output path/to/output/
"""

import argparse
import time
import urllib.request
from pathlib import Path
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult,
    HandLandmarksConnections,
    PoseLandmarker, PoseLandmarkerOptions, PoseLandmarkerResult,
    PoseLandmarksConnections,
    FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult,
    FaceLandmarksConnections,
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
)


# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = Path(__file__).parent / "models"

MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}

N_HAND = 21
N_POSE = 33

# Upper body: tête, yeux, bouche repères, épaules, coudes, poignets, hanches
UPPER_BODY_INDICES = [
    0,                        # nez
    2, 5,                     # yeux (intérieur gauche/droit)
    7, 8,                     # oreilles
    9, 10,                    # coins bouche
    11, 12, 13, 14, 15, 16,   # épaules, coudes, poignets
    17, 18, 19, 20,           # index et pinky (tips) — utile pour orientation main
    23, 24,                   # hanches (référence)
]

# Landmarks faciaux clés pour ASL (expressions grammaticales)
# Lèvres — contour externe
LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
# Sourcils
BROW_INDICES = [
    70, 63, 105, 66, 107,     # sourcil gauche
    336, 296, 334, 293, 300,   # sourcil droit
]
FACE_KEY_INDICES = LIPS_INDICES + BROW_INDICES


@dataclass
class ExtractionConfig:
    """Configuration de l'extraction."""
    use_hands: bool = True
    use_pose: bool = True
    use_face: bool = True
    face_subset_only: bool = True    # lèvres + sourcils uniquement

    # MediaPipe Tasks params
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5
    min_pose_detection_confidence: float = 0.5
    min_pose_tracking_confidence: float = 0.5
    min_face_detection_confidence: float = 0.5
    min_face_tracking_confidence: float = 0.5

    # Normalisation
    normalize: bool = True

    def features_per_frame(self) -> int:
        n = 0
        if self.use_hands:
            n += N_HAND * 3 * 2          # 2 mains × 21 × (x,y,z)
        if self.use_pose:
            n += len(UPPER_BODY_INDICES) * 4  # × (x,y,z,visibility)
        if self.use_face:
            n_face = len(FACE_KEY_INDICES) if self.face_subset_only else 478
            n += n_face * 3
        return n


# =============================================================================
# Téléchargement des modèles
# =============================================================================

def download_models():
    """Télécharge les modèles .task si absents."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"  ✅ {filename} (déjà présent)")
            continue
        print(f"  ⬇️  Téléchargement {filename}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  ✅ {filename} ({dest.stat().st_size / 1e6:.1f} Mo)")
    print("Modèles prêts.")


# =============================================================================
# Extracteur de keypoints
# =============================================================================

class KeypointExtractor:
    """Extrait et normalise les keypoints via MediaPipe Tasks."""

    def __init__(self, config: ExtractionConfig | None = None):
        self.config = config or ExtractionConfig()
        self._hand_landmarker = None
        self._pose_landmarker = None
        self._face_landmarker = None

    def __enter__(self):
        self._ensure_models_downloaded()

        if self.config.use_hands:
            self._hand_landmarker = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=self.config.num_hands,
                    min_hand_detection_confidence=self.config.min_hand_detection_confidence,
                    min_tracking_confidence=self.config.min_hand_tracking_confidence,
                )
            )

        if self.config.use_pose:
            self._pose_landmarker = PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(MODELS_DIR / "pose_landmarker_lite.task")
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    min_pose_detection_confidence=self.config.min_pose_detection_confidence,
                    min_tracking_confidence=self.config.min_pose_tracking_confidence,
                )
            )

        if self.config.use_face:
            self._face_landmarker = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(MODELS_DIR / "face_landmarker.task")
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=self.config.min_face_detection_confidence,
                    min_tracking_confidence=self.config.min_face_tracking_confidence,
                )
            )

        return self

    def __exit__(self, *args):
        if self._hand_landmarker:
            self._hand_landmarker.close()
        if self._pose_landmarker:
            self._pose_landmarker.close()
        if self._face_landmarker:
            self._face_landmarker.close()

    @staticmethod
    def _ensure_models_downloaded():
        missing = [f for f in MODEL_URLS if not (MODELS_DIR / f).exists()]
        if missing:
            print("📦 Modèles manquants, téléchargement...")
            download_models()

    # -----------------------------------------------------------------
    # Extraction
    # -----------------------------------------------------------------

    def extract_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> dict:
        """
        Extrait les keypoints d'une frame BGR.

        Args:
            frame_bgr: image BGR (OpenCV)
            timestamp_ms: timestamp en millisecondes (doit être croissant en mode VIDEO)

        Returns:
            dict: 'left_hand', 'right_hand', 'pose', 'face' (np.array ou None),
                  plus 'hand_result', 'pose_result', 'face_result' pour le dessin.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        extracted = {}

        # --- Mains ---
        if self._hand_landmarker:
            hand_result: HandLandmarkerResult = self._hand_landmarker.detect_for_video(
                mp_image, timestamp_ms
            )
            extracted['hand_result'] = hand_result

            left, right = None, None
            for i, handedness_list in enumerate(hand_result.handedness):
                label = handedness_list[0].category_name  # "Left" ou "Right"
                landmarks = hand_result.hand_landmarks[i]
                arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                # MediaPipe renvoie "Left"/"Right" depuis le POV de la caméra
                # = inversé par rapport au signeur face à la caméra
                if label == "Right":  # main droite vue caméra = main gauche du signeur
                    left = arr
                else:
                    right = arr
            extracted['left_hand'] = left
            extracted['right_hand'] = right
        else:
            extracted['hand_result'] = None
            extracted['left_hand'] = None
            extracted['right_hand'] = None

        # --- Pose ---
        if self._pose_landmarker:
            pose_result: PoseLandmarkerResult = self._pose_landmarker.detect_for_video(
                mp_image, timestamp_ms
            )
            extracted['pose_result'] = pose_result

            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                pose_all = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                    dtype=np.float32
                )
                extracted['pose'] = pose_all[UPPER_BODY_INDICES]
            else:
                extracted['pose'] = None
        else:
            extracted['pose_result'] = None
            extracted['pose'] = None

        # --- Visage ---
        if self._face_landmarker:
            face_result: FaceLandmarkerResult = self._face_landmarker.detect_for_video(
                mp_image, timestamp_ms
            )
            extracted['face_result'] = face_result

            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]
                face_all = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks],
                    dtype=np.float32
                )
                if self.config.face_subset_only:
                    extracted['face'] = face_all[FACE_KEY_INDICES]
                else:
                    extracted['face'] = face_all
            else:
                extracted['face'] = None
        else:
            extracted['face_result'] = None
            extracted['face'] = None

        return extracted

    def extract_and_flatten(self, frame_bgr: np.ndarray, timestamp_ms: int, data: dict | None = None) -> np.ndarray:
        """
        Extrait, assemble et normalise les keypoints en vecteur 1D.
        C'est cette sortie qui alimente le classifieur.

        Args:
            data: résultat de extract_frame() si déjà appelé (évite un double appel avec le même timestamp)

        Returns:
            np.array de shape (features_per_frame,)
        """
        if data is None:
            data = self.extract_frame(frame_bgr, timestamp_ms)
        parts = []

        # Mains
        if self.config.use_hands:
            lh = data['left_hand'] if data['left_hand'] is not None else np.zeros((N_HAND, 3), dtype=np.float32)
            rh = data['right_hand'] if data['right_hand'] is not None else np.zeros((N_HAND, 3), dtype=np.float32)
            parts.extend([lh.flatten(), rh.flatten()])

        # Pose
        if self.config.use_pose:
            pose = data['pose'] if data['pose'] is not None else np.zeros((len(UPPER_BODY_INDICES), 4), dtype=np.float32)
            parts.append(pose.flatten())

        # Visage
        if self.config.use_face:
            n_face = len(FACE_KEY_INDICES) if self.config.face_subset_only else 478
            face = data['face'] if data['face'] is not None else np.zeros((n_face, 3), dtype=np.float32)
            parts.append(face.flatten())

        vector = np.concatenate(parts).astype(np.float32)

        if self.config.normalize and self.config.use_pose:
            vector = self._normalize(vector, data)

        return vector

    # -----------------------------------------------------------------
    # Normalisation
    # -----------------------------------------------------------------

    def _normalize(self, vector: np.ndarray, data: dict) -> np.ndarray:
        """
        Normalise les coordonnées:
        - Centre: milieu des épaules
        - Scale: distance inter-épaules
        Rend invariant à la position/taille dans l'image.
        """
        pose = data.get('pose')
        if pose is None:
            return vector

        # Épaules = indices 11 et 12 dans la pose complète
        # Dans UPPER_BODY_INDICES, leur position:
        idx_l = UPPER_BODY_INDICES.index(11)
        idx_r = UPPER_BODY_INDICES.index(12)
        l_shoulder = pose[idx_l, :3]
        r_shoulder = pose[idx_r, :3]

        center = (l_shoulder + r_shoulder) / 2.0
        scale = np.linalg.norm(l_shoulder - r_shoulder)

        if scale < 1e-6:
            return vector

        normalized = vector.copy()
        offset = 0

        # Mains: 2 × 21 × 3
        if self.config.use_hands:
            for i in range(2 * N_HAND):
                idx = offset + i * 3
                normalized[idx:idx + 3] = (normalized[idx:idx + 3] - center) / scale
            offset += 2 * N_HAND * 3

        # Pose: upper_body × 4 (normalise x,y,z, garde visibility)
        if self.config.use_pose:
            for i in range(len(UPPER_BODY_INDICES)):
                idx = offset + i * 4
                normalized[idx:idx + 3] = (normalized[idx:idx + 3] - center) / scale
            offset += len(UPPER_BODY_INDICES) * 4

        # Visage: n_face × 3
        if self.config.use_face:
            n_face = len(FACE_KEY_INDICES) if self.config.face_subset_only else 478
            for i in range(n_face):
                idx = offset + i * 3
                normalized[idx:idx + 3] = (normalized[idx:idx + 3] - center) / scale

        return normalized

    # -----------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------

    def draw_landmarks(self, frame_bgr: np.ndarray, data: dict) -> np.ndarray:
        """Dessine les landmarks détectés sur la frame."""
        annotated = frame_bgr.copy()

        # Mains
        hand_result = data.get('hand_result')
        if hand_result and hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, hand_landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # Pose
        pose_result = data.get('pose_result')
        if pose_result and pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated, pose_result.pose_landmarks[0],
                PoseLandmarksConnections.POSE_LANDMARKS,
                mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Visage (contours seulement)
        face_result = data.get('face_result')
        if face_result and face_result.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated, face_result.face_landmarks[0],
                FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

        return annotated


# =============================================================================
# Modes d'exécution
# =============================================================================

def run_webcam(extractor: KeypointExtractor, output_dir: Path | None = None, display: bool = True):
    """Mode webcam live avec visualisation et enregistrement optionnel."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        return

    print("📸 Webcam ouverte — 'q' pour quitter, 'r' pour démarrer/arrêter l'enregistrement")
    print(f"   Features par frame: {extractor.config.features_per_frame()}")
    if not display:
        print("   (mode sans affichage — Ctrl+C pour arrêter)")

    recording = False
    recorded_frames = []
    fps_list = []
    frame_idx = 0

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = frame_idx * 33  # ~30 FPS → 33ms par frame

            data = extractor.extract_frame(frame, timestamp_ms)
            keypoints = extractor.extract_and_flatten(frame, timestamp_ms, data=data)

            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)

            if display:
                annotated = extractor.draw_landmarks(frame, data)
                avg_fps = np.mean(fps_list)

                has_lh = data['left_hand'] is not None
                has_rh = data['right_hand'] is not None
                has_face = data['face'] is not None

                cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"L: {'OK' if has_lh else '--'}  R: {'OK' if has_rh else '--'}  F: {'OK' if has_face else '--'}",
                             (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if recording:
                    recorded_frames.append(keypoints)
                    cv2.putText(annotated, f"REC [{len(recorded_frames)}]",
                                 (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Sign2Speak — Keypoints", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not recording:
                        recording = True
                        recorded_frames = []
                        print("🔴 Enregistrement démarré")
                    else:
                        recording = False
                        if output_dir and recorded_frames:
                            output_dir.mkdir(parents=True, exist_ok=True)
                            filename = output_dir / f"capture_{int(time.time())}.npy"
                            np.save(filename, np.array(recorded_frames))
                            print(f"💾 {filename} ({len(recorded_frames)} frames, shape {np.array(recorded_frames).shape})")
                        recorded_frames = []
            else:
                if recording:
                    recorded_frames.append(keypoints)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n⏹️  Arrêt demandé")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        if recording and output_dir and recorded_frames:
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = output_dir / f"capture_{int(time.time())}.npy"
            np.save(filename, np.array(recorded_frames))
            print(f"💾 {filename} ({len(recorded_frames)} frames)")


def process_video(
    extractor: KeypointExtractor,
    video_path: Path,
    output_path: Path | None = None,
    display: bool = False,
) -> np.ndarray:
    """Traite un fichier vidéo et retourne la séquence de keypoints."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir: {video_path}")
        return np.array([])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 {video_path.name}: {total_frames} frames @ {fps:.0f} FPS")
    if display:
        print("   Affichage activé — 'q' pour quitter")

    all_keypoints = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= 0:
            timestamp_ms = int(frame_idx * (1000.0 / max(fps, 1)))

        if display:
            data = extractor.extract_frame(frame, timestamp_ms)
            keypoints = extractor.extract_and_flatten(frame, timestamp_ms, data=data)
            annotated = extractor.draw_landmarks(frame, data)
            cv2.imshow("Sign2Speak — Keypoints", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            keypoints = extractor.extract_and_flatten(frame, timestamp_ms)

        all_keypoints.append(keypoints)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"   {frame_idx}/{total_frames}")

    cap.release()
    if display:
        cv2.destroyAllWindows()

    sequence = np.array(all_keypoints, dtype=np.float32)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, sequence)
        print(f"💾 {output_path} — shape {sequence.shape}")

    return sequence


def process_batch(extractor: KeypointExtractor, input_dir: Path, output_dir: Path):
    """Traite un dossier de vidéos (ex: WLASL)."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    videos = sorted([
        f for f in input_dir.rglob('*')
        if f.suffix.lower() in video_extensions
    ])

    if not videos:
        print(f"❌ Aucune vidéo trouvée dans {input_dir}")
        return

    print(f"📂 {len(videos)} vidéos trouvées")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, video_path in enumerate(videos):
        print(f"\n[{i + 1}/{len(videos)}] ", end="")
        relative = video_path.relative_to(input_dir)
        output_path = output_dir / relative.with_suffix('.npy')

        if output_path.exists():
            print(f"⏭️  {relative}")
            continue

        process_video(extractor, video_path, output_path)

    print(f"\n✅ Batch terminé — {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sign2Speak — Extraction keypoints MediaPipe Tasks")
    parser.add_argument('--mode', choices=['webcam', 'video', 'batch'], default='webcam')
    parser.add_argument('--input', type=str, default=None,
                        help='Fichier vidéo ou dossier (modes video/batch)')
    parser.add_argument('--output', type=str, default=None,
                        help='Sortie (.npy ou dossier)')
    parser.add_argument('--no-face', action='store_true',
                        help='Désactiver le visage')
    parser.add_argument('--full-face', action='store_true',
                        help='478 landmarks visage au lieu du subset')
    parser.add_argument('--download-models', action='store_true',
                        help='Télécharger les modèles et quitter')
    parser.add_argument('--display', action='store_true', default=None,
                        help='Forcer l\'affichage vidéo (défaut: activé pour webcam, désactivé pour video/batch)')
    parser.add_argument('--no-display', action='store_true',
                        help='Désactiver l\'affichage vidéo (mode headless)')

    args = parser.parse_args()

    if args.download_models:
        download_models()
        return

    config = ExtractionConfig(
        use_face=not args.no_face,
        face_subset_only=not args.full_face,
    )

    print(f"🔧 {config.features_per_frame()} features/frame")
    print(f"   Mains: ✅ | Pose: ✅ | Face: {'subset' if config.use_face and config.face_subset_only else ('full' if config.use_face else '❌')}")

    with KeypointExtractor(config) as extractor:
        if args.mode == 'webcam':
            display = not args.no_display
            output = Path(args.output) if args.output else None
            run_webcam(extractor, output, display=display)

        elif args.mode == 'video':
            if not args.input:
                print("❌ --input requis")
                return
            display = bool(args.display) and not args.no_display
            output = Path(args.output) if args.output else None
            process_video(extractor, Path(args.input), output, display=display)

        elif args.mode == 'batch':
            if not args.input:
                print("❌ --input requis")
                return
            output = Path(args.output) if args.output else Path(args.input) / "keypoints"
            process_batch(extractor, Path(args.input), output)


if __name__ == "__main__":
    main()
