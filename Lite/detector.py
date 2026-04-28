import cv2
import numpy as np
import mediapipe as mp

from pathlib import Path
from typing import Optional
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

PATH = Path('detector.task')

class Detector:
    __FACE_OVAL = [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]

    def __init__(
            self,
            path  : Path  = PATH,
            size  : int   = 112,
            ratio : float = 0.85,
        ):
        self._size  = size
        self._ratio = ratio
        self._mp_lmr = None
        self.__load(path)

    def __load(self, model_path: Path) -> None:
        if model_path.exists():
            try:
                self._mp_lmr = mp_vision.FaceLandmarker.create_from_options(
                    mp_vision.FaceLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                )
            except Exception as e:
                raise ImportError(f'MediaPipe FaceLandmarker Error: {e}.')

    def __lm_xy(self, lm, w, h):
        if isinstance(lm, tuple):
            return lm[0] * w, lm[1] * h
        return lm.x * w, lm.y * h

    def detect(self, img_bgr: np.ndarray) -> tuple[Optional[list], Optional[tuple]]:
        '''
        Process a single BGR image.
        Returns
        -------
        - Optional[list]   MediaPipe 478 landmarks
        - Optional[tuple]  Bounding box (x, y, w, h)
        '''
        N = (None, None)

        if self._mp_lmr is None:
            return N

        result = self._mp_lmr.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            )
        )

        if not result.face_landmarks:
            return N

        landmarks = result.face_landmarks[0]
        h, w = img_bgr.shape[:2]

        xs = [l.x * w for l in landmarks]
        ys = [l.y * h for l in landmarks]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        pad = int((x2 - x1) * 0.15)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        bbox = (x1, y1, x2 - x1, y2 - y1)

        return list(landmarks), bbox

    def align(self, img_bgr: np.ndarray, landmarks: list, bbox: tuple) -> Optional[np.ndarray]:
        '''
        Align face to canonical pose using eye corners.
        Returns size×size RGB np.ndarray or None.
        '''
        def _align_face(img_bgr, landmarks, size, ratio):
            h, w = img_bgr.shape[:2]

            def lm(idx):
                x, y = self.__lm_xy(landmarks[idx], w, h)
                return np.array([x, y], dtype=np.float32)

            try:
                face_points = np.array([lm(i) for i in self.__FACE_OVAL])
                l_eye = (lm(33) + lm(133)) / 2
                r_eye = (lm(362) + lm(263)) / 2
            except Exception:
                return None

            dY = r_eye[1] - l_eye[1]
            dX = r_eye[0] - l_eye[0]
            angle = float(np.degrees(np.arctan2(dY, dX)))

            mass_center  = np.mean(face_points, axis=0)
            rotation_mx  = cv2.getRotationMatrix2D(tuple(mass_center), angle, 1.0)
            rotated_pts  = np.hstack(
                [face_points, np.ones((len(face_points), 1))]
            ).dot(rotation_mx.T)

            min_x, min_y = np.min(rotated_pts, axis=0)
            max_x, max_y = np.max(rotated_pts, axis=0)
            scale = (size * ratio) / max(max_x - min_x, max_y - min_y - 10)

            M = cv2.getRotationMatrix2D(tuple(mass_center), angle, scale)
            M[0, 2] += (size // 2) - mass_center[0]
            M[1, 2] += (size // 2) - mass_center[1]

            aligned = cv2.warpAffine(
                img_bgr, M, (size, size),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101
            )
            return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        if landmarks is not None:
            aligned = _align_face(img_bgr, landmarks, self._size, self._ratio)
            if aligned is not None:
                return aligned

        # fallback — bbox crop
        x, y, w, h = [int(v) for v in bbox]
        crop = img_bgr[y:y+h, x:x+w]
        if crop.size != 0:
            return cv2.cvtColor(
                cv2.resize(crop, (self._size, self._size)),
                cv2.COLOR_BGR2RGB
            )

        return None

    def detect_align(self, img_bgr: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        '''
        Detect and align face in a single call.
        Returns
        -------
        - bool                 True if face detected and aligned successfully
        - Optional[np.ndarray] Aligned size×size RGB image, or None if failed
        '''
        landmarks, bbox = self.detect(img_bgr)
        if landmarks is None and bbox is None:
            return False, None
        
        aligned = self.align(img_bgr, landmarks, bbox)
        if aligned is None:
            return False, None
        
        return True, aligned
