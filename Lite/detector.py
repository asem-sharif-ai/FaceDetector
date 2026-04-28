import cv2
import numpy as np
import mediapipe as mp

from pathlib import Path
from typing import Optional
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

PATH = Path('detector.task')

class Get:
    FACE       = 'face'
    EYES       = 'eyes'
    R_EYE      = 'r_eye'
    L_EYE      = 'l_eye'
    BROWS      = 'brows'
    R_BROW     = 'r_brow'
    L_BROW     = 'l_brow'
    EYES_BROWS = 'eyes_brows'
    R_EYE_BROW = 'r_eye_brow'
    L_EYE_BROW = 'l_eye_brow'
    GLASSES    = 'glasses'
    FOREHEAD   = 'forehead'
    NOSE       = 'nose'
    MOUTH      = 'mouth'
    MASK       = 'mask'
    CHIN       = 'chin'

class Size:
    ALIGN      = 112
    EYES       = (120, 50)
    R_EYE      = (80,  50)
    L_EYE      = (80,  50)
    BROWS      = (120, 35)
    R_BROW     = (80,  35)
    L_BROW     = (80,  35)
    R_EYE_BROW = (90,  55)
    L_EYE_BROW = (90,  55)
    EYES_BROWS = (120, 55)
    GLASSES    = (140, 100)
    FOREHEAD   = (108, 50)
    NOSE       = (50,  60)
    MOUTH      = (100, 60)
    MASK       = (120, 90)
    CHIN       = (90,  55)

class Detector:
    MIRROR = False

    __FACE_OVAL = [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]

    __R_EYE  = [33,  160, 158, 133, 153, 144, 7,   163, 246, 161]
    __L_EYE  = [362, 385, 387, 263, 373, 380, 384, 398, 466, 388]
    __R_BROW = [70,  63,  105, 66,  107, 55,  65,  52,  53,  46 ]
    __L_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    
    __FOREHEAD = [
        10,  338, 297, 332, 284, 251, 109, 67,  103, 54,  21,  162, 127,
        151, 9,   8,   70,  63,  105, 66,  107, 336, 296, 334, 293, 300
    ]

    __NOSE = [1, 2, 5, 4, 19, 94, 64, 98, 97, 96, 3, 195, 327, 294, 328, 326]
    __NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19, 94]

    __MOUTH  = [61,  185, 40,  39,  37,  0,  267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,  181, 91,  146]
    __CHIN   = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 356, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]
    
    __MASK   = [
        64,  98,  97,  96,  3,   195, 327, 294, 328, 326, 129, 209, 49, 131, 134, 51, 281, 363, 360,
        61,  291, 17,  314, 405, 321, 375, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 356,
        323, 361, 288, 397, 365, 379, 378, 400, 377
    ]

    __REGION_LM = {
        Get.FACE       : __FACE_OVAL,
        Get.EYES       : [*__R_EYE, *__L_EYE],
        Get.R_EYE      : __R_EYE,
        Get.L_EYE      : __L_EYE,
        Get.BROWS      : [*__R_BROW, *__L_BROW],
        Get.R_BROW     : __R_BROW,
        Get.L_BROW     : __L_BROW,
        Get.EYES_BROWS : [*__R_EYE, *__L_EYE, *__R_BROW, *__L_BROW],
        Get.R_EYE_BROW : [*__R_EYE, *__R_BROW],
        Get.L_EYE_BROW : [*__L_EYE, *__L_BROW],
        Get.GLASSES    : [*__R_EYE, *__L_EYE, *__R_BROW, *__L_BROW, *__NOSE_BRIDGE],
        Get.NOSE       : __NOSE,
        Get.MOUTH      : __MOUTH,
        Get.MASK       : __MASK,
        Get.CHIN       : __CHIN,
        Get.FOREHEAD   : __FOREHEAD,
    }

    __PADDING = {
        Get.EYES_BROWS : (0.10, 0.20),
        Get.R_EYE_BROW : (0.10, 0.25),
        Get.L_EYE_BROW : (0.10, 0.25),
        Get.EYES       : (0.10, 0.30),
        Get.R_EYE      : (0.10, 0.35),
        Get.L_EYE      : (0.10, 0.35),
        Get.BROWS      : (0.05, 0.55),
        Get.R_BROW     : (0.10, 0.60),
        Get.L_BROW     : (0.10, 0.60),
        Get.GLASSES    : (0.25, 0.10),
        Get.NOSE       : (0.20, 0.00),
        Get.MOUTH      : (0.10, 0.30),
        Get.MASK       : (0.05, 0.05),
        Get.CHIN       : (0.10, 0.10),
        Get.FOREHEAD   : (0.10, 0.25),
    }

    def __init__(self, path: Path = PATH, ratio: float = 0.85):
        self._size   = Size.ALIGN
        self._ratio  = ratio
        self._mp_lmr = None

        self.__load(path)

    def __load(self, model_path: Path) -> None:
        if not model_path.exists():
            return
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

    def __affine_matrix(self, img_bgr: np.ndarray, landmarks: list) -> Optional[np.ndarray]:
        h, w = img_bgr.shape[:2]
        size, ratio = self._size, self._ratio

        def lm(idx):
            x, y = self.__lm_xy(landmarks[idx], w, h)
            return np.array([x, y], dtype=np.float32)

        try:
            face_pts = np.array([lm(i) for i in self.__FACE_OVAL])
            l_eye = (lm(33)  + lm(133)) / 2
            r_eye = (lm(362) + lm(263)) / 2
        except Exception:
            return None

        angle  = float(np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0])))
        center = np.mean(face_pts, axis=0)
        rot_mx = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        rot_pt = np.hstack([face_pts, np.ones((len(face_pts), 1))]).dot(rot_mx.T)
        mn, mx = rot_pt.min(axis=0), rot_pt.max(axis=0)
        scale  = (size * ratio) / max(mx[0] - mn[0], mx[1] - mn[1] - 10)

        M = cv2.getRotationMatrix2D(tuple(center), angle, scale)
        M[0, 2] += (size // 2) - center[0]
        M[1, 2] += (size // 2) - center[1]

        return M

    def __project_landmarks(self, landmarks: list, m: np.ndarray, w: int, h: int) -> np.ndarray:
        pts = np.array(
            [self.__lm_xy(lm, w, h) for lm in landmarks],
            dtype=np.float32
        )
        return np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)]).dot(m.T)

    def detect(self, img_bgr: np.ndarray) -> tuple[Optional[list], Optional[tuple]]:
        if self._mp_lmr is None:
            return (None, None)

        result = self._mp_lmr.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            )
        )
        if not result.face_landmarks:
            return (None, None)

        h, w = img_bgr.shape[:2]
        landmarks = result.face_landmarks[0]

        xs = [l.x * w for l in landmarks]
        ys = [l.y * h for l in landmarks]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        pad  = int((x2 - x1) * 0.15)
        bbox = (
            max(0, x1 - pad) , max(0, y1 - pad),
            min(w, x2 + pad) - max(0, x1 - pad),
            min(h, y2 + pad) - max(0, y1 - pad),
        )
        return list(landmarks), bbox

    def align(self, img_bgr  : np.ndarray, landmarks: list, bbox     : tuple) -> Optional[np.ndarray]:
        if (M := self.__affine_matrix(img_bgr, landmarks)) is not None:
            return cv2.cvtColor(
                cv2.warpAffine(
                    img_bgr, M, (self._size, self._size),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT_101
                ),
                cv2.COLOR_BGR2RGB
            )

        x, y, w, h = [int(v) for v in bbox]
        if (crop := img_bgr[y:y+h, x:x+w]).size != 0:
            return cv2.cvtColor(cv2.resize(crop, (self._size, self._size)), cv2.COLOR_BGR2RGB)

        return None

    def detect_align(self, img_bgr: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        lm, bbox = self.detect(img_bgr)
        if lm is None:
            return False, None

        aligned = self.align(img_bgr, lm, bbox)
        return (aligned is not None, aligned)

    def detect_section(
            self,
            img_bgr  : np.ndarray,
            section  : str,
            landmarks: Optional[list] = None,
    ) -> tuple[bool, Optional[np.ndarray]]:

        if section not in self.__REGION_LM:
            return False, None

        if landmarks is None:
            landmarks, _ = self.detect(img_bgr)
            if landmarks is None:
                return False, None

        if Detector.MIRROR:
            section = {
                Get.R_EYE      : Get.L_EYE,
                Get.L_EYE      : Get.R_EYE,
                Get.R_BROW     : Get.L_BROW,
                Get.L_BROW     : Get.R_BROW,
                Get.R_EYE_BROW : Get.L_EYE_BROW,
                Get.L_EYE_BROW : Get.R_EYE_BROW,
            }.get(section, section)

        h_img, w_img = img_bgr.shape[:2]
        out_w, out_h = getattr(Size, section.upper(), (Size.ALIGN, Size.ALIGN))

        M = self.__affine_matrix(img_bgr, landmarks)
        if M is None:
            return False, None

        projected_lm = self.__project_landmarks(landmarks, M, w_img, h_img)

        aligned_bgr = cv2.warpAffine(
            img_bgr, M, (self._size, self._size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101
        )

        hp, vp = self.__PADDING.get(section, (0.15, 0.15))

        pts = projected_lm[self.__REGION_LM[section]]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        pw = (x2 - x1) * hp
        ph = (y2 - y1) * vp

        if section == Get.CHIN:
            y1 = max(y1, projected_lm[self.__MOUTH, 1].max())
        elif section == Get.FOREHEAD:
            y2 = min(y2, projected_lm[self.__R_BROW + self.__L_BROW, 1].min())

        x1 = int(max(0, x1 - pw))
        y1 = int(max(0, y1 - ph))
        x2 = int(min(self._size, x2 + pw))
        y2 = int(min(self._size, y2 + ph))

        crop = aligned_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return False, None

        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        return True, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
