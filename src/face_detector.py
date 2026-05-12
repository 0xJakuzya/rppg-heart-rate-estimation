import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from src import config


class FaceDetector:
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=config.FACE_MAX_NUM,
            min_face_detection_confidence=config.FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        h, w = frame.shape[:2]
        return [(int(p.x * w), int(p.y * h)) for p in result.face_landmarks[0]]

    def make_mask(self, frame, landmarks, idxs, crop_top_frac=None, crop_bottom_frac=None):
        pts = np.array([landmarks[i] for i in idxs], np.int32)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        if crop_top_frac is not None or crop_bottom_frac is not None:
            y_min = int(pts[:, 1].min())
            y_max = int(pts[:, 1].max())
            h_roi = y_max - y_min
            if crop_top_frac is not None:
                mask[:y_min + int(h_roi * crop_top_frac)] = 0
            if crop_bottom_frac is not None:
                mask[y_min + int(h_roi * crop_bottom_frac):] = 0
        return mask

    def get_multi_roi_patches(self, frame, landmarks, patch_size: int | None = None):
        patch_size = patch_size or config.ROI_PATCH_SIZE
        return [
            self.extract_patch_from_mask(frame, roi_mask, patch_size)
            for roi_mask in self.get_multi_roi_masks(frame, landmarks)
        ]

    def get_multi_roi_masks(self, frame, landmarks):
        forehead = self.make_mask(frame, landmarks, config.FOREHEAD_IDX, crop_bottom_frac=0.40)
        left_cheek = self.make_mask(frame, landmarks, config.LEFT_CHEEK_IDX)
        right_cheek = self.make_mask(frame, landmarks, config.RIGHT_CHEEK_IDX)
        masks = []
        masks.extend(self.split_mask(forehead, axis=1, parts=4))
        masks.extend(self.split_mask(left_cheek, axis=0, parts=2))
        masks.extend(self.split_mask(right_cheek, axis=0, parts=2))
        return masks

    def extract_patch_from_mask(self, frame, mask: np.ndarray, patch_size: int) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        cropped_frame = frame[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]
        patch = cv2.bitwise_and(cropped_frame, cropped_frame, mask=cropped_mask)
        return cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    def split_mask(self, mask, axis: int, parts: int):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return [mask.copy() for _ in range(parts)]
        coords = xs if axis == 1 else ys
        lo, hi = int(coords.min()), int(coords.max()) + 1
        edges = np.linspace(lo, hi, parts + 1).astype(int)
        split_masks = []
        for i in range(parts):
            part = np.zeros_like(mask)
            if axis == 1:
                selector = (xs >= edges[i]) & (xs < edges[i + 1])
            else:
                selector = (ys >= edges[i]) & (ys < edges[i + 1])
            part[ys[selector], xs[selector]] = 255
            split_masks.append(part)
        return split_masks

    def close(self):
        self.landmarker.close()
