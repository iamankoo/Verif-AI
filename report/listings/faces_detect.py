    def detect(self, rgb: np.ndarray) -> List[DetectedFace]:
        """rgb: (H,W,3) uint8. Detection runs on a downscaled copy; crops are taken from the original."""
        h, w = rgb.shape[:2]
        scale = min(1.0, self.s.detection_max_side / max(h, w))
        small = cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        try:
            with self._lock:
                boxes, probs = self.mtcnn.detect(small)
        except Exception:        # extremely small / degenerate images
            return []
        if boxes is None or len(boxes) == 0:
            return []
        boxes = np.asarray(boxes, dtype=np.float64) / scale
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        keep = []
        for i, (b, p) in enumerate(zip(boxes, probs)):
            side = min(b[2] - b[0], b[3] - b[1])
            if p >= self.s.face_min_confidence and side >= self.s.face_min_px:
                keep.append(i)
        if not keep:
            return []
        keep.sort(key=lambda i: -((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])))
        keep = keep[: self.s.max_faces_per_image]
        sel = boxes[keep]
        with self._lock:
            crops = self.mtcnn.extract(rgb, sel, None)
        if crops.ndim == 3:
            crops = crops.unsqueeze(0)
        out = []
        for j, i in enumerate(keep):
            b = boxes[i]
            side = float(min(b[2] - b[0], b[3] - b[1]))
            out.append(DetectedFace(box=[float(v) for v in b], confidence=float(probs[i]), crop=crops[j],
                                    min_side_px=side, low_resolution=side < self.s.face_warn_px))
        return out
