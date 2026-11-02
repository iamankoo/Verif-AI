def jpeg_roundtrip(face_0_255: torch.Tensor, quality: int = TRAINING_JPEG_QUALITY) -> torch.Tensor:
    """(3,H,W) float 0..255 -> same shape after JPEG encode/decode exactly as training crops were stored."""
    arr = face_0_255.detach().cpu().float().clamp(0, 255).round().byte().permute(1, 2, 0).numpy()
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    img = Image.open(io.BytesIO(buf.tobytes())).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()


def normalise(x_0_255: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) float 0..255 -> ImageNet-normalised; resized to 224 if needed."""
    x = x_0_255.float() / 255.0
    if tuple(x.shape[-2:]) != (INPUT_SIZE, INPUT_SIZE):
        x = torch.nn.functional.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_faces(crops: Sequence[torch.Tensor], match_training_jpeg: bool = True) -> torch.Tensor:
    """List of (3,H,W) crops (0..255) -> (N,3,224,224) model input."""
    if not crops:
        return torch.empty(0, 3, INPUT_SIZE, INPUT_SIZE)
    proc: List[torch.Tensor] = [jpeg_roundtrip(c) if match_training_jpeg else c.detach().cpu().float() for c in crops]
    return normalise(torch.stack(proc))
