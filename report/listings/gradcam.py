    def heatmaps(self, batch: torch.Tensor) -> Optional[Sequence[np.ndarray]]:
        """(N,3,224,224) normalised batch -> list of (224,224,3) uint8 overlays, or None if unavailable."""
        if not self.available or batch.shape[0] == 0:
            return None
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        base = denormalise_to_uint8(batch).astype(np.float32) / 255.0
        out = []
        with self.bundle.lock:
            m = self.bundle.model
            m.eval()
            with torch.enable_grad():
                cam = GradCAM(model=m, target_layers=[m.spatial_net._blocks[-1]])
                try:
                    gray = cam(input_tensor=batch.to(self.bundle.device), targets=[ClassifierOutputTarget(0)] * batch.shape[0])
                finally:
                    try:
                        cam.activations_and_grads.release()
                    except Exception:
                        pass
        for i in range(batch.shape[0]):
            out.append(show_cam_on_image(base[i], gray[i], use_rgb=True))
        return out
