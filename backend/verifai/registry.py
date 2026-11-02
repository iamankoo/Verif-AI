"""Model registry: which checkpoints the application knows about and what role each one plays.

Roles (see docs/MODEL_REGISTRY.md):
  * ``production`` -- the checkpoint the server loads at start-up (``VERIFAI_MODEL_PATH``; Model 4 by default). Every request that does not
    name a model uses it.
  * ``rollback``   -- production v2, kept unchanged so the default can be switched back at any time.
  * ``candidate``  -- experiment checkpoints that are selectable for evaluation only (currently E4). A candidate is **not** approved for
    production: it failed the frozen promotion rule (criterion 2, DeepFakeFace missed-fake rate). Selecting one never changes the default.

Nothing here changes a checkpoint, a threshold or an evaluation protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import Settings

# Known checkpoint files -> (registry id, label, short description). A checkpoint that is not listed gets an id derived from its file name.
KNOWN = {
    "verifai_v3_celebdf_qa.pth": ("model4", "Model 4 (DeepFakeFace + Celeb-DF + quality augmentation)",
                                  "Current default (owner decision 2026-11-01). Fixes false alarms on genuine video; weaker than v2 on DeepFakeFace fakes."),
    "verifai_v2_best.pth": ("v2", "v2 (DeepFakeFace only)",
                            "Production v2 and the rollback model. Best on DeepFakeFace; at chance level on held-out Celeb-DF video."),
    "verifai_e4_candidate.pth": ("e4", "E4 (candidate: DeepFakeFace x4 + Celeb-DF + modern generators, quality augmentation)",
                                 "Candidate only. Failed promotion criterion 2 (DeepFakeFace missed-fake rate 0.208 internal, 0.509 cross-generator)."),
}

ROLE_LABEL = {
    "production": "Production default",
    "rollback": "Rollback model",
    "candidate": "Candidate - not approved for production",
}


@dataclass(frozen=True)
class RegisteredModel:
    id: str
    role: str
    label: str
    description: str
    path: Path

    @property
    def role_label(self) -> str:
        return ROLE_LABEL[self.role]

    @property
    def available(self) -> bool:
        return self.path.is_file()


def _entry(path: Path, role: str) -> RegisteredModel:
    mid, label, desc = KNOWN.get(path.name, (path.stem.lower().replace(" ", "-"), path.stem, "Custom checkpoint."))
    return RegisteredModel(id=mid, role=role, label=label, description=desc, path=path)


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        entries: List[RegisteredModel] = [_entry(settings.model_path, "production")]
        extra = [(settings.rollback_model_path, "rollback"), (settings.candidate_model_path, "candidate")]
        for p, role in extra:
            if p and Path(p) != settings.model_path:
                entries.append(_entry(Path(p), role))
        seen: Dict[str, RegisteredModel] = {}
        for e in entries:                    # first entry wins when two roles point at the same file
            seen.setdefault(e.id, e)
        self.entries = seen
        self.default_id = entries[0].id

    def selectable(self, model_id: str) -> bool:
        """The default is always selectable; others only when candidate/rollback selection is enabled and the file exists."""
        e = self.entries.get(model_id)
        if e is None:
            return False
        return model_id == self.default_id or (self.settings.enable_model_selection and e.available)

    def get(self, model_id: Optional[str]) -> Optional[RegisteredModel]:
        return self.entries.get(model_id or self.default_id)

    def describe(self, loaded_ids: List[str]) -> List[dict]:
        out = []
        for e in self.entries.values():
            if e.id != self.default_id and not self.settings.enable_model_selection:
                continue                      # hosted deployments expose only the model they actually serve
            out.append({"id": e.id, "label": e.label, "role": e.role, "role_label": e.role_label, "description": e.description,
                        "checkpoint_file": e.path.name, "available": e.available, "selectable": self.selectable(e.id),
                        "is_default": e.id == self.default_id, "loaded": e.id in loaded_ids})
        return out
