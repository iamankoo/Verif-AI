"""Model registry: which checkpoints the application knows about and what role each one plays.

Roles (see docs/E4_INTEGRATION.md):
  * ``default``     -- the checkpoint the server loads at start-up (``VERIFAI_MODEL_PATH``). Every request that does not name a model uses it.
                       Locally this is E4 (the model chosen for the examiner demonstration); the hosted free-tier demo serves v2.
  * ``alternative`` -- another trained model that can be selected per analysis (Model 4, E4 when it is not the default).
  * ``rollback``    -- production v2, kept unchanged so the default can be switched back at any time.
  * ``candidate``   -- extra experiment checkpoints supplied through ``VERIFAI_CANDIDATE_MODEL_PATH`` (none by default).

Being the default does NOT mean a model passed the project's frozen promotion rule: E4 did not (criterion 2). Every model carries a
status note that says so and that note is shown in the dashboard and printed in the PDF report.
Nothing here changes a checkpoint, a threshold or an evaluation protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import Settings

# Known checkpoint files -> (registry id, label, status note). A checkpoint that is not listed gets an id derived from its file name.
KNOWN = {
    "verifai_e4_candidate.pth": (
        "e4", "E4 (DeepFakeFace x4 + Celeb-DF + modern generators, quality augmentation)",
        "Latest trained model, selected as the default for the local examiner demonstration. It failed frozen promotion criterion 2 "
        "(DeepFakeFace missed-fake rate 0.208 internal, 0.509 cross-generator), so it is not approved for production and is not deployed live."),
    "verifai_v3_celebdf_qa.pth": (
        "model4", "Model 4 (DeepFakeFace + Celeb-DF + quality augmentation)",
        "Previous default (owner decision 2026-11-01). Fixes false alarms on genuine video; weaker than v2 on DeepFakeFace fakes."),
    "verifai_v2_best.pth": (
        "v2", "v2 (DeepFakeFace only)",
        "Production v2: the rollback model and the model served by the live deployment. Best on DeepFakeFace; at chance level on held-out Celeb-DF video."),
}

ROLE_LABEL = {
    "default": "Default model",
    "alternative": "Alternative model",
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
    mid, label, note = KNOWN.get(path.name, (path.stem.lower().replace(" ", "-"), path.stem, "Custom checkpoint."))
    return RegisteredModel(id=mid, role=role, label=label, description=note, path=path)


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        entries: List[RegisteredModel] = [_entry(settings.model_path, "default")]
        extra = [(settings.alternative_model_path, "alternative"), (settings.rollback_model_path, "rollback"),
                 (settings.e4_model_path, "alternative"), (settings.candidate_model_path, "candidate")]
        for p, role in extra:
            if p and Path(p) != settings.model_path:
                e = _entry(Path(p), role)
                if role == "alternative" and e.id == "v2":
                    e = _entry(Path(p), "rollback")        # v2 is always the rollback model
                entries.append(e)
        seen: Dict[str, RegisteredModel] = {}
        for e in entries:                    # first entry wins when two roles point at the same file
            seen.setdefault(e.id, e)
        self.entries = seen
        self.default_id = entries[0].id

    def selectable(self, model_id: str) -> bool:
        """The default is always selectable; others only when selection is enabled and the file exists."""
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
