from typing import Dict, List, Any
import json
import os

class TemplateRegistry:
    """Registry that loads templates from disk and exposes idx<->template mappings.

    Fail-fast behaviour: raises if the template file is missing or empty.
    """

    def __init__(self, templates_path: str = "src/configs/templates.json") -> None:
        if not os.path.exists(templates_path):
            raise FileNotFoundError(f"Template file not found: {templates_path}")

        with open(templates_path, "r") as f:
            template_data = json.load(f)

        # Allow either {"templates": [...]} or a bare list.
        if isinstance(template_data, dict) and "templates" in template_data:
            self.templates: List[Dict[str, Any]] = template_data["templates"]
        elif isinstance(template_data, list):
            self.templates = template_data
        else:
            raise ValueError("templates.json must contain a list or a dict with key 'templates'.")

        if len(self.templates) == 0:
            raise ValueError("No templates found in templates.json – cannot create high-level action space.")

        # Build fast lookup tables.
        self.idx2template: Dict[int, Dict[str, Any]] = {idx: tpl for idx, tpl in enumerate(self.templates)}
        self.id2idx: Dict[str, int] = {tpl["id"]: idx for idx, tpl in enumerate(self.templates)}

    # ------------------------------ public helpers ------------------------------
    def __len__(self) -> int:
        return len(self.templates)

    def get_template(self, idx: int) -> str:
        """Return the raw template string for a given index."""
        return self.idx2template[idx]["template"]

    def get_full_entry(self, idx: int) -> Dict[str, Any]:
        """Return the full template dict (id/type/instruction/template)."""
        return self.idx2template[idx]

    def id_to_idx(self, template_id: str) -> int:
        return self.id2idx[template_id]

    def idx_to_id(self, idx: int) -> str:
        return self.idx2template[idx]["id"] 