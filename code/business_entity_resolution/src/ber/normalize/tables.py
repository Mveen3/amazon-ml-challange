"""Container for lookup tables mined from training pairs (see ``ber.mining``)."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Tables:
    token_map: dict = field(default_factory=dict)   # Indic token -> Latin token(s)
    name_abbr: dict = field(default_factory=dict)   # profile -> {short: long}
    addr_abbr: dict = field(default_factory=dict)   # profile -> {short: long}
    comp_map: dict = field(default_factory=dict)    # profile -> {address component: canonical}
    name_affix: dict = field(default_factory=dict)  # profile -> set of tokens noise tends to add to names
    addr_affix: dict = field(default_factory=dict)  # profile -> set of tokens noise tends to add to addresses

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: Path | None) -> "Tables":
        if path is None or not Path(path).exists():
            return cls()
        with open(path, "rb") as f:
            return cls(**pickle.load(f))


EMPTY = Tables()
