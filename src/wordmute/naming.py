import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_EPISODE = re.compile(r"[Ss](\d{1,2})[ ._-]?[Ee](\d{1,3})")
_SEASON_DIR = re.compile(r"(season|series)\s*\d+|specials", re.IGNORECASE)


@dataclass
class Names:
    rel: Path  # e.g. Ted Lasso/Season 04 (empty for a film)
    name: str  # e.g. Ted Lasso - S04E01
    work_key: str  # filesystem- and shell-safe unique folder name for scratch files


def derive(source: Path, template: str, show: str | None = None) -> Names:
    stem = source.stem
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80] + "-" + hashlib.sha1(str(source).encode()).hexdigest()[:8]
    found = _EPISODE.search(stem)
    if not found:
        return Names(Path("."), stem, key)
    season, episode = int(found.group(1)), int(found.group(2))
    if not show:
        if _SEASON_DIR.fullmatch(source.parent.name.strip()) and source.parent.parent.name:
            show = source.parent.parent.name
        else:
            show = re.sub(r"[._]+", " ", stem[: found.start()]).strip(" -") or source.parent.name
    show = show.replace("/", "-").strip()
    name = template.format(show=show, season=season, episode=episode, stem=stem).replace("/", "-")
    return Names(Path(show) / f"Season {season:02d}", name, key)
