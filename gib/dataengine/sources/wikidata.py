"""WikiData entity linking and the *sibling entity graph*.

The sibling graph is what makes entity swaps principled instead of arbitrary. For
an entity E with ``instance of: C``, its siblings are the other members of C --
apple's siblings under *fruit* are orange, pear, mango. Those are semantically
near and factually distinct, which is precisely the hard-negative property Axis 1
needs and the plausible-deception property Axis 3 needs.

Two modes:

* **offline** (default) -- a curated sibling table covering the COCO/VG
  vocabulary. Deterministic, no network, and sufficient for the released build.
* **live** -- the WikiData REST + SPARQL endpoints, cached to disk. Used to
  extend coverage to entities outside the curated table.

Aliases returned here are also consumed by the scorer's alias-match stage
(Layer 2.3), so an answer of "sofa" is not marked wrong against "couch".
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from gib.config import CACHE_DIR

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "GIB-15K-benchmark/1.0 (research; contact: repo maintainer)"

# --------------------------------------------------------------------------- #
# Curated sibling graph
# --------------------------------------------------------------------------- #

SIBLING_CLASSES: dict[str, list[str]] = {
    "fruit": ["apple", "orange", "banana", "pear", "mango", "peach", "lemon", "plum"],
    "vegetable": ["broccoli", "carrot", "cucumber", "pepper", "onion", "lettuce"],
    "prepared food": ["pizza", "sandwich", "hot dog", "donut", "cake", "burger"],
    "domestic animal": ["dog", "cat", "horse", "sheep", "cow", "goat", "rabbit"],
    "wild animal": ["bear", "zebra", "giraffe", "elephant", "lion", "deer"],
    "bird": ["bird", "duck", "pigeon", "seagull", "owl", "parrot"],
    "land vehicle": ["car", "bus", "truck", "motorcycle", "bicycle", "train", "van"],
    "watercraft": ["boat", "ship", "canoe", "kayak", "ferry"],
    "aircraft": ["airplane", "helicopter", "glider"],
    "sports equipment": [
        "frisbee", "kite", "skateboard", "surfboard", "tennis racket",
        "baseball bat", "skis", "snowboard", "sports ball", "ball",
    ],
    "drinking vessel": ["cup", "mug", "wine glass", "bottle", "glass", "flask"],
    "tableware": ["fork", "knife", "spoon", "bowl", "plate", "tray"],
    "seat": ["chair", "couch", "bench", "stool", "sofa", "armchair"],
    "furniture": ["bed", "dining table", "desk", "cabinet", "shelf", "dresser"],
    "personal accessory": ["backpack", "handbag", "suitcase", "tie", "umbrella", "wallet"],
    "electronic device": [
        "laptop", "cell phone", "tv", "keyboard", "mouse", "remote", "monitor", "tablet",
    ],
    "kitchen appliance": ["microwave", "oven", "toaster", "refrigerator", "blender", "kettle"],
    "timepiece": ["clock", "watch", "sundial", "hourglass"],
    "container": ["vase", "jar", "box", "basket", "bucket", "pot"],
    "written work": ["book", "magazine", "newspaper", "notebook", "journal"],
    "plant": ["potted plant", "tree", "flower", "bush", "cactus"],
    "street furniture": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "toy": ["teddy bear", "doll", "puzzle", "kite", "ball"],
}

ENTITY_TO_CLASS: dict[str, str] = {
    e: cls for cls, members in SIBLING_CLASSES.items() for e in members
}

# Curated aliases, used by the scorer as well as by the generators.
ALIASES: dict[str, list[str]] = {
    "couch": ["sofa", "settee"],
    "sofa": ["couch", "settee"],
    "tv": ["television", "tv set", "monitor"],
    "cell phone": ["mobile phone", "phone", "smartphone"],
    "sports ball": ["ball"],
    "ball": ["sports ball"],
    "motorcycle": ["motorbike"],
    "airplane": ["aeroplane", "plane", "jet"],
    "dining table": ["table"],
    "potted plant": ["houseplant", "plant"],
    "gray": ["grey"],
    "grey": ["gray"],
    "bicycle": ["bike"],
    "person": ["man", "woman", "human", "people", "guy", "lady"],
    "hot dog": ["hotdog"],
    "donut": ["doughnut"],
}


def _cache_path() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "wikidata_cache.json"


def _load_cache() -> dict[str, Any]:
    p = _cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    _cache_path().write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")


class SiblingGraph:
    """Sibling lookup with an offline table and an optional live backend."""

    def __init__(self, live: bool = False, timeout: float = 20.0):
        self.live = live
        self.timeout = timeout
        self._cache = _load_cache() if live else {}

    # -- offline ---------------------------------------------------------- #

    def siblings(self, entity: str) -> list[str]:
        """Same-category, factually distinct alternatives for ``entity``."""
        e = entity.strip().lower()
        cls = ENTITY_TO_CLASS.get(e)
        if cls:
            return [s for s in SIBLING_CLASSES[cls] if s != e]
        if self.live:
            return self._live_siblings(e)
        return []

    def entity_class(self, entity: str) -> str:
        return ENTITY_TO_CLASS.get(entity.strip().lower(), "")

    def aliases(self, term: str) -> list[str]:
        t = term.strip().lower()
        out = set(ALIASES.get(t, []))
        if self.live:
            out.update(self._live_aliases(t))
        return sorted(out)

    # -- live ------------------------------------------------------------- #

    def _get_json(self, url: str, params: dict) -> dict:
        for attempt in range(4):
            try:
                r = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                )
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 503):
                    time.sleep(2**attempt)
                    continue
                return {}
            except requests.RequestException:
                time.sleep(2**attempt)
        return {}

    def qid(self, entity: str) -> str:
        key = f"qid:{entity}"
        if key in self._cache:
            return self._cache[key]
        data = self._get_json(
            WIKIDATA_API,
            {
                "action": "wbsearchentities",
                "search": entity,
                "language": "en",
                "format": "json",
                "limit": 1,
                "type": "item",
            },
        )
        hits = data.get("search") or []
        out = hits[0]["id"] if hits else ""
        self._cache[key] = out
        _save_cache(self._cache)
        return out

    def _live_siblings(self, entity: str, limit: int = 12) -> list[str]:
        key = f"sib:{entity}"
        if key in self._cache:
            return self._cache[key]
        q = self.qid(entity)
        if not q:
            self._cache[key] = []
            return []
        # Members of the same `instance of` (P31) class, excluding the entity.
        sparql = f"""
        SELECT DISTINCT ?siblingLabel WHERE {{
          wd:{q} wdt:P31 ?class .
          ?sibling wdt:P31 ?class .
          FILTER(?sibling != wd:{q})
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT {limit}
        """
        data = self._get_json(WIKIDATA_SPARQL, {"query": sparql, "format": "json"})
        out = []
        for b in data.get("results", {}).get("bindings", []):
            label = b.get("siblingLabel", {}).get("value", "").strip().lower()
            if label and not label.startswith("q") and label != entity:
                out.append(label)
        self._cache[key] = out
        _save_cache(self._cache)
        return out

    def _live_aliases(self, entity: str) -> list[str]:
        key = f"alias:{entity}"
        if key in self._cache:
            return self._cache[key]
        q = self.qid(entity)
        if not q:
            self._cache[key] = []
            return []
        data = self._get_json(
            WIKIDATA_API,
            {"action": "wbgetentities", "ids": q, "props": "aliases", "languages": "en", "format": "json"},
        )
        aliases = (
            data.get("entities", {}).get(q, {}).get("aliases", {}).get("en", [])
        )
        out = sorted({a["value"].strip().lower() for a in aliases})
        self._cache[key] = out
        _save_cache(self._cache)
        return out


_DEFAULT: SiblingGraph | None = None


def default_graph(live: bool = False) -> SiblingGraph:
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.live != live:
        _DEFAULT = SiblingGraph(live=live)
    return _DEFAULT
