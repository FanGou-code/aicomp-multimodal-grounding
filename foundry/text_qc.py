"""Deterministic text QC over assembled query records (frozen adjudication).

Reproduces the 2026-09-06 corpus rulings (train 445 + val 134 edits) in two
stages, applied to the final query text after planning:

1. article engine — a ``with/wearing/holding/carrying`` tail lacking an
   article gets one (a/an by vowel letter), unless the tail is in KEEP,
   ends in a plural/mass word, or matches an EXCEPTIONS ruling.
2. echo table — ``spec/text_qc_echo_table.json`` holds the hand-adjudicated
   head-echo fixes keyed by ``(item_id, before)``; a ruling fires only while
   the item's current query still equals its recorded before text, so the
   replay is idempotent and cannot over-apply.

The word-count field is deliberately left at its pre-QC value (the historical
pipeline audited pre-QC counts); edit logging matches the historical
text_edits.json record shape. The constants below are adjudication DATA of
record — change them only with the administrator.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ECHO_TABLE_PATH = Path(__file__).resolve().parents[1] / "spec" / "text_qc_echo_table.json"

TAIL_RE = re.compile(r"\b(with|wearing|holding|carrying) (?!(?:a |an |the ))(.+)$")

KEEP = {
    "large antlers", "dense foliage", "wooden planks", "bushes", "tall trees", "leafy trees", "spots",
    "long tail feathers", "white tail feathers", "blue tail feathers", "green wings",
    "shorter tail feathers", "wheels", "platform with wheels", "cart with wheels", "dark clothing",
    "light-colored clothing", "black shorts", "pink socks", "green socks", "yellow socks", "white text",
    "text and logo", "text and model number", "white text and symbols", "white logo",
    "red border and text", "dense shrubbery", "mixed shrubbery", "leafy foliage", "tree foliage",
    "evergreen foliage", "bushy foliage", "dark reddish foliage", "bare branches", "leafy branches",
    "windows", "buds on branches", "raised arms", "black handlebars", "multiple devices", "fluffy fur",
    "muddy fur", "striped fur", "light-colored fur", "white feathers", "long legs", "small antlers",
    "antlers", "smaller antlers", "metal railing with vertical bars", "metal railing with blue accents",
    "broken log segments", "grass on sloped bank", "grass on sloped area", "dark pants",
    "trees behind bridge", "trees and foliage", "small narrow leaves", "branches",
    "tall trees with pink blossoms", "bushes near water", "white text and symbols",
    "gray leggings", "shorts", "pants and white shoes", "wooden slats", "long hair, carrying a bag",
    "dark pants on person",
}
VOWEL = "aeiou"
PLURAL_MASS_LASTWORDS = {
    "pants", "socks", "shorts", "jeans", "antlers", "feathers", "wheels", "branches", "leaves", "trees",
    "bushes", "windows", "buds", "handlebars", "devices", "bars", "segments", "spots", "limbs", "feet",
    "clothing", "foliage", "fur", "water", "grass", "text", "shrubbery", "equipment", "bark", "hair",
    "plants", "ears",
    "signtext",
}

# (new_prep or "" for drop, new_tail)
EXCEPTIONS = {
    "in light clothing": ("", "in light clothing"),
    "in jacket": ("", "in a jacket"),
    "in water": ("", "in the water"),
    "in distance": ("", "in the distance"),
    "in dark jacket": ("", "in a dark jacket"),
    "in light pink clothing": ("", "in light pink clothing"),
    "in shaded area": ("", "in a shaded area"),
    "perched on branch": ("", "perched on a branch"),
    "perched on middle branch": ("", "perched on a middle branch"),
    "perched on lower branch": ("", "perched on a lower branch"),
    "perched on a branch tip": ("", "perched on a branch tip"),
    "perched on boat edge": ("", "perched on the boat edge"),
    "on desk": ("", "on a desk"),
    "on windowsill": ("", "on a windowsill"),
    "on stairs": ("", "on the stairs"),
    "on wooden structure near stairs": ("", "on a wooden structure near stairs"),
    "on grass": ("", "on grass"),
    "in grass": ("", "in grass"),
    "head down": ("with", "its head down"),
    "head visible": ("with", "its head visible"),
    "head visible, alert posture": ("with", "its head visible, alert posture"),
    "head": ("with", "its head"),
    "horn visible": ("with", "its horn visible"),
    "hump on back": ("with", "a hump on its back"),
    "saddle on back": ("with", "a saddle on its back"),
    "brown patches on back": ("with", "brown patches on its back"),
    "white patch on rear": ("with", "a white patch on its rear"),
    "white marking on face": ("with", "a white marking on its face"),
    "taxi sign on roof": ("with", "a taxi sign on its roof"),
    "shirt on person": ("with", "a shirt on"),
    "partially visible": ("", "partially visible"),
    "rock with peacock": ("with", "a rock"),
    "small dark shape on water": ("with", "a small dark shape on the water"),
    "wooden boat on water": ("with", "a wooden boat on the water"),
    "canopy on boat": ("with", "a canopy on the boat"),
    "large rock in water": ("with", "a large rock in the water"),
    "rock in water area": ("with", "a rock in the water"),
    "rocky formation near water": ("with", "a rocky formation near the water"),
    "animal grazing near water puddle": ("with", "an animal grazing near a water puddle"),
    "animal grazing at trough": ("with", "an animal grazing at a trough"),
    "animal standing in shallow water": ("with", "an animal standing in shallow water"),
    "large bear sitting on ground": ("with", "a large bear sitting on the ground"),
    "monkey sitting on rock": ("with", "a monkey sitting on a rock"),
    "lion lying on grass": ("with", "a lion lying on grass"),
    "peacock standing on rock": ("with", "a peacock standing on a rock"),
    "small animal on concrete patch": ("with", "a small animal on a concrete patch"),
    "small animal on ground": ("with", "a small animal on the ground"),
    "small size, sitting on ground": ("with", "a small size, sitting on the ground"),
    "small size near fence": ("with", "a small size near a fence"),
    "small car parked near cones": ("with", "a small car parked near the cones"),
    "small vehicle in distance": ("with", "a small vehicle in the distance"),
    "small square shape on wall": ("with", "a small square shape on the wall"),
    "small utility box on sidewalk": ("with", "a small utility box on the sidewalk"),
    "box on trolley": ("with", "a box on a trolley"),
    "box truck with logo on front": ("with", "a box truck with a logo on the front"),
    "sedan parked on road": ("with", "a sedan parked on the road"),
    "sneaker with number 197": ("with", "a sneaker with the number 197"),
    "speed limit sign with number 5": ("with", "a speed limit sign with the number 5"),
    "blue vehicle with speed limit sign with number 5": ("with", "a blue vehicle with a speed limit sign with the number 5"),
    "taxi": ("with", "a taxi"),
    "taxi sign on roof ": ("with", "a taxi sign on its roof"),
    "person wearing backpack": ("with", "a person wearing a backpack"),
    "person wearing dark pants": ("with", "a person wearing dark pants"),
    "person wearing hat": ("with", "a person wearing a hat"),
    "person in wheelchair": ("with", "a person in a wheelchair"),
    "person near horse": ("with", "a person near a horse"),
    "child sitting on pony": ("with", "a child sitting on a pony"),
    "wearing graphic t-shirt": ("with", "wearing a graphic t-shirt"),
    "wearing hat": ("with", "wearing a hat"),
    "wearing backpack": ("with", "wearing a backpack"),
    "carrying bag": ("with", "carrying a bag"),
    "near building entrance": ("", "near the building entrance"),
    "near the building entrance": ("", "near the building entrance"),
    "near water": ("", "near water"),
    "by water's edge": ("", "by the water's edge"),
    "short hair, crouching posture": ("with", "short hair, crouching posture"),
    "long hair, crouching posture": ("with", "long hair, crouching posture"),
    "ornate railing on bridge": ("with", "an ornate railing on the bridge"),
    "sedan with visible front grille": ("with", "a sedan with a visible front grille"),
    "plush toy with yellow belly": ("with", "a plush toy with a yellow belly"),
    "plush toy with pink ribbon": ("with", "a plush toy with a pink ribbon"),
    "plush toy with rounded shape": ("with", "a plush toy with a rounded shape"),
    "plush toy with cartoonish face": ("with", "a plush toy with a cartoonish face"),
    "plush toy with yellow detail": ("with", "a plush toy with a yellow detail"),
    "plush toy with blue bow": ("with", "a plush toy with a blue bow"),
    "plush toy with yellow bow": ("with", "a plush toy with a yellow bow"),
    "plush toy with white face": ("with", "a plush toy with a white face"),
    "office chair with black seat": ("with", "an office chair with a black seat"),
    "wheeled platform with handle": ("with", "a wheeled platform with a handle"),
    "wooden door with handle": ("with", "a wooden door with a handle"),
    "wooden door with metal handle": ("with", "a wooden door with a metal handle"),
    "double door with handle": ("with", "a double door with a handle"),
    "sneaker with white sole": ("with", "a sneaker with a white sole"),
    "shoe with white sole": ("with", "a shoe with a white sole"),
    "wired keyboard with cord": ("with", "a wired keyboard with a cord"),
    "cart with metal frame": ("with", "a cart with a metal frame"),
    "stone wall with metal railing": ("with", "a stone wall with a metal railing"),
    "smooth bark with subtle texture": ("with", "smooth bark with a subtle texture"),
    "leafy bush with white blossoms": ("with", "a leafy bush with white blossoms"),
    "leafy plant with broad leaves": ("with", "a leafy plant with broad leaves"),
    "bird with dark markings": ("with", "a bird with dark markings"),
    "anemometer with three cups": ("with", "an anemometer with three cups"),
    "wheeled platform": ("with", "a wheeled platform"),
}


def article(tail: str) -> str:
    return "an" if tail[0].lower() in VOWEL else "a"


def is_plural_or_mass(tail: str) -> bool:
    first_np = re.split(r"\b(?:with|on|in|near|by|along|behind|at|over|under|for|from)\b", tail)[0]
    last = re.findall(r"[a-z]+", first_np.lower())
    return bool(last) and last[-1] in PLURAL_MASS_LASTWORDS


def adjudicate(tail: str) -> tuple[str, str] | None:
    """Return (new_prep, new_tail); None = keep unchanged."""
    tail = tail.strip()
    if tail in EXCEPTIONS:
        return EXCEPTIONS[tail]
    if tail in KEEP or is_plural_or_mass(tail):
        return None
    return ("WITH", article(tail) + " " + tail)


def qc_query(query: str) -> tuple[str, str] | None:
    """Article-engine stage: (fixed_query, matched_tail), or None unchanged."""
    match = TAIL_RE.search(query)
    if not match:
        return None
    prep, tail = match.group(1), match.group(2).strip()
    verdict = adjudicate(tail)
    if verdict is None:
        return None
    new_prep, new_tail = verdict
    if new_prep == "WITH":
        new_query = query[: match.start()] + f"{prep} {new_tail}"
    elif new_prep == "":
        new_query = query[: match.start()] + new_tail
    else:
        new_query = query[: match.start()] + f"{new_prep} {new_tail}"
    return " ".join(new_query.split()), tail


def load_echo_table() -> dict[tuple[str, str], str]:
    data = json.loads(ECHO_TABLE_PATH.read_text(encoding="utf-8"))
    return {(entry["item_id"], entry["before"]): entry["after"] for entry in data}


def apply_text_qc(records: list) -> list[dict]:
    """Run both QC stages over assembled records in place; return the edit log.

    ``records`` items are AssemblyRecord dataclasses (sample_id, object_index,
    query, edited). The returned log entries match the historical
    text_edits.json shape: {item_id, before, after, reason}.
    """
    echo = load_echo_table()
    edits: list[dict] = []
    for record in records:
        item_id = f"{record.sample_id}#{record.object_index:02d}"
        query = record.query
        fixed = qc_query(query)
        if fixed is not None:
            new_query, tail = fixed
            edits.append({
                "item_id": item_id, "before": query, "after": new_query,
                "reason": f"article/echo adjudication: tail='{tail}'",
            })
            query = new_query
        echo_after = echo.get((item_id, query))
        if echo_after is not None and echo_after != query:
            edits.append({
                "item_id": item_id, "before": query, "after": echo_after,
                "reason": "head-echo adjudication",
            })
            query = echo_after
        if query != record.query:
            record.query = query
            record.edited = True
    return edits
