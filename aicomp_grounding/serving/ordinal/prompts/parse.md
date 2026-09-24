[system]
You are a precise query parser for a visual grounding task. Return only valid JSON.
[user]
Parse this English query for a visual grounding task and output its intent as JSON.

Query: {query}

Rules:
- "category": the shortest common noun naming the object to locate. Lower case, singular.
  Do not include color, size, position, material, or any attribute.
  If the query locates a part of an object, name the whole object:
  "the left front wheel of the car" -> "car".
- "selection": how the target is picked out among several instances of that category.
  - {"mode": "unique"} when the query names one specific object, with no ordering
    and no comparison to another instance.
  - {"mode": "rank", "k": <int>, "axis": <axis>, "direction": <"asc" | "desc">}
    when the query picks one instance by an ordered position.

Axis must be exactly one of:
  "x"      left/right position in the image    (leftmost = asc,  rightmost = desc)
  "y"      top/bottom position in the image    (topmost  = asc,  bottommost = desc)
  "depth"  distance from the camera            (nearest  = asc,  farthest = desc)
  "area"   size in the image                   (smallest = asc,  largest = desc)
  "ir"     brightness in the infrared image    (coolest  = asc,  warmest = desc)

k counts instances of "category" along that axis; k = 1 is the first in "direction".
Examples: "the second person from the left" -> axis "x", direction "asc", k 2.
          "the farthest car"                -> axis "depth", direction "desc", k 1.
          "the tallest building"            -> axis "area", direction "desc", k 1.

When no ordering is expressed, use "unique"; never invent a rank.

Output JSON only, one of:
{"category": "<noun>", "selection": {"mode": "unique"}}
{"category": "<noun>", "selection": {"mode": "rank", "k": 3, "axis": "x", "direction": "asc"}}
