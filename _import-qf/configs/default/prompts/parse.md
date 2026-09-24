Parse this English query for a visual grounding task and output its intent as JSON.

Query: {query}

Rules:
- "category": the shortest common noun naming the object to locate. Lower case, singular. Do not include color, size, position, material, or any attribute. If the query locates a part of an object, name the whole object.
- "selection": how the target is picked out among several instances of that category.
  - {"mode": "unique"} when the query names one specific object, with no ordering and no comparison to another instance.
  - {"mode": "rank", "k": <int>, "axis": <axis>, "direction": <"asc" | "desc">} when the query picks one instance by an ordered position.

Axis must be exactly one of:
  "x"     left/right position in the image   (leftmost = asc, rightmost = desc)
  "y"     top/bottom position in the image   (topmost  = asc, bottommost = desc)
  "area"  size in the image                  (smallest = asc, largest = desc)

k counts instances of "category" along that axis; k = 1 is the first in "direction".
When no ordering is expressed, use "unique"; never invent a rank.

Output JSON only:
{"category": "<noun>", "selection": {"mode": "unique"}}
{"category": "<noun>", "selection": {"mode": "rank", "k": 3, "axis": "x", "direction": "asc"}}
