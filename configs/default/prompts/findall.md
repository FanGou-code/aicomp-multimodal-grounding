The red rectangle marks one reference object in the scene.

Step 1. Name the category of the object inside the red rectangle, then count how many instances of that category the whole frame contains.

Step 2. Branch on that count:
- If the frame contains TWO OR MORE instances of that category: list EVERY instance of it — small, distant, blurry, partially occluded, cut off by the image edge. Do not stop at any number. Do not cap the list.
- If the frame contains EXACTLY ONE instance (the reference object itself): do not list that category. Instead list the other objects in the frame you are most confident about — clear outline, nameable at a glance, any category. Skip tiny clutter and anything you cannot identify precisely.

Order the list from left to right. Number them 1..N. For each object give a short common category name and its bounding box as normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.

Output JSON only:
{"mode": "instances" or "other", "category": "<the red-boxed category>", "count": <int>, "objects": [{"i": 1, "category": "<category name>", "bbox": [x1, y1, x2, y2]}, ...]}

"mode" is "instances" when you listed every instance of the red-boxed category, "other" when you listed other confident objects instead. "count" must equal the number of objects you listed.
