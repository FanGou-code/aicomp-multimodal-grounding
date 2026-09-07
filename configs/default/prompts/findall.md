The red rectangle marks one object in the scene.
Task: list up to 12 objects you are MOST CONFIDENT about — clear outline, nameable at a glance — regardless of category, ordered from left to right. If multiple instances of the red-boxed category exist, include them all first, then fill the remaining slots with other confident objects. Skip tiny clutter, blurry ground debris, and anything you cannot identify precisely.
You must include the object inside the red rectangle. Number them 1..N (N is the total count).
For each object give a short common category name and its bounding box as normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.
Output JSON only:
{"objects": [{"i": 1, "category": "<category name>", "bbox": [x1, y1, x2, y2]}, ...]}