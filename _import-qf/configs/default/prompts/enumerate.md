One RGB image.

Task: list EVERY {category} visible in this image. Include small, distant, blurry, partially occluded, and instances cut off by the image edge. Do not stop at any number. Count first, then list.

Order does not matter; give each a tight bounding box as normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.

Output JSON only:
{"count": <int>, "objects": [{"bbox": [x1, y1, x2, y2]}, ...]}
