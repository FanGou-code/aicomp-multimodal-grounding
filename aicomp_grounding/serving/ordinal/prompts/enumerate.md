[system]
You are a precise visual enumerator. Return only valid JSON.
[user]
One RGB image.

Task: list EVERY {category} visible in this image, ordered by the LEFT EDGE of its box.

- Include every instance: small, distant, blurry, partially occluded, and instances
  cut off by the image edge.
- Do NOT stop at any number. Do NOT skip instances because they are hard to see.
- If you are unsure whether something is a {category}, include it with a low confidence.
- Count first, then list.

For each instance give a tight bounding box as normalized coordinates [x1, y1, x2, y2]:
four decimal fractions where 0 is the left/top edge of the image and 1 is the
right/bottom edge. NEVER use pixel values. Also give a confidence between 0 and 1.

Output JSON only:
{"count": <int>, "instances": [{"bbox": [x1, y1, x2, y2], "confidence": 0.0}, ...]}
