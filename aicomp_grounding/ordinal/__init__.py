"""Ordinal post-processing: locate the k-th instance of a category, deterministically.

Autoregressive decoding has no discrete counter: "the third person from the
left" is answered by writing coordinates, not by counting. This module moves
counting out of the decoder and into code.

Pipeline position (before WBF)::

    parse (text only, greedy) -> enumerate (base weights, RGB, thinking) -> select

``qwen3_5`` walks that flow once and writes one repaired prediction file; the
three serving models are still fused by WBF afterwards, and only the ordinal
model's own box is rewritten.

Enumeration loads base weights, not the LoRA adapter: the adapter was trained on
the "thinking off, box tokens only" format.

Ranking key
    Left edge ``x1``.

Axes (closed set)
    ``x``      left edge                  asc = leftmost,  desc = rightmost
    ``y``      top edge                   asc = topmost,   desc = bottommost
    ``depth``  16-bit mm median in the box asc = nearest,  desc = farthest
    ``ir``     infrared mean in the box    asc = coolest,   desc = warmest
    ``area``   box area                    asc = smallest,  desc = largest

    An axis whose data is missing is *unsupported*: no replacement happens.

Gates
    All of them must pass; any failure keeps the base box unchanged.

    1. parse valid: category non-empty, ``k >= 1``, axis in the set, direction asc/desc
    2. the reply is a well-formed instance list
    3. self-reported count == number of listed instances
    4. count > 0
    5. a thinking block that commits to a number names that same count
    6. ``k <= N``
    7. the axis is computable for every instance

Replacement rule
    The base box enters only the last comparison: k-th instance is the same
    object -> keep the base box; a different object or no usable base box ->
    replace with it.
"""
