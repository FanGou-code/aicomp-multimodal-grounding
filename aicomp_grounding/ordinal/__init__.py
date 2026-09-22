"""Ordinal post-processing: locate the k-th instance of a category, deterministically.

This subpackage fixes the one failure mode the fine-tuned VLMs cannot fix from
the inside: autoregressive decoding has no discrete counter, so "the third
person from the left" is answered by writing coordinates, not by counting.
The module moves counting out of the decoder and into code.

Pipeline position (per model, before WBF)::

    model box -> parse (text only) -> enumerate x2 (base weights, RGB) -> select

Each of the serving models walks that whole flow for itself; the three
selections are then fused by WBF as usual.

Ranking key
    Left edge ``x1``.  This is the key the enumeration prompt asks the model to
    order by, and the key the reviewer's numbered view uses.

Axes (closed set)
    ``x``      left edge                  asc = leftmost,  desc = rightmost
    ``y``      top edge                   asc = topmost,   desc = bottommost
    ``depth``  16-bit mm median in the box asc = nearest,  desc = farthest
    ``ir``     infrared mean in the box    asc = coolest,   desc = warmest
    ``area``   box area                    asc = smallest,  desc = largest

    An axis whose data is missing is *unsupported*: no replacement happens.
    The axis set is closed because every axis must be computable from what the
    sample actually carries (boxes, the 16-bit depth map, the infrared image).

Gates
    All of them must pass; any failure keeps the base box unchanged.

    1. parse valid: category non-empty, ``k >= 1``, axis in the set, direction asc/desc
    2. every run is a well-formed instance list
    3. neither run truncated: self-reported count == number of listed instances
    4. the two runs reconcile one-to-one (IoU >= 0.5) with nothing unmatched
    5. ``k <= N``
    6. the axis is computable for every instance

Replacement rule
    The base box enters only the last comparison: k-th instance is the same
    object -> keep the base box (a WBF coordinate is more precise than a single
    enumeration); a different object or no usable base box -> replace with it.
"""
