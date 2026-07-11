"""Optional serving utilities that sit *outside* the dataset pipeline.

These are stopgaps/tools (not pipeline stages) and depend only on the ``serve``
extra, so importing this package must not require the core pipeline's runtime.
"""
