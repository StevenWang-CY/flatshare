"""The engine change on vLLM 0.32.0: Flatshare's make-room variant, vLLM #63350 and #63351 backported as patches with
the variant's own, and the pr54530 baseline. build.py writes a variant's scheduler module at engine start; it and the
probes beside the patches run as scripts, with the standard library only.

The engine plugins this package held, LeasePolicy and LeaseConnector, were built for the lease design and removed after
v0.3 (ADR-0006).
"""
