"""Feature engineering.

Every feature group is a pure function of the base frame that returns new
columns in the same row order. Aggregates only look at transactions with an
earlier timestamp, and the leakage harness (`features.leakage`) proves it.
"""
