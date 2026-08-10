"""Phase 7: credit scoring engine, risk categories and lending recommendations.

`scorecard.py` converts a calibrated default probability into a 300-900
credit score; `categories.py` buckets a score into a risk band;
`recommendation.py` turns probability + score + policy rules into an
APPROVE/REVIEW/REJECT decision; `engine.py` is the single entry point
(`score_application`) that ties them together with the Phase 6 model and
the Phase 4 feature pipeline.
"""

from __future__ import annotations
