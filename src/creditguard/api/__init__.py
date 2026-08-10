"""Phase 8: FastAPI real-time scoring service.

A thin transport layer over `creditguard.scoring.engine` -- validation,
authentication, error handling, observability and persistence of the
request live here; scoring logic does not.
"""

from __future__ import annotations
