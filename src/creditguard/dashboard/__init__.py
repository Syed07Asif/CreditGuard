"""Phase 9: the CreditGuard Streamlit dashboard.

A pure HTTP client of the Phase 8 API (`creditguard.dashboard.api_client`)
-- nothing under this package imports the model, the feature pipeline or
`creditguard.scoring`/`creditguard.models` directly. That boundary is what
keeps the API's deployment contract honest: if the dashboard needed direct
access to any of those, the API wouldn't actually be complete.
"""
