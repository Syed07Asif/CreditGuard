"""Reusable Streamlit UI building blocks shared across dashboard pages.

Pages compose these and call `creditguard.dashboard.api_client` -- no
business logic (scoring, banding, decisioning) lives in a page or a
component; it all comes from the API response already computed.
"""
