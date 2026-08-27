"""Sentiment Lab utility package.

Modules
=======
    preprocessing      conservative text cleaning + input validation
    sentiment_analyzer VADER wrapper (v1 contract preserved)
    language_detect    lightweight English / Hindi / Hinglish detection
    multilingual       lazily-loaded transformer model (cached)
    engine             the single routing pipeline all sources feed into
    bulk_processor     CSV column discovery + batch scoring
    youtube            YouTube Data API v3 comment retrieval
    blog               WordPress REST / structured data / HTML comment reader
"""
