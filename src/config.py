"""
AinSeba - Configuration Management
Centralized settings for the entire project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ============================================
# Path Configuration
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# Chunking Configuration
# ============================================
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", 600))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", 100))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ============================================
# Logging Configuration
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================
# Law Document Registry
# Each entry contains metadata about a law PDF.
# This registry drives the entire ingestion pipeline.
# ============================================
LAW_REGISTRY = [
    {
        "id": "labour_act_2006",
        "name": "Bangladesh Labour Act 2006",
        "filename": "bangladesh_labour_act_2006.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-952.html",
        "priority": "P0",
        "category": "Employment",
        "year": 2006,
        "language": "english",
    },
    {
        "id": "penal_code_1860",
        "name": "The Penal Code 1860",
        "filename": "penal_code_1860.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-11.html",
        "priority": "P0",
        "category": "Criminal Law",
        "year": 1860,
        "language": "english",
    },
    {
        "id": "consumer_rights_2009",
        "name": "Consumer Rights Protection Act 2009",
        "filename": "consumer_rights_protection_act_2009.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-1035.html",
        "priority": "P0",
        "category": "Consumer Rights",
        "year": 2009,
        "language": "english",
    },
    {
        "id": "cyber_security_2023",
        "name": "Cyber Security Act 2023",
        "filename": "cyber_security_act_2023.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-details-1470.html",
        "priority": "P0",
        "category": "Cyber Law",
        "year": 2023,
        "language": "english",
    },
    {
        "id": "rent_control_1991",
        "name": "The Rent Control Act 1991",
        "filename": "rent_control_act_1991.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-786.html",
        "priority": "P1",
        "category": "Property",
        "year": 1991,
        "language": "english",
    },
    {
        "id": "muslim_family_law_1961",
        "name": "Muslim Family Laws Ordinance 1961",
        "filename": "muslim_family_laws_ordinance_1961.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-305.html",
        "priority": "P1",
        "category": "Family Law",
        "year": 1961,
        "language": "english",
    },
    {
        "id": "companies_act_1994",
        "name": "The Companies Act 1994",
        "filename": "companies_act_1994.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-788.html",
        "priority": "P1",
        "category": "Business",
        "year": 1994,
        "language": "english",
    },
    {
        "id": "constitution_bd",
        "name": "The Constitution of the People's Republic of Bangladesh",
        "filename": "constitution_of_bangladesh.pdf",
        "source_url": "http://bdlaws.minlaw.gov.bd/act-367.html",
        "priority": "P2",
        "category": "Constitutional Law",
        "year": 1972,
        "language": "english",
    },
]


def get_law_by_id(law_id: str) -> dict | None:
    """Look up a law entry by its ID."""
    for law in LAW_REGISTRY:
        if law["id"] == law_id:
            return law
    return None


def get_laws_by_priority(priority: str) -> list[dict]:
    """Get all laws matching a priority level (P0, P1, P2)."""
    return [law for law in LAW_REGISTRY if law["priority"] == priority]
