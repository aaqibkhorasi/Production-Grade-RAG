from dataclasses import dataclass


@dataclass(frozen=True)
class ManifestEntry:
    filename: str
    url: str
    file_type: str  # "docx" | "pdf" | "html"
    effective_date: str  # ISO 8601 date
    program: str  # "core" | "7(a)" | "504" | "affiliation"
    # Floor for extracted text length, set well below what each source actually
    # yields. A fetch that returns a bot-check page or a parse that stops
    # recognising the markup still "succeeds" and produces a few junk chunks,
    # which only shows up much later as that document never being retrievable.
    # Failing ingestion loudly here is far cheaper to diagnose.
    min_chars: int = 0


CORPUS_MANIFEST: list[ManifestEntry] = [
    ManifestEntry(
        filename="sop_50_10_8_1.docx",
        url=(
            "https://legacy.sba.gov/sites/default/files/2026-08/"
            "SOP%2050%2010%208.1%20effective%2010.1.2026_0.docx"
        ),
        file_type="docx",
        effective_date="2026-10-01",
        program="core",
        min_chars=100_000,
    ),
    ManifestEntry(
        filename="cfr_121_affiliation.html",
        url="https://www.ecfr.gov/current/title-13/chapter-I/part-121",
        file_type="html",
        effective_date="2026-09-22",
        program="affiliation",
        min_chars=50_000,
    ),
    ManifestEntry(
        filename="notice_7a_fees_fy2026.pdf",
        url=(
            "https://legacy.sba.gov/sites/default/files/2025-08/"
            "Info%20Notice%20-%207(a)%20Fees%20FY%202026%20(FINAL%208-28-25).pdf"
        ),
        file_type="pdf",
        effective_date="2025-08-28",
        program="7(a)",
        min_chars=5_000,
    ),
]
