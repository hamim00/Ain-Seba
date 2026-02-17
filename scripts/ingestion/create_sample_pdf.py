"""
AinSeba - Sample PDF Generator
Creates a realistic test PDF that mimics the structure of Bangladesh law documents.
Use this to test the full pipeline before downloading real law PDFs.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install PyMuPDF")
    sys.exit(1)

from src.config import RAW_DATA_DIR


SAMPLE_LAW_TEXT = """
THE SAMPLE WORKERS PROTECTION ACT, 2024

An Act to provide for the protection of workers' rights and welfare
in the People's Republic of Bangladesh.

Whereas it is expedient to make provisions for the protection of the
rights and welfare of workers in Bangladesh;

It is hereby enacted as follows:

PART I
PRELIMINARY

CHAPTER I
INTRODUCTION AND DEFINITIONS

Section 1. Short title, extent and commencement
(1) This Act may be called the Sample Workers Protection Act, 2024.
(2) It extends to the whole of Bangladesh.
(3) It shall come into force on such date as the Government may, by notification in the official Gazette, appoint.

Section 2. Definitions
In this Act, unless there is anything repugnant in the subject or context,
(a) "appropriate Government" means the Government of the People's Republic of Bangladesh;
(b) "employer" means any person who employs, whether directly or through another person, or whether on behalf of himself or any other person, one or more workers in any establishment;
(c) "establishment" means any shop, commercial establishment, industrial establishment, factory, farm, plantation, workshop, or premises where any industry, trade, business, occupation or service is carried on;
(d) "wages" means all remuneration expressed in terms of money or capable of being so expressed which would, if the terms of employment were fulfilled, be payable to a worker;
(e) "worker" means any person employed in any establishment to do any manual, unskilled, skilled, technical, operational, clerical or administrative work for hire or reward.

CHAPTER II
EMPLOYMENT CONDITIONS

Section 3. Terms of employment
(1) Every employer shall, within thirty days of the employment of a worker, provide the worker with a written contract of employment.
(2) The contract of employment shall contain the following particulars:
(a) the name and address of the employer;
(b) the name, address and date of birth of the worker;
(c) the nature of employment;
(d) the date of commencement of employment;
(e) the amount of wages and the period of payment;
(f) the hours of work;
(g) the leave entitlement of the worker.

Section 4. Maximum working hours
(1) No worker shall be required or allowed to work in an establishment for more than eight hours in any day.
(2) No worker shall be required or allowed to work for more than forty-eight hours in any week.
(3) Where a worker works for more than the prescribed hours, the employer shall pay overtime at the rate of twice the ordinary rate of wages.

Section 5. Rest intervals
(1) Every worker who works for more than six consecutive hours shall be entitled to a rest interval of at least one hour.
(2) The rest interval shall not be counted as part of the working hours.

PART II
WAGES AND BENEFITS

CHAPTER III
PAYMENT OF WAGES

Section 6. Time of payment of wages
(1) Every employer shall be responsible for payment of wages to workers employed by him.
(2) Wages shall be paid before the expiry of the seventh working day following the last day of the wage period.
(3) Where the employment of any worker is terminated by or on behalf of the employer, the wages earned by him shall be paid before the expiry of the seventh working day from the day on which his employment is terminated.

Section 7. Deductions from wages
(1) Notwithstanding any contract to the contrary, no deduction shall be made from the wages of a worker except those authorized by or under this Act.
(2) Deductions from the wages of a worker shall be made only on account of:
(a) fines imposed under section 15;
(b) deductions for absence from duty;
(c) deductions for damage to or loss of goods;
(d) deductions for recovery of advances;
(e) deductions for income tax payable by the worker.

Section 8. Minimum wages
(1) The Government may, by notification in the official Gazette, fix the minimum rates of wages for workers employed in any establishment or class of establishments.
(2) The minimum rates of wages may be fixed for:
(a) different classes of workers;
(b) different localities;
(c) different establishments or classes of establishments.
(3) No employer shall pay to any worker wages at a rate less than the minimum rate of wages fixed under this section.

CHAPTER IV
LEAVE AND HOLIDAYS

Section 9. Annual leave with wages
(1) Every worker who has completed twelve months of continuous service in an establishment shall be allowed annual leave with wages for a number of days calculated at the rate of one day for every eighteen days of work.
(2) The leave shall be exclusive of all holidays.
(3) The wages payable to a worker during the period of leave shall be computed at the rate equal to the daily average of his total full time earnings for the days on which he actually worked during the preceding twelve months.

Section 10. Sick leave
(1) Every worker shall be entitled to sick leave with full wages for fourteen days in a calendar year.
(2) No worker shall be entitled to sick leave unless he produces a medical certificate from a registered medical practitioner.

Section 11. Casual leave
(1) Every worker shall be entitled to casual leave with full wages for ten days in a calendar year.
(2) Casual leave shall not be accumulated and shall lapse at the end of the calendar year.

PART III
SAFETY AND HEALTH

CHAPTER V
WORKPLACE SAFETY

Section 12. Cleanliness
(1) Every establishment shall be kept clean and free from effluvia arising from any drain, privy or other nuisance.
(2) The floors of every workroom shall be cleaned at least once every week.

Section 13. Ventilation and temperature
(1) Effective and suitable provision shall be made in every establishment for securing and maintaining adequate ventilation.
(2) Such temperature shall be maintained as to secure to workers therein reasonable conditions of comfort.

Section 14. Safety of buildings and machinery
(1) If it appears to the Inspector that any building or part of a building used as an establishment is in such a condition that it is dangerous to human life or safety, the Inspector may serve a notice on the owner or occupier of the building requiring him to take such measures as may be specified in the notice.
(2) Every dangerous part of any machinery shall be securely fenced.

CHAPTER VI
PENALTIES AND ENFORCEMENT

Section 15. Penalties for violations
(1) Whoever contravenes any provision of this Act or any rule made thereunder shall be punishable with imprisonment for a term which may extend to one year, or with fine which may extend to fifty thousand taka, or with both.
(2) Where a contravention is committed by a company, every person who at the time of the contravention was in charge of and responsible to the company for the conduct of the business shall be deemed to be guilty.

Section 16. Powers of Inspectors
(1) An Inspector may, for the purpose of carrying out the provisions of this Act:
(a) enter any premises which is used as an establishment;
(b) examine any person found in such premises;
(c) require the production of any register, record or document;
(d) take copies of such registers, records or documents;
(e) seize any register, record or document which may be material evidence.

Section 17. Complaints and proceedings
(1) Any worker may present a complaint to the Inspector if the employer fails to comply with any provision of this Act.
(2) The Inspector shall, within thirty days of receipt of the complaint, inquire into the matter and pass orders thereon.
(3) An appeal against the order of the Inspector shall lie to the Labour Court within thirty days.
"""


def create_sample_pdf(output_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Create a sample law PDF for testing the ingestion pipeline.

    Returns:
        Path to the created PDF file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "sample_workers_protection_act_2024.pdf"
    pdf_path = output_dir / filename

    doc = fitz.open()

    # Split text into pages (~3000 chars per page)
    lines = SAMPLE_LAW_TEXT.strip().split("\n")
    current_page_text = ""
    page_num = 0

    for line in lines:
        current_page_text += line + "\n"

        if len(current_page_text) > 2500:
            page = doc.new_page(width=595, height=842)  # A4 size
            page_num += 1

            # Add header
            header_rect = fitz.Rect(50, 20, 545, 40)
            page.insert_textbox(
                header_rect,
                f"Sample Workers Protection Act, 2024 — Page {page_num}",
                fontsize=8,
                color=(0.5, 0.5, 0.5),
            )

            # Add main text
            text_rect = fitz.Rect(50, 50, 545, 790)
            page.insert_textbox(
                text_rect,
                current_page_text,
                fontsize=10,
                fontname="helv",
            )

            # Add footer
            footer_rect = fitz.Rect(50, 800, 545, 820)
            page.insert_textbox(
                footer_rect,
                f"— {page_num} —",
                fontsize=8,
                color=(0.5, 0.5, 0.5),
                align=1,  # Center
            )

            current_page_text = ""

    # Last page
    if current_page_text.strip():
        page = doc.new_page(width=595, height=842)
        page_num += 1

        header_rect = fitz.Rect(50, 20, 545, 40)
        page.insert_textbox(
            header_rect,
            f"Sample Workers Protection Act, 2024 — Page {page_num}",
            fontsize=8,
            color=(0.5, 0.5, 0.5),
        )

        text_rect = fitz.Rect(50, 50, 545, 790)
        page.insert_textbox(
            text_rect,
            current_page_text,
            fontsize=10,
            fontname="helv",
        )

        footer_rect = fitz.Rect(50, 800, 545, 820)
        page.insert_textbox(
            footer_rect,
            f"— {page_num} —",
            fontsize=8,
            color=(0.5, 0.5, 0.5),
            align=1,
        )

    doc.save(str(pdf_path))
    doc.close()

    print(f"✓ Sample PDF created: {pdf_path}")
    print(f"  Pages: {page_num}")
    print(f"  Size: {pdf_path.stat().st_size / 1024:.1f} KB")

    return pdf_path


# Also update the config to include this sample document
SAMPLE_LAW_CONFIG = {
    "id": "sample_workers_2024",
    "name": "Sample Workers Protection Act 2024",
    "filename": "sample_workers_protection_act_2024.pdf",
    "source_url": "N/A (test document)",
    "priority": "TEST",
    "category": "Employment",
    "year": 2024,
    "language": "english",
}


if __name__ == "__main__":
    pdf_path = create_sample_pdf()
    print(f"\nYou can now test the pipeline with:")
    print(f"  python scripts/run_pipeline.py --sample")
