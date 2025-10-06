import arxiv
import datetime
import os
from pathlib import Path

# Configuration
SEARCH_QUERIES = [
    "quantum circuits AND machine learning",
    "quantum machine learning AND Fourier",
    "variational quantum circuits",
    "parametrized quantum circuits",
    "quantum neural networks",
    "barren plateaus quantum",
    "quantum data encoding",
    "quantum Fourier analysis"
]

# Key papers to find citations for (from your bibliography)
KEY_REFERENCE_PAPERS = {
    "Schuld 2021": "2008.08605",  # Effect of data encoding
    "McClean 2018": "barren plateaus quantum neural network",
    "Pérez-Salinas 2020": "data re-uploading universal quantum classifier",
    "Larocca 2024": "barren plateaus variational quantum computing review",
    "Abbas 2021": "power of quantum neural networks",
    "Cerezo 2021": "variational quantum algorithms",
    "Mitarai 2018": "quantum circuit learning",
    "Caro 2022": "generalization quantum machine learning few training data",
    "Ragone 2024": "Lie algebraic theory barren plateaus",
    "Barthe 2024": "gradients frequency profiles quantum re-uploading"
}

CATEGORIES = ["quant-ph", "cs.LG", "cs.AI"]
MAX_RESULTS_PER_QUERY = 25
DAYS_BACK = 7  # Extended lookback period to catch delayed publications


def format_paper(paper):
    """Format a single paper's information."""
    authors = ", ".join([author.name for author in paper.authors[:3]])
    if len(paper.authors) > 3:
        authors += " et al."

    # Truncate abstract if too long
    abstract = paper.summary.replace('\n', ' ')
    if len(abstract) > 400:
        abstract = abstract[:400] + "..."

    return f"""
### [{paper.title}]({paper.entry_id})
**Authors:** {authors}  
**Published:** {paper.published.strftime('%Y-%m-%d')}  
**Updated:** {paper.updated.strftime('%Y-%m-%d')}  
**Categories:** {', '.join(paper.categories)}  

**Abstract:** {abstract}

[View on arXiv]({paper.entry_id}) | [PDF]({paper.pdf_url})

---
"""


def search_arxiv_papers():
    """Search arXiv for recent papers matching our criteria."""
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=DAYS_BACK)
    all_papers = []
    seen_ids = set()

    print(f"Searching from {cutoff_date.strftime('%Y-%m-%d')} onwards...")

    # Search general queries
    for query in SEARCH_QUERIES:
        print(f"  Query: {query}")

        search = arxiv.Search(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        for paper in search.results():
            # Check both published and updated dates
            paper_date = max(paper.published, paper.updated)
            if paper_date >= cutoff_date and paper.entry_id not in seen_ids:
                if any(cat in paper.categories for cat in CATEGORIES):
                    all_papers.append(paper)
                    seen_ids.add(paper.entry_id)

    # Search for papers citing key references
    print("\nSearching for papers citing key references...")
    for ref_name, ref_query in KEY_REFERENCE_PAPERS.items():
        print(f"  Looking for citations to: {ref_name}")

        # Search for papers that might cite this work
        search = arxiv.Search(
            query=ref_query,
            max_results=15,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        for paper in search.results():
            paper_date = max(paper.published, paper.updated)
            if paper_date >= cutoff_date and paper.entry_id not in seen_ids:
                if any(cat in paper.categories for cat in CATEGORIES):
                    all_papers.append(paper)
                    seen_ids.add(paper.entry_id)

    # Sort by most recent first
    all_papers.sort(key=lambda x: max(x.published, x.updated), reverse=True)
    return all_papers


def generate_markdown_report(papers):
    """Generate a markdown report of found papers."""
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    report = f"""# arXiv Daily Digest - {today}

**Search Period:** Last {DAYS_BACK} days  
**Papers Found:** {len(papers)}

## Summary

This digest includes papers on:
- Quantum circuits for machine learning
- Fourier analysis of quantum models
- Variational quantum algorithms
- Barren plateaus and trainability
- Data encoding strategies

---

## Papers

"""

    if not papers:
        report += "*No new papers found matching your criteria in the last {} days.*\n".format(DAYS_BACK)
    else:
        for paper in papers:
            report += format_paper(paper)

    report += f"""
---

## Search Configuration

**Queries:**
"""
    for query in SEARCH_QUERIES:
        report += f"- {query}\n"

    report += f"\n**Categories:** {', '.join(CATEGORIES)}\n"
    report += f"**Lookback Period:** {DAYS_BACK} days\n"

    return report


def save_report(report, output_dir="reports"):
    """Save the report to a markdown file."""
    # Ensure directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    filename = f"{output_dir}/arxiv_digest_{today}.md"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {filename}")
    return filename


def generate_email_body(papers):
    """Generate a concise email body."""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    paper_count = len(papers)

    # Start with paper count for easy parsing
    email = f"FOUND {paper_count} RELEVANT PAPERS\n"
    email += f"Date: {today}\n"
    email += f"Search period: Last {DAYS_BACK} days\n\n"

    if not papers:
        email += f"No new papers found in the last {DAYS_BACK} days.\n"
        return email

    email += "=" * 60 + "\n"
    email += "NEW PAPERS:\n"
    email += "=" * 60 + "\n\n"

    for i, paper in enumerate(papers[:10], 1):  # Limit to top 10 for email
        email += f"{i}. {paper.title}\n"
        email += f"   {paper.entry_id}\n"
        email += f"   Published: {paper.published.strftime('%Y-%m-%d')}\n\n"

    if len(papers) > 10:
        email += f"\n... and {len(papers) - 10} more papers.\n"

    email += f"\nView full report on GitHub: https://github.com/YOUR_USERNAME/YOUR_REPO/blob/main/reports/arxiv_digest_{today}.md"

    return email


def main():
    print("=" * 60)
    print("arXiv Quantum ML Paper Monitor")
    print("=" * 60)
    print(f"Search date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Looking back {DAYS_BACK} day(s)")
    print("-" * 60)

    # Ensure reports directory exists
    Path("reports").mkdir(parents=True, exist_ok=True)

    papers = search_arxiv_papers()
    print(f"\n{'=' * 60}")
    print(f"FOUND {len(papers)} RELEVANT PAPERS")
    print("=" * 60)

    report = generate_markdown_report(papers)
    filename = save_report(report)

    # Generate email body for GitHub Actions
    email_body = generate_email_body(papers)
    email_file = "reports/email_body.txt"
    with open(email_file, 'w', encoding='utf-8') as f:
        f.write(email_body)

    print(f"Email body saved to: {email_file}")

    # Print summary to console
    if papers:
        print("\nPaper titles:")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper.title}")
            print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")

    return len(papers)


if __name__ == "__main__":
    main()