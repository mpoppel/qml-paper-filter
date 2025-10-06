import arxiv
import datetime
from datetime import timezone
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

# Keywords for relevance scoring (from your research interests)
RELEVANCE_KEYWORDS = {
    'high': ['fourier', 'fourier series', 'barren plateau', 'data encoding', 'expressivity',
             'frequency', 'spectral', 'trainability'],
    'medium': ['quantum circuit', 'variational quantum', 'parametrized quantum',
               'quantum machine learning', 'qml', 'quantum neural network'],
    'low': ['quantum computing', 'quantum algorithm', 'qubit']
}

CATEGORIES = ["quant-ph", "cs.LG", "cs.AI"]
MAX_RESULTS_PER_QUERY = 25
DAYS_BACK = 7  # Extended lookback period to catch delayed publications


def calculate_relevance_score(paper):
    """Calculate relevance score for a paper based on keywords in title and abstract."""
    text = (paper.title + " " + paper.summary).lower()
    score = 0
    matched_keywords = []

    # High priority keywords
    for keyword in RELEVANCE_KEYWORDS['high']:
        if keyword in text:
            score += 3
            matched_keywords.append(keyword)

    # Medium priority keywords
    for keyword in RELEVANCE_KEYWORDS['medium']:
        if keyword in text:
            score += 2
            matched_keywords.append(keyword)

    # Low priority keywords
    for keyword in RELEVANCE_KEYWORDS['low']:
        if keyword in text:
            score += 1
            matched_keywords.append(keyword)

    return score, matched_keywords


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
    # Create timezone-aware cutoff date (UTC)
    cutoff_date = datetime.datetime.now(timezone.utc) - datetime.timedelta(days=DAYS_BACK)
    all_papers = []
    seen_ids = set()

    print(f"Searching from {cutoff_date.strftime('%Y-%m-%d')} onwards...")

    # Create arxiv client (new API)
    client = arxiv.Client()

    # Search general queries
    for query in SEARCH_QUERIES:
        print(f"  Query: {query}")

        search = arxiv.Search(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        try:
            for paper in client.results(search):
                # Check both published and updated dates
                paper_date = max(paper.published, paper.updated)
                if paper_date >= cutoff_date and paper.entry_id not in seen_ids:
                    if any(cat in paper.categories for cat in CATEGORIES):
                        all_papers.append(paper)
                        seen_ids.add(paper.entry_id)
        except Exception as e:
            print(f"  Warning: Error searching '{query}': {e}")
            continue

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

        try:
            for paper in client.results(search):
                paper_date = max(paper.published, paper.updated)
                if paper_date >= cutoff_date and paper.entry_id not in seen_ids:
                    if any(cat in paper.categories for cat in CATEGORIES):
                        all_papers.append(paper)
                        seen_ids.add(paper.entry_id)
        except Exception as e:
            print(f"  Warning: Error searching citations for '{ref_name}': {e}")
            continue

    # Sort by relevance score first, then by date
    for paper in all_papers:
        paper.relevance_score, paper.matched_keywords = calculate_relevance_score(paper)

    all_papers.sort(key=lambda x: (x.relevance_score, max(x.published, x.updated)), reverse=True)
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
    """Generate a rich HTML email body with paper summaries."""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    paper_count = len(papers)

    # Plain text version for the body parameter
    plain_text = f"FOUND {paper_count} RELEVANT PAPERS\n"
    plain_text += f"Date: {today}\n"
    plain_text += f"Search period: Last {DAYS_BACK} days\n"

    if not papers:
        plain_text += f"\nNo new papers found in the last {DAYS_BACK} days.\n"

    return plain_text


def generate_email_html(papers):
    """Generate rich HTML email with paper summaries and relevance scores."""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    paper_count = len(papers)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .paper {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .paper-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .paper-title a {{
            color: #667eea;
            text-decoration: none;
        }}
        .paper-title a:hover {{
            text-decoration: underline;
        }}
        .paper-meta {{
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }}
        .paper-meta span {{
            margin-right: 15px;
        }}
        .relevance-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 10px;
        }}
        .relevance-high {{
            background-color: #48bb78;
            color: white;
        }}
        .relevance-medium {{
            background-color: #ed8936;
            color: white;
        }}
        .relevance-low {{
            background-color: #cbd5e0;
            color: #2d3748;
        }}
        .paper-abstract {{
            font-size: 14px;
            color: #4a5568;
            margin: 10px 0;
            line-height: 1.6;
        }}
        .paper-keywords {{
            font-size: 12px;
            color: #667eea;
            margin-top: 10px;
            font-style: italic;
        }}
        .paper-links {{
            margin-top: 15px;
        }}
        .paper-links a {{
            display: inline-block;
            padding: 8px 15px;
            margin-right: 10px;
            border-radius: 5px;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
        }}
        .btn-arxiv {{
            background-color: #667eea;
            color: white;
        }}
        .btn-pdf {{
            background-color: #f56565;
            color: white;
        }}
        .btn-arxiv:hover, .btn-pdf:hover {{
            opacity: 0.9;
        }}
        .summary {{
            background-color: #edf2f7;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #666;
            font-size: 14px;
        }}
        .no-papers {{
            text-align: center;
            padding: 40px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 arXiv Quantum ML Digest</h1>
        <p>{today} • {paper_count} papers found</p>
        <p>Ranked by relevance to your research</p>
    </div>
"""

    if not papers:
        html += """
    <div class="no-papers">
        <h2>No new papers found</h2>
        <p>No papers matching your criteria were published in the last {} days.</p>
    </div>
""".format(DAYS_BACK)
    else:
        html += f"""
    <div class="summary">
        <strong>Search Summary:</strong> Found {paper_count} papers in the last {DAYS_BACK} days, 
        sorted by relevance to quantum circuits, machine learning, and Fourier analysis.
    </div>
"""

        for i, paper in enumerate(papers, 1):
            # Get relevance level
            score = paper.relevance_score
            if score >= 6:
                relevance_class = "relevance-high"
                relevance_text = f"High Relevance (Score: {score})"
            elif score >= 3:
                relevance_class = "relevance-medium"
                relevance_text = f"Medium Relevance (Score: {score})"
            else:
                relevance_class = "relevance-low"
                relevance_text = f"Low Relevance (Score: {score})"

            # Format authors
            authors = ", ".join([author.name for author in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += " et al."

            # Truncate abstract for email
            abstract = paper.summary.replace('\n', ' ').strip()
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."

            # Format matched keywords
            keywords_text = ""
            if paper.matched_keywords:
                keywords_text = "Matched keywords: " + ", ".join(list(set(paper.matched_keywords))[:5])

            html += f"""
    <div class="paper">
        <div class="paper-title">
            <span class="relevance-badge {relevance_class}">{relevance_text}</span>
            <br>
            {i}. <a href="{paper.entry_id}">{paper.title}</a>
        </div>
        <div class="paper-meta">
            <span>👤 {authors}</span>
            <span>📅 Published: {paper.published.strftime('%Y-%m-%d')}</span>
            <span>🏷️ {', '.join(paper.categories[:3])}</span>
        </div>
        <div class="paper-abstract">
            {abstract}
        </div>
        {f'<div class="paper-keywords">{keywords_text}</div>' if keywords_text else ''}
        <div class="paper-links">
            <a href="{paper.entry_id}" class="btn-arxiv">📄 View on arXiv</a>
            <a href="{paper.pdf_url}" class="btn-pdf">📥 Download PDF</a>
        </div>
    </div>
"""

    html += f"""
    <div class="footer">
        <p>This is an automated digest from your arXiv paper monitor.</p>
        <p>View full reports on <a href="https://github.com/YOUR_USERNAME/YOUR_REPO/tree/main/reports">GitHub</a></p>
    </div>
</body>
</html>
"""

    return html


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

    # Generate HTML email
    email_html = generate_email_html(papers)
    html_file = "reports/email_body.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(email_html)

    print(f"Email body saved to: {email_file}")
    print(f"Email HTML saved to: {html_file}")

    # Print summary to console
    if papers:
        print("\nPaper titles:")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper.title}")
            print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")

    return len(papers)


if __name__ == "__main__":
    main()