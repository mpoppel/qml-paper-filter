import arxiv
import datetime
from datetime import timezone
import os
from pathlib import Path

# Configuration
DAYS_BACK = 7
CATEGORIES = ["quant-ph", "cs.LG", "cs.AI", "stat.ML"]
MAX_RESULTS_PER_QUERY = 50  # Fewer queries, more results each

# Consolidated into ~6 broad boolean queries instead of 55 individual ones
SEARCH_QUERIES = [
    # Fourier / spectral / frequency analysis of PQCs
    'ti:"quantum circuit" AND (ti:fourier OR ti:frequency OR ti:spectral OR abs:expressivity)',

    # Barren plateaus, trainability, initialization
    '(ti:"barren plateau" OR ti:"loss landscape" OR ti:"near-zero initialization") AND quantum',

    # DLA, QFIM, overparameterization
    '(ti:"dynamical Lie" OR ti:"Lie algebra" OR ti:"quantum Fisher" OR ti:overparameterization) AND quantum',

    # Data encoding / re-uploading / feature maps
    '(ti:"data re-uploading" OR ti:"data encoding" OR ti:"feature map") AND (quantum OR qubit)',

    # Variational quantum algorithms / QNN architectures broadly
    '(ti:"variational quantum" OR ti:"quantum neural network" OR ti:"parameterized quantum") AND (machine learning OR trainability OR expressivity)',

    # VQE / Hamiltonian learning
    '(ti:"variational quantum eigensolver" OR ti:VQE OR ti:"transverse field Ising") AND (barren OR landscape OR layer)',
]

# Tracked authors — searched individually but capped tightly
TRACKED_AUTHORS = [
    "Maria Schuld",
    "Zoe Holmes",
    "Marco Cerezo",
    "Martin Larocca",
    "Elies Gil-Fuster",
    "Adrian Perez-Salinas",
    "Johannes Jakob Meyer",
    "Frederic Sauvage",
    "Lennart Bittel",
]

# Keywords for relevance scoring
RELEVANCE_KEYWORDS = {
    'high': [
        'fourier', 'fourier series', 'fourier spectrum', 'fourier coefficient',
        'frequency spectrum', 'spectral',
        'barren plateau', 'barren plateaus',
        'dynamical lie algebra', 'dla',
        'quantum fisher information', 'qfim',
        'overparameterization', 'over-parameterization',
        'data re-uploading', 're-uploading',
        'trainable frequency', 'trainable frequencies',
        'near-zero initialization', 'nzi',
        'serial quantum', 'parallel quantum',
        'expressivity', 'expressibility',
        'data encoding',
    ],
    'medium': [
        'parameterized quantum circuit', 'parametrized quantum circuit',
        'variational quantum circuit', 'variational quantum algorithm',
        'quantum neural network', 'qnn',
        'quantum machine learning', 'qml',
        'loss landscape', 'gradient vanishing',
        'quantum kernel', 'feature map',
        'trainability', 'gradient flow',
        'lie algebra', 'lie group',
        'representational power',
    ],
    'low': [
        'variational quantum eigensolver', 'vqe',
        'quantum computing', 'quantum algorithm',
        'qubit', 'quantum gate',
        'generalization', 'learnability',
        'hamiltonian simulation',
    ]
}


def calculate_relevance_score(paper):
    text = (paper.title + " " + paper.summary).lower()
    title_lower = paper.title.lower()
    score = 0
    matched_keywords = []

    for keyword in RELEVANCE_KEYWORDS['high']:
        if keyword in text:
            score += 3
            matched_keywords.append(keyword)
        if keyword in title_lower:
            score += 2  # extra title bonus

    for keyword in RELEVANCE_KEYWORDS['medium']:
        if keyword in text:
            score += 2
            matched_keywords.append(keyword)

    for keyword in RELEVANCE_KEYWORDS['low']:
        if keyword in text:
            score += 1
            matched_keywords.append(keyword)

    paper_author_names = [a.name for a in paper.authors]
    for tracked in TRACKED_AUTHORS:
        if any(tracked.lower() in a.lower() for a in paper_author_names):
            score += 4
            matched_keywords.append(f"author:{tracked}")

    return score, matched_keywords


def format_paper(paper):
    authors = ", ".join([author.name for author in paper.authors[:3]])
    if len(paper.authors) > 3:
        authors += " et al."

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
    cutoff_date = datetime.datetime.now(timezone.utc) - datetime.timedelta(days=DAYS_BACK)
    all_papers = []
    seen_ids = set()

    print(f"Searching from {cutoff_date.strftime('%Y-%m-%d')} onwards...")

    client = arxiv.Client(
        page_size=50,
        delay_seconds=3,   # polite delay between requests
        num_retries=3
    )

    # Consolidated topic queries
    for query in SEARCH_QUERIES:
        print(f"  Query: {query[:80]}...")

        search = arxiv.Search(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY,
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
            print(f"  Warning: Error on query: {e}")
            continue

    # Author searches — kept but capped at 3 results each to stay fast
    print("\nSearching tracked authors...")
    for author in TRACKED_AUTHORS:
        print(f"  Author: {author}")
        search = arxiv.Search(
            query=f'au:"{author}"',
            max_results=3,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        try:
            for paper in client.results(search):
                paper_date = max(paper.published, paper.updated)
                if paper_date >= cutoff_date and paper.entry_id not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(paper.entry_id)
        except Exception as e:
            print(f"  Warning: Error searching author '{author}': {e}")
            continue

    # Score and sort
    for paper in all_papers:
        paper.relevance_score, paper.matched_keywords = calculate_relevance_score(paper)

    all_papers.sort(key=lambda x: (x.relevance_score, max(x.published, x.updated)), reverse=True)
    return all_papers


def generate_markdown_report(papers):
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    report = f"""# arXiv Daily Digest - {today}

**Search Period:** Last {DAYS_BACK} days  
**Papers Found:** {len(papers)}

## Summary

This digest covers:
- Serial vs. parallel QNN architectures (expressivity, trainability)
- Fourier analysis of parameterized quantum circuits
- Dynamical Lie algebra (DLA) and QFIM rank theory
- Barren plateaus, overparameterization, near-zero initialization
- Data re-uploading / trainable frequency feature maps
- VQE and Hamiltonian learning

---

## Papers

"""

    if not papers:
        report += f"*No new papers found matching your criteria in the last {DAYS_BACK} days.*\n"
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

    report += f"\n**Tracked Authors:** {', '.join(TRACKED_AUTHORS)}\n"
    report += f"\n**Categories:** {', '.join(CATEGORIES)}\n"
    report += f"**Lookback Period:** {DAYS_BACK} days\n"

    return report


def save_report(report, output_dir="reports"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    filename = f"{output_dir}/arxiv_digest_{today}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {filename}")
    return filename


def generate_email_body(papers):
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    plain_text = f"FOUND {len(papers)} RELEVANT PAPERS\n"
    plain_text += f"Date: {today}\n"
    plain_text += f"Search period: Last {DAYS_BACK} days\n"
    if not papers:
        plain_text += f"\nNo new papers found in the last {DAYS_BACK} days.\n"
    return plain_text


def generate_email_html(papers):
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    paper_count = len(papers)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6; color: #333; max-width: 800px;
            margin: 0 auto; padding: 20px; background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 30px; border-radius: 10px;
            margin-bottom: 30px; text-align: center;
        }}
        .header h1 {{ margin: 0; font-size: 28px; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .paper {{
            background: white; border-radius: 8px; padding: 20px;
            margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .paper-title {{ font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .paper-title a {{ color: #667eea; text-decoration: none; }}
        .paper-meta {{ font-size: 14px; color: #666; margin-bottom: 10px; }}
        .paper-meta span {{ margin-right: 15px; }}
        .relevance-badge {{
            display: inline-block; padding: 3px 10px; border-radius: 12px;
            font-size: 12px; font-weight: bold; margin-right: 10px;
        }}
        .relevance-high  {{ background-color: #48bb78; color: white; }}
        .relevance-medium {{ background-color: #ed8936; color: white; }}
        .relevance-low   {{ background-color: #cbd5e0; color: #2d3748; }}
        .paper-abstract {{ font-size: 14px; color: #4a5568; margin: 10px 0; line-height: 1.6; }}
        .paper-keywords {{ font-size: 12px; color: #667eea; margin-top: 10px; font-style: italic; }}
        .paper-links {{ margin-top: 15px; }}
        .paper-links a {{
            display: inline-block; padding: 8px 15px; margin-right: 10px;
            border-radius: 5px; text-decoration: none; font-size: 14px; font-weight: 500;
        }}
        .btn-arxiv {{ background-color: #667eea; color: white; }}
        .btn-pdf   {{ background-color: #f56565; color: white; }}
        .summary {{ background-color: #edf2f7; padding: 15px; border-radius: 8px; margin-bottom: 30px; }}
        .footer {{ text-align: center; margin-top: 30px; padding: 20px; color: #666; font-size: 14px; }}
        .no-papers {{ text-align: center; padding: 40px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 arXiv QML Digest</h1>
        <p>{today} • {paper_count} papers found</p>
        <p>Ranked by relevance · Serial/Parallel QNNs · DLA · QFIM · Fourier</p>
    </div>
"""

    if not papers:
        html += f"""
    <div class="no-papers">
        <h2>No new papers found</h2>
        <p>No papers matching your criteria were published in the last {DAYS_BACK} days.</p>
    </div>
"""
    else:
        html += f"""
    <div class="summary">
        <strong>Search Summary:</strong> Found {paper_count} papers in the last {DAYS_BACK} days,
        sorted by relevance to your research on serial/parallel QNNs, DLA, QFIM, Fourier analysis,
        barren plateaus, and trainable-frequency feature maps.
    </div>
"""
        for i, paper in enumerate(papers, 1):
            score = paper.relevance_score
            if score >= 8:
                relevance_class, relevance_text = "relevance-high", f"High Relevance (Score: {score})"
            elif score >= 4:
                relevance_class, relevance_text = "relevance-medium", f"Medium Relevance (Score: {score})"
            else:
                relevance_class, relevance_text = "relevance-low", f"Low Relevance (Score: {score})"

            authors = ", ".join([a.name for a in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += " et al."

            abstract = paper.summary.replace('\n', ' ').strip()
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."

            keywords_text = ""
            if paper.matched_keywords:
                keywords_text = "Matched: " + ", ".join(list(dict.fromkeys(paper.matched_keywords))[:6])

            html += f"""
    <div class="paper">
        <div class="paper-title">
            <span class="relevance-badge {relevance_class}">{relevance_text}</span><br>
            {i}. <a href="{paper.entry_id}">{paper.title}</a>
        </div>
        <div class="paper-meta">
            <span>👤 {authors}</span>
            <span>📅 {paper.published.strftime('%Y-%m-%d')}</span>
            <span>🏷️ {', '.join(paper.categories[:3])}</span>
        </div>
        <div class="paper-abstract">{abstract}</div>
        {f'<div class="paper-keywords">{keywords_text}</div>' if keywords_text else ''}
        <div class="paper-links">
            <a href="{paper.entry_id}" class="btn-arxiv">📄 View on arXiv</a>
            <a href="{paper.pdf_url}" class="btn-pdf">📥 Download PDF</a>
        </div>
    </div>
"""

    html += """
    <div class="footer">
        <p>Automated digest · arXiv QML monitor</p>
        <p>Focus: Serial/Parallel QNNs · DLA · QFIM · Fourier · Barren Plateaus · NZI</p>
    </div>
</body>
</html>
"""
    return html


def main():
    print("=" * 60)
    print("arXiv QML Monitor — Serial/Parallel QNNs, DLA, QFIM")
    print("=" * 60)
    print(f"Search date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Looking back {DAYS_BACK} day(s)")
    print("-" * 60)

    Path("reports").mkdir(parents=True, exist_ok=True)

    papers = search_arxiv_papers()
    print(f"\n{'=' * 60}")
    print(f"FOUND {len(papers)} RELEVANT PAPERS")
    print("=" * 60)

    report = generate_markdown_report(papers)
    save_report(report)

    with open("reports/email_body.txt", 'w', encoding='utf-8') as f:
        f.write(generate_email_body(papers))

    with open("reports/email_body.html", 'w', encoding='utf-8') as f:
        f.write(generate_email_html(papers))

    print("Email body saved to: reports/email_body.txt")
    print("Email HTML saved to: reports/email_body.html")

    if papers:
        print("\nTop 10 papers by relevance:")
        for i, paper in enumerate(papers[:10], 1):
            print(f"{i:2}. [Score {paper.relevance_score:2d}] {paper.title}")
            print(f"     {paper.published.strftime('%Y-%m-%d')} | {', '.join(paper.matched_keywords[:3])}")

    return len(papers)


if __name__ == "__main__":
    main()
