import arxiv
import datetime
from datetime import timezone
import os
from pathlib import Path

# Configuration
SEARCH_QUERIES = [
    # Core architecture focus: serial vs parallel QNNs
    "serial parallel quantum neural network architecture",
    "quantum neural network expressivity trainability",

    # Fourier / spectral theory of PQCs
    "quantum circuits Fourier spectrum data encoding",
    "parameterized quantum circuits frequency analysis",
    "quantum machine learning Fourier coefficients",

    # DLA / QFIM / overparameterization
    "dynamical Lie algebra quantum circuits barren plateaus",
    "quantum Fisher information matrix variational quantum",
    "overparameterization quantum neural networks",
    "QFIM rank parameterized quantum circuits",

    # Barren plateaus — trainability & initialization
    "barren plateaus variational quantum circuits",
    "near-zero initialization quantum circuits trainability",
    "loss landscape quantum neural networks",

    # Data encoding / feature maps / re-uploading
    "data re-uploading quantum classifier",
    "quantum feature map encoding expressivity",
    "trainable frequency quantum machine learning",

    # Generalization & learning theory
    "generalization bounds quantum machine learning",
    "quantum kernel methods learning theory",

    # VQE & Hamiltonian simulation (for your TLFIM work)
    "variational quantum eigensolver barren plateaus",
    "transverse field Ising model variational quantum",
]

# Key papers to find citations for (your active bibliography)
KEY_REFERENCE_PAPERS = {
    # Foundational papers you build on
    "Schuld 2021 data encoding":        "2008.08605",
    "Pérez-Salinas 2020 re-uploading":  "data re-uploading universal quantum classifier",
    "Mitarai 2018 circuit learning":    "quantum circuit learning",
    "McClean 2018 barren plateaus":     "barren plateaus quantum neural network landscapes",

    # Theory papers central to your framework
    "Cerezo 2021 VQA review":           "variational quantum algorithms review",
    "Larocca 2024 BP review":           "barren plateaus variational quantum computing review",
    "Ragone 2024 Lie algebraic BP":     "Lie algebraic theory barren plateaus",
    "Abbas 2021 QNN power":             "power of quantum neural networks",
    "Caro 2022 generalization":         "generalization quantum machine learning few training data",

    # Papers directly related to your current work
    "Barthe 2024 gradients frequencies":"gradients frequency profiles quantum re-uploading",
    "Li 2025 serial gradient":          "2603.18479",   # serial gradient suppression theorem
    "Hashimoto 2025 VQE":               "2602.03291",   # VQE paper you replicated

    # DLA / Lie algebra expressivity
    "Larocca 2022 DLA expressivity":    "diagnosing barren plateaus Lie algebra",
    "Wiersema 2023 DLA":                "classification quantum neural network Lie algebras",
    "Fontana 2023 DLA adjoint":         "classical simulations variational quantum circuits Lie",

    # QFIM / overparameterization
    "Meyer 2021 QFIM":                  "2103.05523",   # QFIM expressibility
    "Haug 2021 scalable QFIM":          "scalable quantum Fisher information",
    "Kiani 2022 overparameterization":  "dimensions overparameterization variational quantum",

    # Encoding strategies (ternary / Golomb)
    "Melo 2023 re-uploading strategies": "re-uploading strategies multivariate quantum",
    "Schreiber 2023 classical surrogates": "classical surrogates parameterized quantum circuits Fourier",
}

# Author-based tracking (key researchers in your space)
TRACKED_AUTHORS = [
    "Maria Schuld",
    "Nathan Killoran",
    "Zoë Holmes",
    "Zoe Holmes",
    "Marco Cerezo",
    "Patrick Coles",
    "Martin Larocca",
    "Frederic Sauvage",
    "Elies Gil-Fuster",
    "Adrián Pérez-Salinas",
    "Adrian Perez-Salinas",
    "Johannes Jakob Meyer",
    "Michael Hartmann",
    "Michael Kölle",     # TUM / Aqarios colleague proximity
    "Tobias Rosskopf",
    "Lennart Bittel",
    "Mikael Svensson",
    "Qiaochu Zhang",     # Zhang et al. NZI theorem
]

# Keywords for relevance scoring — tuned to your research hierarchy
RELEVANCE_KEYWORDS = {
    'high': [
        # Your core theoretical objects
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
        'quantum circuit learning',
        'trainability', 'gradient flow',
        'lie algebra', 'lie group',
        'representational power',
        'circuit ansatz', 'hardware efficient',
    ],
    'low': [
        'variational quantum eigensolver', 'vqe',
        'quantum computing', 'quantum algorithm',
        'qubit', 'quantum gate',
        'pennylane', 'qiskit',
        'quantum advantage',
        'generalization', 'learnability',
        'hamiltonian simulation',
    ]
}

CATEGORIES = ["quant-ph", "cs.LG", "cs.AI", "stat.ML"]
MAX_RESULTS_PER_QUERY = 25
DAYS_BACK = 7


def calculate_relevance_score(paper):
    """Calculate relevance score for a paper based on keywords in title and abstract."""
    text = (paper.title + " " + paper.summary).lower()
    score = 0
    matched_keywords = []

    for keyword in RELEVANCE_KEYWORDS['high']:
        if keyword in text:
            score += 3
            matched_keywords.append(keyword)

    for keyword in RELEVANCE_KEYWORDS['medium']:
        if keyword in text:
            score += 2
            matched_keywords.append(keyword)

    for keyword in RELEVANCE_KEYWORDS['low']:
        if keyword in text:
            score += 1
            matched_keywords.append(keyword)

    # Bonus: title-only match (stronger signal)
    title_lower = paper.title.lower()
    for keyword in RELEVANCE_KEYWORDS['high']:
        if keyword in title_lower:
            score += 2  # extra weight for title hits

    # Bonus: tracked author match
    paper_author_names = [a.name for a in paper.authors]
    for tracked in TRACKED_AUTHORS:
        if any(tracked.lower() in a.lower() for a in paper_author_names):
            score += 4
            matched_keywords.append(f"author:{tracked}")

    return score, matched_keywords


def format_paper(paper):
    """Format a single paper's information."""
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
    """Search arXiv for recent papers matching our criteria."""
    cutoff_date = datetime.datetime.now(timezone.utc) - datetime.timedelta(days=DAYS_BACK)
    all_papers = []
    seen_ids = set()

    print(f"Searching from {cutoff_date.strftime('%Y-%m-%d')} onwards...")

    client = arxiv.Client()

    # General topic queries
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
                paper_date = max(paper.published, paper.updated)
                if paper_date >= cutoff_date and paper.entry_id not in seen_ids:
                    if any(cat in paper.categories for cat in CATEGORIES):
                        all_papers.append(paper)
                        seen_ids.add(paper.entry_id)
        except Exception as e:
            print(f"  Warning: Error searching '{query}': {e}")
            continue

    # Citation proximity searches — papers citing your key references
    print("\nSearching for papers citing key references...")
    for ref_name, ref_query in KEY_REFERENCE_PAPERS.items():
        print(f"  Looking for citations to: {ref_name}")

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

    # Author-based searches — new papers from tracked researchers
    print("\nSearching for new papers from tracked authors...")
    for author in TRACKED_AUTHORS:
        print(f"  Author: {author}")
        search = arxiv.Search(
            query=f"au:{author}",
            max_results=5,
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
    """Generate a markdown report of found papers."""
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

    report += f"\n**Tracked Authors:** {', '.join(TRACKED_AUTHORS)}\n"
    report += f"\n**Categories:** {', '.join(CATEGORIES)}\n"
    report += f"**Lookback Period:** {DAYS_BACK} days\n"

    return report


def save_report(report, output_dir="reports"):
    """Save the report to a markdown file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    filename = f"{output_dir}/arxiv_digest_{today}.md"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {filename}")
    return filename


def generate_email_body(papers):
    """Generate plain-text email body."""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    paper_count = len(papers)

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
        .header h1 {{ margin: 0; font-size: 28px; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .paper {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .paper-title {{ font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .paper-title a {{ color: #667eea; text-decoration: none; }}
        .paper-title a:hover {{ text-decoration: underline; }}
        .paper-meta {{ font-size: 14px; color: #666; margin-bottom: 10px; }}
        .paper-meta span {{ margin-right: 15px; }}
        .relevance-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 10px;
        }}
        .relevance-high  {{ background-color: #48bb78; color: white; }}
        .relevance-medium {{ background-color: #ed8936; color: white; }}
        .relevance-low   {{ background-color: #cbd5e0; color: #2d3748; }}
        .paper-abstract {{ font-size: 14px; color: #4a5568; margin: 10px 0; line-height: 1.6; }}
        .paper-keywords {{ font-size: 12px; color: #667eea; margin-top: 10px; font-style: italic; }}
        .paper-links {{ margin-top: 15px; }}
        .paper-links a {{
            display: inline-block;
            padding: 8px 15px;
            margin-right: 10px;
            border-radius: 5px;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
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
                relevance_class = "relevance-high"
                relevance_text = f"High Relevance (Score: {score})"
            elif score >= 4:
                relevance_class = "relevance-medium"
                relevance_text = f"Medium Relevance (Score: {score})"
            else:
                relevance_class = "relevance-low"
                relevance_text = f"Low Relevance (Score: {score})"

            authors = ", ".join([author.name for author in paper.authors[:3]])
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

    html += f"""
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
    print("arXiv QML Paper Monitor — Serial/Parallel QNNs, DLA, QFIM")
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
    filename = save_report(report)

    email_body = generate_email_body(papers)
    with open("reports/email_body.txt", 'w', encoding='utf-8') as f:
        f.write(email_body)

    email_html = generate_email_html(papers)
    with open("reports/email_body.html", 'w', encoding='utf-8') as f:
        f.write(email_html)

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
