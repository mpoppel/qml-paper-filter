"""
Quantum ML Paper Filter for GitHub Actions
Enhanced version with proper output handling and GitHub integration
"""

import argparse
import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import requests
import xml.etree.ElementTree as ET
import time

# Import our configuration and reference checker
from config import QMLConfig
from reference_checker import ReferenceChecker


class GitHubQMLFilter:

    def __init__(self, config: QMLConfig):
        self.config = config
        self.reference_checker = ReferenceChecker()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def fetch_recent_papers(self, days_back: int = 1) -> List[Dict]:
        """Fetch recent papers from arXiv with progress tracking"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        print(f"📅 Fetching papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        papers = []
        categories = ['quant-ph', 'cs.LG', 'cs.AI', 'stat.ML', 'physics.comp-ph']

        for i, category in enumerate(categories):
            print(f"🔍 Searching category {category} ({i + 1}/{len(categories)})")

            query = f"cat:{category}"
            base_url = "http://export.arxiv.org/api/query"

            params = {
                'search_query': query,
                'start': 0,
                'max_results': min(200, 1000 // len(categories)),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            try:
                response = requests.get(base_url, params=params, timeout=30)
                category_papers = self._parse_arxiv_response(response.text, start_date)
                papers.extend(category_papers)
                print(f"   Found {len(category_papers)} recent papers in {category}")
                time.sleep(1)
            except Exception as e:
                print(f"❌ Error fetching {category}: {e}")

        # ✅ Deduplicate by arXiv ID
        unique_papers = {}

        for paper in papers:
            unique_papers[paper['id']] = paper  # overwrite duplicates safely

        #print(f"📊 Total papers fetched (before dedup): {len(papers)}")
        #print(f"📊 Unique papers after deduplication: {len(unique_papers)}")

        return list(unique_papers.values())

    def _parse_arxiv_response(self, xml_content: str, start_date: datetime) -> List[Dict]:
        """Parse arXiv API XML response with error handling"""
        try:
            root = ET.fromstring(xml_content)
            papers = []

            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                try:
                    paper = {}

                    # Basic info
                    paper['id'] = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                    paper['title'] = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                    paper['summary'] = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                    paper['url'] = f"https://arxiv.org/abs/{paper['id']}"
                    paper['pdf_url'] = f"https://arxiv.org/pdf/{paper['id']}.pdf"

                    # Authors
                    authors = []
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                        name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                        if name_elem is not None:
                            authors.append(name_elem.text)
                    paper['authors'] = authors

                    # Date
                    published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                    if published_elem is not None:
                        published = published_elem.text
                        paper['published'] = datetime.fromisoformat(published.replace('Z', '+00:00'))

                        # Only include recent papers
                        if paper['published'].date() >= start_date.date():
                            papers.append(paper)

                except Exception as e:
                    print(f"⚠️  Error parsing individual paper: {e}")
                    continue

        except Exception as e:
            print(f"❌ Error parsing XML response: {e}")
            return []

        return papers

    def score_paper_relevance(self, paper: Dict) -> Dict:
        """Enhanced scoring with detailed breakdown"""
        score = 0
        reasons = []

        title = paper['title'].lower()
        abstract = paper['summary'].lower()
        full_text = f"{title} {abstract}"
        authors = [author.lower() for author in paper['authors']]

        # Keyword scoring with detailed tracking
        for category, keywords in self.config.qml_keywords.items():
            category_score = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword.lower() in full_text:
                    weight = self.config.keyword_weights.get(category, 5)
                    category_score += weight
                    matched_keywords.append(keyword)

            if category_score > 0:
                score += category_score
                reasons.append(f"{category.title()}: {', '.join(matched_keywords)} (+{category_score})")

        # Author scoring
        matched_authors = []
        for key_author in self.config.key_authors:
            if any(key_author.lower() in author for author in authors):
                score += 15
                matched_authors.append(key_author)

        if matched_authors:
            reasons.append(f"Key authors: {', '.join(matched_authors)} (+{len(matched_authors) * 15})")

        # Category bonus (if available)
        if hasattr(paper, 'categories'):
            relevant_cats = [cat for cat in paper.get('categories', []) if cat in ['quant-ph', 'cs.LG']]
            if relevant_cats:
                score += 5
                reasons.append(f"Categories: {', '.join(relevant_cats)} (+5)")

        return {
            'score': score,
            'reasons': reasons,
            'confidence': 'high' if score >= 20 else 'medium' if score >= 10 else 'low'
        }

    def filter_and_enhance_papers(self, papers: List[Dict], min_score: int = 5) -> List[Dict]:
        """Filter papers and add reference analysis"""
        print(f"🎯 Filtering {len(papers)} papers (minimum score: {min_score})")

        relevant_papers = []

        for i, paper in enumerate(papers):
            if i > 0 and i % 10 == 0:
                print(f"   Processed {i}/{len(papers)} papers")

            # Basic relevance scoring
            relevance = self.score_paper_relevance(paper)
            paper.update(relevance)

            # Only do expensive reference checking for promising papers
            if relevance['score'] >= min_score or any('quantum' in reason.lower() for reason in relevance['reasons']):
                try:
                    # Reference analysis with timeout
                    ref_analysis = self.reference_checker.check_paper_references(
                        paper['id'],
                        paper_data=paper,
                        timeout=30
                    )
                    paper['reference_analysis'] = ref_analysis
                    paper['reference_score'] = ref_analysis.get('total_score', 0)
                except Exception as e:
                    print(f"⚠️  Reference check failed for {paper['id']}: {e}")
                    paper['reference_analysis'] = {'error': str(e), 'total_score': 0}
                    paper['reference_score'] = 0
            else:
                paper['reference_analysis'] = {'skipped': True, 'total_score': 0}
                paper['reference_score'] = 0

            # Combined scoring
            paper['combined_score'] = paper['score'] + (paper['reference_score'] * 0.7)

            # Include if meets threshold
            if paper['combined_score'] >= min_score:
                relevant_papers.append(paper)

        # Sort by combined score
        relevant_papers.sort(key=lambda x: x['combined_score'], reverse=True)

        print(f"✅ Found {len(relevant_papers)} relevant papers")
        return relevant_papers

    def generate_text_digest(self, papers: List[Dict], date_str: str) -> str:
        """Generate plain text digest"""
        if not papers:
            return f"No relevant quantum ML papers found for {date_str}."

        digest_lines = [
            f"🔬 Quantum ML Daily Digest - {date_str}",
            f"Found {len(papers)} relevant papers",
            "=" * 60,
            ""
        ]

        for i, paper in enumerate(papers[:15], 1):  # Top 15 papers
            digest_lines.extend([
                f"{i}. {paper['title']}",
                f"   arXiv:{paper['id']} | Combined Score: {paper['combined_score']:.1f}",
                f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}",
                f"   Relevance: {'; '.join(paper['reasons'][:2])}",
                f"   References: {paper.get('reference_score', 0)} points",
                f"   URL: {paper['url']}",
                f"   Abstract: {paper['summary'][:200]}...",
                ""
            ])

        return "\n".join(digest_lines)

    def generate_html_digest(self, papers: List[Dict], date_str: str) -> str:
            """Generate HTML email digest - Fixed version"""

            # Simplified HTML template to avoid quote issues
            html_template = '''<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Quantum ML Daily Digest</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5; 
            }}
            .container {{ 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 8px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }}
            .header {{ 
                text-align: center; 
                color: #2c3e50; 
                margin-bottom: 30px; 
            }}
            .summary {{ 
                background: #e8f4fd; 
                padding: 15px; 
                border-radius: 5px; 
                margin-bottom: 20px; 
            }}
            .paper {{ 
                border-left: 4px solid #3498db; 
                padding: 15px; 
                margin-bottom: 20px; 
                background: #fafafa; 
            }}
            .paper-title {{ 
                font-size: 18px; 
                font-weight: 600; 
                color: #2c3e50; 
                margin-bottom: 8px; 
            }}
            .paper-meta {{ 
                color: #7f8c8d; 
                font-size: 14px; 
                margin-bottom: 10px; 
            }}
            .paper-score {{ 
                background: #27ae60; 
                color: white; 
                padding: 2px 8px; 
                border-radius: 12px; 
                font-size: 12px; 
            }}
            .paper-reasons {{ 
                color: #27ae60; 
                font-size: 13px; 
                margin-bottom: 8px; 
            }}
            .paper-abstract {{ 
                color: #34495e; 
                line-height: 1.4; 
                margin-bottom: 10px; 
            }}
            .paper-links a {{ 
                color: #3498db; 
                text-decoration: none; 
                margin-right: 15px; 
            }}
            .paper-links a:hover {{ 
                text-decoration: underline; 
            }}
            .footer {{ 
                text-align: center; 
                color: #7f8c8d; 
                font-size: 12px; 
                margin-top: 30px; 
                padding-top: 20px; 
                border-top: 1px solid #ecf0f1; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 Quantum ML Daily Digest</h1>
                <h2>{date_str}</h2>
            </div>

            <div class="summary">
                <strong>📊 Summary:</strong> Found {num_papers} relevant papers from arXiv
            </div>

            {papers_html}

            <div class="footer">
                <p>Generated by Quantum ML Paper Filter</p>
            </div>
        </div>
    </body>
    </html>'''

            papers_html = ""
            for i, paper in enumerate(papers[:10], 1):  # Top 10 for email
                # Clean up text to avoid HTML issues
                title = paper['title'].replace('<', '&lt;').replace('>', '&gt;')
                abstract = paper['summary'][:250].replace('<', '&lt;').replace('>', '&gt;')

                reasons_text = "; ".join(paper['reasons'][:3])
                reasons_text = reasons_text.replace('<', '&lt;').replace('>', '&gt;')

                ref_info = ""
                if paper.get('reference_score', 0) > 0:
                    ref_info = f" | References: {paper['reference_score']} pts"

                author_text = paper['authors'][0] if paper['authors'] else 'Unknown'
                if len(paper['authors']) > 1:
                    author_text += ' et al.'
                author_text = author_text.replace('<', '&lt;').replace('>', '&gt;')

                papers_html += f'''
                <div class="paper">
                    <div class="paper-title">{i}. {title}</div>
                    <div class="paper-meta">
                        <span class="paper-score">{paper['combined_score']:.1f} points</span>
                        {author_text} • arXiv:{paper['id']}{ref_info}
                    </div>
                    <div class="paper-reasons">🎯 {reasons_text}</div>
                    <div class="paper-abstract">{abstract}...</div>
                    <div class="paper-links">
                        <a href="{paper['url']}" target="_blank">📄 View Paper</a>
                        <a href="{paper['pdf_url']}" target="_blank">📥 Download PDF</a>
                    </div>
                </div>
                '''

            try:
                return html_template.format(
                    date_str=date_str,
                    num_papers=len(papers),
                    papers_html=papers_html
                )
            except Exception as e:
                # Fallback to simple text if HTML generation fails
                print(f"⚠️ HTML generation failed: {e}, falling back to simple HTML")
                return f'''<!DOCTYPE html>
    <html><head><title>Quantum ML Digest</title></head>
    <body>
    <h1>🔬 Quantum ML Daily Digest - {date_str}</h1>
    <p>Found {len(papers)} relevant papers</p>
    <p>HTML generation encountered an error. Check the text digest for full results.</p>
    </body></html>'''

    def save_outputs(self, papers: List[Dict], date_str: str) -> Dict[str, str]:
        """Save all output formats - Fixed version"""
        outputs = {}

        # Text digest
        text_digest = self.generate_text_digest(papers, date_str)
        text_file = self.output_dir / f"digest_{date_str}.txt"
        text_file.write_text(text_digest, encoding='utf-8')
        outputs['text_file'] = str(text_file)

        # HTML email
        html_digest = self.generate_html_digest(papers, date_str)
        html_file = self.output_dir / "digest_email.html"
        html_file.write_text(html_digest, encoding='utf-8')
        outputs['html_file'] = str(html_file)

        # JSON data
        json_file = self.output_dir / f"papers_data_{date_str}.json"
        json_file.write_text(json.dumps(papers, indent=2, default=str), encoding='utf-8')
        outputs['json_file'] = str(json_file)

        # GitHub Actions outputs (with better error handling)
        if os.getenv('GITHUB_ACTIONS'):
            try:
                output_file = os.environ.get('GITHUB_OUTPUT')
                if output_file and os.path.exists(os.path.dirname(output_file)):
                    with open(output_file, 'a') as f:
                        f.write(f"relevant_papers={len(papers)}\n")
                        if papers:
                            # Sanitize title for GitHub Actions (remove newlines, limit length)
                            clean_title = papers[0]['title'].replace('\n', ' ').replace('\r', ' ')[:100]
                            clean_title = ''.join(c for c in clean_title if c.isprintable() and c != '"')

                            f.write(f"top_paper={clean_title}\n")
                            f.write(f"top_score={papers[0]['combined_score']:.1f}\n")
                else:
                    print("⚠️  GITHUB_OUTPUT file not accessible, skipping GitHub Actions outputs")
            except Exception as e:
                print(f"⚠️  GitHub Actions output failed: {e}")
                # Don't fail the entire process for output issues

        return outputs
    def run(self, days_back: int = 1, min_score: int = 5) -> Dict:
        """Main execution function"""
        start_time = time.time()
        date_str = datetime.now().strftime("%Y-%m-%d")

        print(f"🚀 Starting Quantum ML Paper Filter for {date_str}")

        try:
            # Fetch papers
            papers = self.fetch_recent_papers(days_back)

            if not papers:
                print("⚠️  No papers found in date range")
                return {'success': False, 'message': 'No papers found'}

            # Filter and enhance
            relevant_papers = self.filter_and_enhance_papers(papers, min_score)

            # Generate outputs
            outputs = self.save_outputs(relevant_papers, date_str)

            # Summary
            elapsed = time.time() - start_time
            print(f"✅ Filter completed in {elapsed:.1f}s")
            print(f"📊 Results: {len(relevant_papers)} relevant papers from {len(papers)} total")

            if relevant_papers:
                print(f"🏆 Top paper: {relevant_papers[0]['title']} (score: {relevant_papers[0]['combined_score']:.1f})")

            return {
                'success': True,
                'total_papers': len(papers),
                'relevant_papers': len(relevant_papers),
                'outputs': outputs,
                'elapsed_time': elapsed
            }

        except Exception as e:
            print(f"❌ Filter failed: {e}")
            return {'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Quantum ML Paper Filter')
    parser.add_argument('--days-back', type=int, default=1, help='Days to look back')
    parser.add_argument('--min-score', type=int, default=5, help='Minimum relevance score')
    parser.add_argument('--output-format', choices=['text', 'html', 'both'], default='both')
    parser.add_argument('--weekly-summary', action='store_true', help='Generate weekly summary')

    args = parser.parse_args()

    # Auto-adjust days_back for weekends
    if args.days_back is None:
        today = datetime.datetime.now().weekday()  # 0 = Monday, 6 = Sunday
        args.days_back = 3 if today == 0 else 1  # 3 days back on Monday, 1 otherwise

    # Load configuration
    config = QMLConfig()

    # Create and run filter
    filter_instance = GitHubQMLFilter(config)
    result = filter_instance.run(
        days_back=args.days_back,
        min_score=args.min_score
    )

    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()