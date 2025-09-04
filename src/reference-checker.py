"""
Reference Checker optimized for GitHub Actions environment
Focuses on reliable API-based checking with fallback methods
"""

import requests
import re
import time
from typing import Dict, List, Optional
from datetime import datetime
import json


class ReferenceChecker:

    def __init__(self):
        # Import configuration
        from config import QMLConfig
        self.config = QMLConfig()

        # Track API usage for rate limiting
        self.last_api_call = {}

    def check_paper_references(self, arxiv_id: str, paper_data: Dict = None, timeout: int = 30) -> Dict:
        """
        Main entry point for reference checking
        Uses multiple methods with graceful degradation
        """
        start_time = time.time()

        # Initialize result structure
        result = {
            'paper_id': arxiv_id,
            'total_score': 0,
            'methods_used': [],
            'found_references': [],
            'processing_time': 0,
            'errors': []
        }

        try:
            # Method 1: Semantic Scholar API (most reliable)
            semantic_result = self._check_semantic_scholar(arxiv_id)
            if semantic_result['score'] > 0:
                result['methods_used'].append('semantic_scholar')
                result['total_score'] += semantic_result['score']
                result['found_references'].extend(semantic_result.get('references', []))

            if semantic_result.get('error'):
                result['errors'].append(f"Semantic Scholar: {semantic_result['error']}")

            # Method 2: Heuristic analysis (always available)
            if paper_data:
                heuristic_result = self._check_heuristic_signals(paper_data)
                result['methods_used'].append('heuristic')
                result['total_score'] += heuristic_result['score']
                result['found_references'].extend(heuristic_result.get('signals', []))

            # Cap total score at reasonable maximum
            result['total_score'] = min(result['total_score'], 100)
            result['processing_time'] = time.time() - start_time

            return result

        except Exception as e:
            result['errors'].append(f"General error: {str(e)}")
            result['processing_time'] = time.time() - start_time
            return result

    def _check_semantic_scholar(self, arxiv_id: str) -> Dict:
        """Check references using Semantic Scholar API"""

        # Rate limiting
        if 'semantic_scholar' in self.last_api_call:
            elapsed = time.time() - self.last_api_call['semantic_scholar']
            if elapsed < self.config.api_settings['semantic_scholar_delay']:
                time.sleep(self.config.api_settings['semantic_scholar_delay'] - elapsed)

        self.last_api_call['semantic_scholar'] = time.time()

        # Clean arXiv ID
        clean_id = arxiv_id.replace("arXiv:", "").replace("v1", "").replace("v2", "")

        try:
            # Semantic Scholar API endpoint
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_id}"

            params = {
                'fields': 'references,references.title,references.authors,references.externalIds,references.year'
            }

            headers = {
                'User-Agent': 'QML-Paper-Filter/1.0 (https://github.com/your-repo)'
            }

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.api_settings['request_timeout']
            )

            if response.status_code == 200:
                data = response.json()
                references = data.get('references', [])
                return self._analyze_references(references)

            elif response.status_code == 404:
                return {'score': 0, 'references': [], 'error': 'Paper not found in Semantic Scholar'}

            else:
                return {'score': 0, 'references': [], 'error': f'HTTP {response.status_code}'}

        except requests.exceptions.Timeout:
            return {'score': 0, 'references': [], 'error': 'Request timeout'}

        except requests.exceptions.RequestException as e:
            return {'score': 0, 'references': [], 'error': f'Request failed: {str(e)}'}

        except Exception as e:
            return {'score': 0, 'references': [], 'error': f'Unexpected error: {str(e)}'}

    def _analyze_references(self, references: List[Dict]) -> Dict:
        """Analyze reference list for target papers"""

        found_refs = []
        total_score = 0

        for ref in references:
            ref_title = ref.get('title', '').lower() if ref.get('title') else ''
            ref_authors = [author.get('name', '').lower() for author in ref.get('authors', []) if author.get('name')]
            external_ids = ref.get('externalIds', {})
            ref_year = ref.get('year')

            # Check against each target paper
            for paper_id, paper_info in self.config.reference_papers.items():
                match_score = 0
                match_reasons = []

                # Direct arXiv ID match (highest confidence)
                arxiv_id = external_ids.get('ArXiv')
                if arxiv_id and arxiv_id == paper_id:
                    match_score = paper_info['weight']
                    match_reasons.append('Exact arXiv ID match')

                # DOI match (high confidence)
                elif external_ids.get('DOI') and 'doi' in paper_info:
                    if external_ids['DOI'].lower() in paper_info['doi'].lower():
                        match_score = paper_info['weight'] * 0.9
                        match_reasons.append('DOI match')

                # Title + author combination (medium confidence)
                elif ref_title and ref_authors:
                    title_words = set(paper_info['title'].lower().split()[:5])  # First 5 words
                    title_overlap = len(title_words.intersection(set(ref_title.split())))

                    author_matches = 0
                    for target_author in paper_info['authors']:
                        if any(target_author.lower() in ref_author for ref_author in ref_authors):
                            author_matches += 1

                    if title_overlap >= 3 and author_matches >= 1:
                        confidence = (title_overlap / 5) * (author_matches / len(paper_info['authors']))
                        match_score = paper_info['weight'] * confidence * 0.7
                        match_reasons.append(f'Title+author similarity (confidence: {confidence:.2f})')

                # Year proximity bonus
                if ref_year and abs(ref_year - paper_info['year']) <= 1:
                    match_score *= 1.1
                    match_reasons.append('Year proximity')

                # Record significant matches
                if match_score >= 3:  # Threshold for counting as a match
                    found_refs.append({
                        'target_paper': paper_id,
                        'target_title': paper_info['title'],
                        'match_score': round(match_score, 1),
                        'confidence': 'high' if match_score >= paper_info['weight'] * 0.8 else 'medium',
                        'reasons': match_reasons,
                        'reference_title': ref.get('title', 'Unknown'),
                        'reference_authors': [author.get('name', '') for author in ref.get('authors', [])][:3]
                    })

                    total_score += match_score

        return {
            'score': round(total_score, 1),
            'references': found_refs,
            'total_references_checked': len(references)
        }

    def _check_heuristic_signals(self, paper_data: Dict) -> Dict:
        """Check for heuristic signals that suggest relevance to target papers"""

        title = paper_data.get('title', '').lower()
        abstract = paper_data.get('summary', '').lower()
        authors = [author.lower() for author in paper_data.get('authors', [])]

        signals = []
        total_score = 0

        # 1. Check for key concepts from target papers
        concept_signals = {
            'data encoding': 12,  # From your seminal paper
            'feature map': 10,
            'expressivity': 10,
            'expressive power': 10,
            'fourier series': 8,
            'variational quantum': 8,
            'barren plateau': 7,
            'quantum embedding': 6,
            'parametrized quantum circuit': 6,
            'quantum neural network': 6,
            'quantum machine learning': 5,
            'quantum advantage': 5,
            'NISQ': 4
        }

        full_text = f"{title} {abstract}"
        for concept, score in concept_signals.items():
            if concept in full_text:
                signals.append({
                    'type': 'concept',
                    'value': concept,
                    'score': score,
                    'location': 'title' if concept in title else 'abstract'
                })
                total_score += score

        # 2. Author overlap with target papers
        for paper_id, paper_info in self.config.reference_papers.items():
            for target_author in paper_info['authors']:
                if any(target_author.lower() in author for author in authors):
                    author_score = paper_info['weight'] * 0.3  # Reduced weight for author overlap
                    signals.append({
                        'type': 'author_overlap',
                        'value': target_author,
                        'target_paper': paper_id,
                        'score': round(author_score, 1)
                    })
                    total_score += author_score

        # 3. Methodological similarity indicators
        method_indicators = {
            'gradient-free optimization': 3,
            'parameter-shift rule': 4,
            'quantum natural gradient': 4,
            'measurement optimization': 3,
            'circuit ansatz': 3,
            'quantum kernel': 4,
            'hybrid optimization': 3
        }

        for method, score in method_indicators.items():
            if method in abstract:  # Only check abstract for methods
                signals.append({
                    'type': 'method',
                    'value': method,
                    'score': score
                })
                total_score += score

        # 4. Recent work bonus (papers citing foundational work are often recent)
        paper_date = paper_data.get('published')
        if paper_date and isinstance(paper_date, datetime):
            days_old = (datetime.now() - paper_date.replace(tzinfo=None)).days
            if days_old <= 30:  # Very recent papers
                recency_bonus = 2
                signals.append({
                    'type': 'recency',
                    'value': f'{days_old} days old',
                    'score': recency_bonus
                })
                total_score += recency_bonus

        return {
            'score': round(min(total_score, 50), 1),  # Cap heuristic score
            'signals': signals,
            'signal_count': len(signals)
        }

    def get_reference_summary(self, result: Dict) -> str:
        """Generate a human-readable summary of reference analysis"""

        if not result.get('found_references'):
            return "No references to target papers detected"

        summary_parts = []

        # Group by method
        semantic_refs = [ref for ref in result['found_references']
                         if isinstance(ref, dict) and 'target_paper' in ref]
        heuristic_signals = [ref for ref in result['found_references']
                             if isinstance(ref, dict) and 'type' in ref]

        if semantic_refs:
            target_papers = set(ref['target_paper'] for ref in semantic_refs)
            summary_parts.append(f"References {len(target_papers)} target paper(s)")

        if heuristic_signals:
            concept_count = len([s for s in heuristic_signals if s.get('type') == 'concept'])
            if concept_count > 0:
                summary_parts.append(f"Contains {concept_count} key concept(s)")

        score_desc = "high relevance" if result['total_score'] >= 15 else "moderate relevance"

        return f"{'; '.join(summary_parts)} ({score_desc})"