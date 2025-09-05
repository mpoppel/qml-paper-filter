"""
Configuration for Quantum ML Paper Filter
Customize these settings for your specific research interests
"""


class QMLConfig:

    def __init__(self):
        # Your key seminal paper
        self.seminal_paper_id = "2008.08605"

        # Quantum ML keywords organized by category with weights
        self.qml_keywords = {
            'essential' : ['quantum'],
            'primary': [
                'quantum machine learning', 'quantum neural network', 'quantum neural networks',
                'supervised learning', 'parametric quantum', 'parametrized quantum',
                'hybrid quantum', 'quantum-classical hybrid', 'fourier'
            ],
            'encoding': [
                'data encoding', 'feature map', 'embedding', 'data uploading',
                'reuploading', 'data re-uploading', 'quantum embedding',
                'encoding circuit', 'feature encoding', 'quantum feature map'
            ],
            'algorithms': [
                'VQC', 'quantum neural network',
            ],
            'theory': [
                'expressivity', 'expressive power', 'barren plateau', 'trainability',
                'fourier analysis',
                'fourier series', 'frequency spectrum', 'universal approximation'
            ],
            'applications': [
                'quantum regression'
            ],
            'techniques': [
                'gradient-based optimization'
            ]
        }

        # Scoring weights for each category
        self.keyword_weights = {
            'essential' : 10 ,
            'primary': 10,  # Core QML terms get highest weight
            'encoding': 8,  # Data encoding is your focus area
            'algorithms': 6,  # Specific algorithms
            'theory': 7,  # Theoretical aspects
            'applications': 4,  # Applications get lower weight
            'techniques': 5  # Implementation techniques
        }

        # Key authors in quantum ML (add/remove as needed)
        self.key_authors = {
            # Pioneers and leading researchers
            'Maria Schuld', 'Marco Cerezo',

            # Data encoding and expressivity specialists
            'Ryan Sweke', 'Johannes Jakob Meyer', 'Adrián Pérez-Salinas',
            'Supanut Thanasilp', 'Sofiene Jerbi', 'Hsin-Yuan Huang',
            'Marco Wiedmann', 'Daniel Scherer', 'Hela Mhiri', 'Jonas Landman'

            # NISQ and variational algorithms
            'Cristina Cîrstoiu', 'Kosuke Mitarai', 'Keisuke Fujii'
        }

        # Related seminal papers to check in references
        self.reference_papers = {
            "2008.08605": {
                "title": "The effect of data encoding on the expressive power of variational quantum machine learning models",
                "authors": ["Schuld", "Sweke", "Meyer"],
                "year": 2021,
                "weight": 15  # Your main reference gets highest weight
            },
            "1804.00633": {
                "title": "Barren plateaus in quantum neural network training landscapes",
                "authors": ["McClean", "Boixo", "Smelyanskiy"],
                "year": 2018,
                "weight": 10
            },
            "2109.11676": {
                "title": "Theory of overparametrization in quantum neural networks",
                "authors": ["Larocca", "Ju", "García-Martín"],
                "year": 2021,
                "weight": 10
            },
            "2411.03450": {
                "title": "TFourier Analysis of Variational Quantum Circuits for Supervised Learning",
                "authors": ["Wiedmann", "Periyasamy", "Daniel D. Scherer"],
                "year": 2024,
                "weight": 10
            }
        }

        # arXiv categories to search (in order of priority)
        self.arxiv_categories = [
            'quant-ph',  # Quantum Physics (highest priority)
            'physics.comp-ph',  # Computational Physics
            'math.OC'  # Optimization and Control
        ]

        # Minimum scores for different priority levels
        self.score_thresholds = {
            'high_priority': 25,  # Definitely relevant
            'medium_priority': 20,  # Likely relevant
            'low_priority': 8,  # Possibly relevant
            'skip': 0  # Not relevant
        }

        # Output preferences
        self.max_papers_in_digest = 15
        self.max_papers_in_email = 10
        self.abstract_preview_length = 200

        # API and rate limiting settings
        self.api_settings = {
            'arxiv_max_results_per_category': 500,
            'request_timeout': 30,
            'rate_limit_delay': 1.0,  # seconds between requests
            'semantic_scholar_delay': 1.5,
            'max_pdf_download_size': 10 * 1024 * 1024  # 10MB limit
        }

    def get_combined_keywords(self) -> list:
        """Get all keywords as a flat list for simple searches"""
        all_keywords = []
        for category_keywords in self.qml_keywords.values():
            all_keywords.extend(category_keywords)
        return list(set(all_keywords))  # Remove duplicates

    def is_key_author(self, author_name: str) -> bool:
        """Check if an author is in our key authors list"""
        author_lower = author_name.lower()
        return any(key_author.lower() in author_lower or author_lower in key_author.lower()
                   for key_author in self.key_authors)

    def get_reference_weight(self, paper_id: str) -> int:
        """Get the importance weight for a reference paper"""
        return self.reference_papers.get(paper_id, {}).get('weight', 5)

    def classify_paper_priority(self, combined_score: float) -> str:
        """Classify paper priority based on combined score"""
        for priority, threshold in sorted(self.score_thresholds.items(),
                                          key=lambda x: x[1], reverse=True):
            if combined_score >= threshold:
                return priority
        return 'skip'