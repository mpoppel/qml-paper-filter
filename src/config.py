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
            'primary': [
                'quantum machine learning', 'quantum neural network', 'quantum neural networks',
                'variational quantum', 'quantum classifier', 'quantum kernel',
                'quantum circuit learning', 'parametric quantum', 'parametrized quantum',
                'hybrid quantum', 'quantum-classical hybrid'
            ],
            'encoding': [
                'data encoding', 'feature map', 'embedding', 'data uploading',
                'reuploading', 'data re-uploading', 'quantum embedding',
                'encoding circuit', 'feature encoding', 'quantum feature map'
            ],
            'algorithms': [
                'VQC', 'QAOA', 'VQE', 'PQC', 'variational quantum circuit',
                'parametrized quantum circuit', 'quantum approximate optimization',
                'variational quantum eigensolver', 'quantum neural network',
                'quantum convolutional', 'quantum recurrent'
            ],
            'theory': [
                'expressivity', 'expressive power', 'barren plateau', 'trainability',
                'quantum advantage', 'quantum supremacy', 'quantum speedup',
                'generalization', 'quantum generalization', 'fourier analysis',
                'fourier series', 'frequency spectrum', 'universal approximation'
            ],
            'applications': [
                'NISQ', 'near-term quantum', 'quantum classification',
                'quantum regression', 'quantum clustering', 'quantum reinforcement',
                'quantum optimization', 'quantum sensing', 'quantum chemistry'
            ],
            'techniques': [
                'gradient-based optimization', 'parameter-shift rule',
                'quantum natural gradient', 'quantum fisher information',
                'measurement optimization', 'observable optimization',
                'quantum kernel methods', 'quantum support vector'
            ]
        }

        # Scoring weights for each category
        self.keyword_weights = {
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
            'Maria Schuld', 'Seth Lloyd', 'John Preskill', 'Jacob Biamonte',
            'Peter Wittek', 'Nathan Wiebe', 'Marcello Benedetti', 'Stefan Woerner',
            'Ryan LaRose', 'Vojtech Dunjko', 'Patrick Coles', 'Marco Cerezo',

            # Data encoding and expressivity specialists
            'Ryan Sweke', 'Johannes Jakob Meyer', 'Adrián Pérez-Salinas',
            'Supanut Thanasilp', 'Sofiene Jerbi', 'Hsin-Yuan Huang',

            # Quantum advantage and complexity
            'Yunchao Liu', 'Srinivasan Arunachalam', 'Juan Carrasquilla',
            'Giacomo Torlai', 'Giuseppe Carleo', 'Lukasz Cincio',

            # NISQ and variational algorithms
            'Andrew Arrasmith', 'Tyler Volkoff', 'Samson Wang',
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
            "1707.08561": {
                "title": "Quantum machine learning: a classical perspective",
                "authors": ["Ciliberto", "Schuld", "Rocchetto"],
                "year": 2018,
                "weight": 12
            },
            "1802.06002": {
                "title": "Classification with quantum neural networks on near term processors",
                "authors": ["Farhi", "Neven"],
                "year": 2018,
                "weight": 10
            },
            "1804.00633": {
                "title": "Barren plateaus in quantum neural network training landscapes",
                "authors": ["McClean", "Boixo", "Smelyanskiy"],
                "year": 2018,
                "weight": 10
            },
            "2001.03622": {
                "title": "Quantum embeddings for machine learning",
                "authors": ["Lloyd", "Schuld", "Ijaz"],
                "year": 2020,
                "weight": 8
            },
            "2103.05561": {
                "title": "Power of data in quantum machine learning",
                "authors": ["Huang", "Broughton", "Mohseni"],
                "year": 2021,
                "weight": 10
            },
            "2109.11676": {
                "title": "Theory of overparametrization in quantum neural networks",
                "authors": ["Larocca", "Ju", "García-Martín"],
                "year": 2021,
                "weight": 8
            },
            "2105.02276": {
                "title": "Generalization in quantum machine learning from few training data",
                "authors": ["Caro", "Huang", "Ezzell"],
                "year": 2022,
                "weight": 8
            }
        }

        # arXiv categories to search (in order of priority)
        self.arxiv_categories = [
            'quant-ph',  # Quantum Physics (highest priority)
            'cs.LG',  # Machine Learning
            'cs.AI',  # Artificial Intelligence
            'stat.ML',  # Statistics - Machine Learning
            'physics.comp-ph',  # Computational Physics
            'math.OC'  # Optimization and Control
        ]

        # Minimum scores for different priority levels
        self.score_thresholds = {
            'high_priority': 25,  # Definitely relevant
            'medium_priority': 15,  # Likely relevant
            'low_priority': 8,  # Possibly relevant
            'skip': 0  # Not relevant
        }

        # Output preferences
        self.max_papers_in_digest = 15
        self.max_papers_in_email = 10
        self.abstract_preview_length = 200

        # API and rate limiting settings
        self.api_settings = {
            'arxiv_max_results_per_category': 200,
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