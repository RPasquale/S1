# PDF Chat Training Configuration

# Document Collection Settings
DOCUMENTS_FOLDER = "documents"  # Default folder for PDF documents
MAX_DOCUMENTS_FOR_INDEXING = 50  # Maximum number of documents to process for indexing
MAX_DOCUMENTS_FOR_LOADING = 50   # Maximum number of documents to load from existing index

# Training Settings
DEFAULT_TRAINING_HOURS = 24.0
DEFAULT_TARGET_EXPERTISE = 0.95
DEFAULT_MAX_TRAINING_HOURS = 72.0

# System Settings
INDEX_FOLDER = "document-index"  # Generic name for document index
CHECKPOINT_FOLDER = "model_checkpoints"
TRAINING_STATE_FOLDER = "training_state"

# Test Mode Settings
TEST_MODE_SAMPLE_DOCS = [
    """Document Analysis: Modern document analysis involves extracting meaningful information from various types of documents. The key insight is that understanding document structure and content relationships can significantly improve information retrieval and comprehension. Effective analysis requires understanding context, semantic relationships, and document-specific patterns.""",
    
    """Information Retrieval: Information retrieval systems are designed to find relevant information from large document collections. These systems use various techniques including keyword matching, semantic analysis, and machine learning to rank and return the most relevant documents based on user queries.""",
    
    """Natural Language Processing: NLP techniques enable computers to understand, interpret, and generate human language. This includes tasks like text classification, named entity recognition, sentiment analysis, and question answering. Modern NLP systems leverage large language models and transformer architectures."""
]

# Training Question Types (Generic)
QUESTION_TYPES = [
    "analysis",
    "comparison", 
    "explanation",
    "synthesis",
    "evaluation",
    "application"
]

# Training Topics (Generic - will be extracted from documents)
GENERIC_TOPICS = [
    'analysis', 'information', 'data', 'research', 'methodology',
    'concepts', 'principles', 'framework', 'strategy', 'implementation',
    'evaluation', 'assessment', 'comparison', 'optimization', 'management',
    'processing', 'understanding', 'interpretation', 'application', 'development'
]
