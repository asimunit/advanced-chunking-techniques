# Advanced Chunking Techniques for Large Language Models

A comprehensive collection of document chunking strategies optimized for use with Large Language Models (LLMs) like Google Gemini, OpenAI GPT, and others. This repository provides production-ready implementations of various chunking methods to handle large documents effectively while preserving context and meaning.

## 🚀 Features

### 📊 Multiple Chunking Strategies
- **Fixed-Size Chunking**: Consistent token-based splitting with configurable overlap
- **Semantic Chunking**: Content-aware splitting that preserves meaning and structure
- **Overlapping Chunks**: Context-preserving chunking with intelligent overlap strategies
- **Hierarchical Chunking**: Structure-aware chunking that maintains document hierarchy

### 🧠 Intelligent Q&A Systems
- Context-aware retrieval across different chunking strategies
- Adaptive chunk selection based on query characteristics
- Multi-level reasoning for complex document analysis
- Performance optimization and trade-off analysis

### 📈 Analysis & Visualization Tools
- Comprehensive performance comparison between methods
- Interactive visualization of chunk distributions and relationships
- Context preservation metrics and quality analysis
- Best practices recommendations for different document types

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key (or other LLM API)
- Jupyter Notebook/Lab (for running examples)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/advanced-chunking-techniques.git
cd advanced-chunking-techniques
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download additional models**
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. **Set up API credentials**
```bash
# Create a .env file
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

### Requirements

```txt
google-generativeai>=0.3.0
sentence-transformers>=2.2.0
spacy>=3.4.0
nltk>=3.8
scikit-learn>=1.1.0
numpy>=1.21.0
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
tiktoken>=0.4.0
networkx>=2.8.0
anytree>=2.8.0
jupyter>=1.0.0
```

## 🎯 Quick Start

### Basic Usage Example

```python
from chunking_techniques import FixedSizeChunker, SemanticChunker, OverlappingChunker, HierarchicalChunker
import google.generativeai as genai

# Configure your LLM
genai.configure(api_key="your-api-key")

# Sample document
document = """
Your large document content here...
Multiple paragraphs with complex structure...
"""

# 1. Fixed-Size Chunking
fixed_chunker = FixedSizeChunker(chunk_size=512, overlap=50)
fixed_chunks = fixed_chunker.chunk_text(document)
print(f"Fixed-size chunks: {len(fixed_chunks)}")

# 2. Semantic Chunking  
semantic_chunker = SemanticChunker(max_chunk_size=512, strategy='paragraph')
semantic_chunks = semantic_chunker.chunk_text(document)
print(f"Semantic chunks: {len(semantic_chunks)}")

# 3. Overlapping Chunks
overlap_chunker = OverlappingChunker(chunk_size=512, overlap_strategy='percentage', overlap_size=0.2)
overlap_chunks = overlap_chunker.chunk_text(document)
print(f"Overlapping chunks: {len(overlap_chunks)}")

# 4. Hierarchical Chunking
hierarchical_chunker = HierarchicalChunker(max_chunk_size=512)
hierarchy_result = hierarchical_chunker.chunk_text(document)
print(f"Hierarchical chunks: {hierarchy_result['total_chunks']}")
```

## 📋 Chunking Methods Overview

### 1. Fixed-Size Chunking 🔧

**Best for**: Consistent processing, batch operations, simple documents

**How it works**: Splits text into uniform chunks based on token count with optional overlap.

```python
from chunking_techniques.fixed_size import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=512,        # Target tokens per chunk
    overlap=50,            # Overlap tokens between chunks
    method='token'         # 'token', 'character', or 'word'
)

chunks = chunker.chunk_text(document)
```

**Advantages**:
- ✅ Predictable chunk sizes
- ✅ Fast processing
- ✅ Simple implementation
- ✅ Consistent token usage

**Use Cases**:
- Batch document processing
- Simple Q&A systems
- Content that doesn't require structure preservation

### 2. Semantic Chunking 🧠

**Best for**: Preserving meaning, complex documents, academic papers

**How it works**: Analyzes content semantics to create chunks that maintain conceptual coherence.

```python
from chunking_techniques.semantic import SemanticChunker

chunker = SemanticChunker(
    max_chunk_size=512,
    strategy='paragraph',     # 'sentence', 'paragraph', 'structure', 'embedding'
    similarity_threshold=0.5
)

chunks = chunker.chunk_text(document)
```

**Strategies**:
- **Sentence-based**: Groups sentences maintaining natural flow
- **Paragraph-based**: Uses paragraph boundaries for chunking
- **Structure-based**: Respects document headers and sections
- **Embedding-based**: Groups semantically similar content

**Advantages**:
- ✅ Preserves semantic meaning
- ✅ Respects natural language boundaries
- ✅ Better context preservation
- ✅ Improved retrieval quality

**Use Cases**:
- Academic research papers
- Technical documentation
- Complex narrative content
- Knowledge extraction systems

### 3. Overlapping Chunks 🔗

**Best for**: Context preservation, preventing information loss, narrative content

**How it works**: Creates chunks with intelligent overlap to maintain context across boundaries.

```python
from chunking_techniques.overlapping import OverlappingChunker

chunker = OverlappingChunker(
    chunk_size=512,
    overlap_strategy='percentage',  # 'fixed', 'percentage', 'sentence'
    overlap_size=0.2               # 20% overlap
)

chunks = chunker.chunk_text(document)
```

**Overlap Strategies**:
- **Fixed**: Consistent token-based overlap
- **Percentage**: Adaptive overlap based on chunk size
- **Sentence**: Natural language boundary overlap
- **Adaptive**: Content-aware dynamic overlap

**Advantages**:
- ✅ Prevents information loss at boundaries
- ✅ Maintains narrative flow
- ✅ Improved context continuity
- ✅ Better cross-reference handling

**Use Cases**:
- Legal documents
- Medical records
- Story/narrative content
- Cross-referential material

### 4. Hierarchical Chunking 🌳

**Best for**: Structured documents, multi-level analysis, complex organizational content

**How it works**: Preserves document hierarchy creating chunks at multiple structural levels.

```python
from chunking_techniques.hierarchical import HierarchicalChunker

chunker = HierarchicalChunker(
    max_chunk_size=512,
    min_chunk_size=50
)

result = chunker.chunk_text(document)
hierarchy_tree = result['hierarchy_tree']
chunks = result['chunks']
```

**Features**:
- **Header Detection**: Automatic identification of document structure
- **Multi-level Chunks**: Creates chunks at different hierarchy levels
- **Parent-Child Relationships**: Maintains structural connections
- **Adaptive Retrieval**: Level-appropriate chunk selection

**Advantages**:
- ✅ Preserves document structure
- ✅ Multi-level retrieval capabilities
- ✅ Contextual relationships maintained
- ✅ Supports both broad and detailed queries

**Use Cases**:
- Technical manuals
- API documentation
- Academic papers with clear structure
- Policy documents
- Books and reports

## 🎯 Q&A Systems

Each chunking method includes an intelligent Q&A system optimized for that approach:

### Fixed-Size Q&A System

```python
from chunking_techniques.fixed_size import FixedSizeQASystem

qa_system = FixedSizeQASystem(chunk_size=400, overlap=50)
qa_system.load_document(document, "My Document")

result = qa_system.answer_question("What are the main topics discussed?")
print(result["answer"])
```

### Semantic Q&A System

```python
from chunking_techniques.semantic import SemanticQASystem

qa_system = SemanticQASystem(chunking_strategy='structure', max_chunk_size=400)
qa_system.load_document(document, "Research Paper")

result = qa_system.answer_question("Explain the methodology used")
print(result["answer"])
```

### Overlapping Q&A System

```python
from chunking_techniques.overlapping import OverlapAwareQASystem

qa_system = OverlapAwareQASystem(
    overlap_strategy='percentage', 
    chunk_size=400, 
    overlap_size=0.25
)
qa_system.load_document(document, "Legal Document")

result = qa_system.answer_question("What are the key requirements?", use_context_expansion=True)
print(result["answer"])
```

### Hierarchical Q&A System

```python
from chunking_techniques.hierarchical import HierarchicalQASystem

qa_system = HierarchicalQASystem(max_chunk_size=400)
qa_system.load_document(document, "Technical Manual")

# Different retrieval strategies
strategies = ['adaptive', 'broad_first', 'detailed_first', 'multi_level']

for strategy in strategies:
    result = qa_system.answer_question("How does this system work?", strategy=strategy)
    print(f"{strategy}: {result['answer'][:100]}...")
```

## 📊 Performance Comparison

### Choosing the Right Method

| Document Type | Recommended Method | Key Benefits | Best Strategy |
|---------------|-------------------|--------------|---------------|
| **Academic Papers** | Hierarchical/Semantic | Structure preservation, citation context | Structure-based semantic |
| **Technical Manuals** | Hierarchical | Procedural flow, dependencies | Multi-level hierarchical |
| **Legal Documents** | Overlapping/Semantic | Clause relationships, precision | High-overlap semantic |
| **News Articles** | Semantic/Fixed-size | Content density, efficiency | Paragraph-based semantic |
| **Books/Reports** | Hierarchical | Navigation, thematic organization | Chapter-section hierarchy |
| **API Documentation** | Hierarchical | Logical structure, inheritance | Service-endpoint hierarchy |
| **General Content** | Fixed-size/Overlapping | Simplicity, consistency | Moderate overlap fixed-size |

### Performance Metrics

```python
from chunking_techniques.analysis import compare_chunking_methods

# Compare all methods on your document
comparison_results = compare_chunking_methods(
    document=your_document,
    question="Your test question",
    methods=['fixed_size', 'semantic', 'overlapping', 'hierarchical']
)

# Analyze results
for method, metrics in comparison_results.items():
    print(f"{method}:")
    print(f"  Chunks: {metrics['chunk_count']}")
    print(f"  Avg tokens: {metrics['avg_tokens']}")
    print(f"  Context quality: {metrics['context_quality_score']}")
    print(f"  Processing time: {metrics['processing_time']:.3f}s")
```

## 🏆 Best Practices

### General Guidelines

1. **Start Simple**: Begin with fixed-size chunking for proof of concept
2. **Consider Structure**: Use hierarchical chunking for well-structured documents
3. **Preserve Context**: Apply overlapping for documents with cross-references
4. **Test Multiple Methods**: Compare approaches with your specific content
5. **Monitor Performance**: Track both quality and computational costs

### Optimization Tips

```python
# 1. Token Management
from chunking_techniques.utils import optimize_chunk_size

optimal_size = optimize_chunk_size(
    documents=your_documents,
    target_model="gpt-4",  # or "gemini-pro"
    quality_metric="retrieval_accuracy"
)

# 2. Caching for Performance
from chunking_techniques.cache import ChunkCache

cache = ChunkCache(redis_url="redis://localhost:6379")
chunker = SemanticChunker(cache=cache)  # Automatically caches results

# 3. Batch Processing
from chunking_techniques.batch import BatchProcessor

processor = BatchProcessor(
    chunker_type="hierarchical",
    parallel_workers=4
)
results = processor.process_documents(document_list)
```

### Quality Assurance

```python
from chunking_techniques.quality import QualityAnalyzer

analyzer = QualityAnalyzer()

# Analyze chunk quality
quality_report = analyzer.analyze_chunks(
    chunks=your_chunks,
    original_document=document,
    metrics=['coherence', 'completeness', 'overlap_efficiency']
)

print(f"Coherence Score: {quality_report['coherence']:.3f}")
print(f"Information Loss: {quality_report['information_loss']:.1f}%")
```

## 🔧 Advanced Configuration

### Custom Chunking Strategies

```python
from chunking_techniques.base import BaseChunker

class CustomChunker(BaseChunker):
    def __init__(self, custom_param=None):
        super().__init__()
        self.custom_param = custom_param
    
    def chunk_text(self, text: str) -> List[Dict]:
        # Your custom chunking logic
        chunks = []
        # ... implementation ...
        return chunks

# Use your custom chunker
custom_chunker = CustomChunker(custom_param="value")
chunks = custom_chunker.chunk_text(document)
```

### Integration with Vector Databases

```python
from chunking_techniques.integrations import PineconeIntegration, WeaviateIntegration

# Pinecone integration
pinecone_client = PineconeIntegration(
    api_key="your-pinecone-key",
    environment="us-west1-gcp"
)

# Automatically chunk and store
pinecone_client.store_document(
    document=document,
    chunker_type="semantic",
    chunker_config={"strategy": "embedding", "max_chunk_size": 512}
)

# Query with automatic retrieval
results = pinecone_client.query(
    question="What are the main topics?",
    top_k=5
)
```

## 📚 Examples and Tutorials

### Running the Jupyter Notebooks

1. **Fixed-Size Chunking Tutorial**
```bash
jupyter notebook notebooks/Fixed_Size_Chunking_with_Gemini.ipynb
```

2. **Semantic Chunking Tutorial**
```bash
jupyter notebook notebooks/Semantic_Chunking_with_Gemini.ipynb
```

3. **Overlapping Chunks Tutorial**
```bash
jupyter notebook notebooks/Overlapping_Chunks_with_Gemini.ipynb
```

4. **Hierarchical Chunking Tutorial**
```bash
jupyter notebook notebooks/Hierarchical_Chunking_with_Gemini.ipynb
```

### Sample Applications

#### 1. Document Analysis Pipeline

```python
from chunking_techniques.pipelines import DocumentAnalysisPipeline

pipeline = DocumentAnalysisPipeline(
    chunker_type="hierarchical",
    qa_model="gemini-pro",
    analysis_depth="comprehensive"
)

# Process document
results = pipeline.analyze_document(
    document_path="path/to/document.pdf",
    analysis_questions=[
        "What are the main topics?",
        "What are the key findings?",
        "What methodologies were used?"
    ]
)

# Get structured analysis
summary = pipeline.generate_summary(results)
print(summary)
```

#### 2. Multi-Document Knowledge Base

```python
from chunking_techniques.knowledge_base import KnowledgeBase

kb = KnowledgeBase(
    chunking_strategy="adaptive",  # Automatically selects best method per document
    vector_store="pinecone"
)

# Add documents
documents = [
    {"path": "doc1.pdf", "type": "academic_paper"},
    {"path": "doc2.html", "type": "technical_manual"},
    {"path": "doc3.txt", "type": "legal_document"}
]

kb.add_documents(documents)

# Query across all documents
answer = kb.query(
    question="How do these documents relate to machine learning?",
    cross_document_synthesis=True
)
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific chunking method tests
pytest tests/test_fixed_size_chunking.py
pytest tests/test_semantic_chunking.py
pytest tests/test_overlapping_chunks.py
pytest tests/test_hierarchical_chunking.py

# Run performance benchmarks
pytest tests/benchmarks/ -v

# Run integration tests
pytest tests/integration/ -v
```

### Benchmark Results

```bash
# Generate benchmark report
python scripts/run_benchmarks.py --output results/benchmark_report.html

# Compare methods on your data
python scripts/compare_methods.py \
  --input your_documents/ \
  --output comparison_results.json \
  --methods fixed_size,semantic,overlapping,hierarchical
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/advanced-chunking-techniques.git
cd advanced-chunking-techniques

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### Areas for Contribution

- 🔧 **New Chunking Methods**: Implement novel chunking strategies
- 📊 **Performance Optimization**: Improve speed and memory usage
- 🧪 **Testing**: Add test cases and benchmarks
- 📚 **Documentation**: Improve examples and tutorials
- 🔌 **Integrations**: Add support for new LLMs and vector databases
- 🎨 **Visualization**: Enhanced analysis and visualization tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) for the powerful language model API
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [spaCy](https://spacy.io/) for natural language processing
- [Hugging Face](https://huggingface.co/) for transformer models
- The open-source community for various libraries and tools

## 📞 Support

- 📖 **Documentation**: [Full documentation](https://your-docs-site.com)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/advanced-chunking-techniques/discussions)
- 🐛 **Issues**: [Report bugs](https://github.com/yourusername/advanced-chunking-techniques/issues)
- 📧 **Email**: support@yourproject.com

## 🔮 Roadmap

### Upcoming Features

- [ ] **Dynamic Chunking**: Real-time adaptation based on content analysis
- [ ] **Multi-language Support**: Chunking for non-English documents
- [ ] **Visual Document Processing**: Handle images, charts, and tables
- [ ] **Streaming Chunking**: Process large documents in streams
- [ ] **Auto-optimization**: Automatic chunking strategy selection
- [ ] **Cloud Integration**: Native cloud platform support
- [ ] **GUI Interface**: Web-based interface for non-technical users

### Version History

- **v1.0.0** - Initial release with basic chunking methods
- **v1.1.0** - Added semantic chunking strategies
- **v1.2.0** - Implemented overlapping chunks with optimization
- **v1.3.0** - Added hierarchical chunking and multi-level retrieval
- **v1.4.0** - Performance improvements and quality metrics

---

**Made with ❤️ for the AI/ML community**

*If you find this project helpful, please consider giving it a ⭐ on GitHub!*
