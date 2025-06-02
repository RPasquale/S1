import dspy
lm = dspy.LM('ollama_chat/deepseek-r1:8b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

# Define structured output signatures
class Outline(dspy.Signature):
    """Outline a thorough overview of a topic."""

    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()
    section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="mapping from section headings to subheadings")

class DraftSection(dspy.Signature):
    """Draft a top-level section of an article."""

    topic: str = dspy.InputField()
    section_heading: str = dspy.InputField()
    section_subheadings: list[str] = dspy.InputField()
    content: str = dspy.OutputField(desc="markdown-formatted section")

class DraftArticle(dspy.Module):
    def __init__(self):
        self.build_outline = dspy.ChainOfThought(Outline)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []
        for heading, subheadings in outline.section_subheadings.items():
            section_heading = f"## {heading}"
            formatted_subheadings = [f"### {subheading}" for subheading in subheadings]
            section = self.draft_section(topic=outline.title, section_heading=section_heading, section_subheadings=formatted_subheadings)
            sections.append(section.content)
        return dspy.Prediction(title=outline.title, sections=sections)

# Now load the embedding model
from pylate import indexes, models, retrieve
pylate_model_id = "lightonai/Reason-ModernColBERT"  # identifier for the ColBERT embedding model

# Step 1: Load the ColBERT model
model = models.ColBERT(
    model_name_or_path=pylate_model_id,
)

import os

# Determine if index already exists
index_folder = "pylate-index"
voyager_index_path = os.path.join(index_folder, "index.voyager")
index_exists = os.path.exists(voyager_index_path)

# Step 1: Initialize or load the Voyager index
from pylate import indexes
if not index_exists:
    index = indexes.Voyager(
        index_folder=index_folder,
        index_name="index",
        override=True,  # build a new index only once
    )
else:
    index = indexes.Voyager(
        index_folder=index_folder,
        index_name="index",
        override=False,  # reuse existing index
    )

# Step 2: Handle document loading and indexing
documents_ids = []
documents = []

if not index_exists:
    print("Index not found. Creating new index from CFA documents...")
    # Step 3: Encode the documents
    import os
    from PyPDF2 import PdfReader

    doc_folder = r"C:\Users\Admin\OneDrive\CFAL2"  # CFAL2 root containing subfolders of PDFs
    
    for root, dirs, files in os.walk(doc_folder):
        for fname in files:
            if fname.lower().endswith('.pdf'):
                file_path = os.path.join(root, fname)
                reader = PdfReader(file_path)
                text_pages = [page.extract_text() or "" for page in reader.pages]
                documents.append("\n".join(text_pages))
                rel_id = os.path.relpath(file_path, doc_folder)
                documents_ids.append(rel_id)

    print(f"Found {len(documents)} documents. Encoding...")
    documents_embeddings = model.encode(
        documents,
        batch_size=32,
        is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
        show_progress_bar=True,
    )

    # Step 4: Add document embeddings to the index by providing embeddings and corresponding ids
    index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=documents_embeddings,
    )
    print("Index created successfully!")
else:
    print("Loading existing index...")
    # Load document texts from existing PDFs (faster than re-encoding)
    import os
    from PyPDF2 import PdfReader
    
    doc_folder = r"C:\Users\Admin\OneDrive\CFAL2"
    for root, dirs, files in os.walk(doc_folder):
        for fname in files:
            if fname.lower().endswith('.pdf'):
                file_path = os.path.join(root, fname)
                reader = PdfReader(file_path)
                text_pages = [page.extract_text() or "" for page in reader.pages]
                documents.append("\n".join(text_pages))
                rel_id = os.path.relpath(file_path, doc_folder)
                documents_ids.append(rel_id)
    print(f"Loaded {len(documents)} documents from existing files.")

# To load an index, simply instantiate it with the correct folder/name and without overriding it
index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="index",
)

# Build a mapping from document IDs to their text content
import sys
if len(documents_ids) != len(documents):
    print("Document ID/text mismatch, exiting.")
    sys.exit(1)
doc_texts = dict(zip(documents_ids, documents))

# Initialize the Voyager retriever
retriever = retrieve.ColBERT(index=index)

# Define optimized RAG pipeline with memory
class RAG(dspy.Module):
    def __init__(self, num_docs=5, max_history=10):
        self.num_docs = num_docs
        self.max_history = max_history
        self.conversation_history = []  # Store conversation history
        self.respond = dspy.ChainOfThought('context, conversation_history, question -> answer')

    def forward(self, question):
        context = self.search(question, k=self.num_docs)
        
        # Format conversation history for context
        history_text = self._format_conversation_history()
        
        prediction = self.respond(
            context=context, 
            conversation_history=history_text,
            question=question
        )
        
        # Store this Q&A in conversation history
        self._add_to_history(question, prediction.answer if hasattr(prediction, 'answer') else prediction.response)
        
        # Ensure both 'answer' and 'response' attributes are available
        if hasattr(prediction, 'answer') and not hasattr(prediction, 'response'):
            prediction.response = prediction.answer
        elif hasattr(prediction, 'response') and not hasattr(prediction, 'answer'):
            prediction.answer = prediction.response
        return prediction
    
    def _format_conversation_history(self):
        """Format conversation history as a readable string"""
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted_history = []
        for i, (q, a) in enumerate(self.conversation_history, 1):
            formatted_history.append(f"Turn {i}:")
            formatted_history.append(f"User: {q}")
            formatted_history.append(f"Assistant: {a}")
            formatted_history.append("")  # Empty line for separation
        
        return "\n".join(formatted_history)
    
    def _add_to_history(self, question, answer):
        """Add a Q&A pair to conversation history"""
        self.conversation_history.append((question, answer))
        
        # Keep only the most recent exchanges to avoid token limits
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def get_history_summary(self):
        """Get a summary of current conversation history"""
        if not self.conversation_history:
            return "No conversation history."
        
        return f"Conversation history ({len(self.conversation_history)} exchanges):\n" + self._format_conversation_history()
    
    def search(self, query: str, k: int = 3) -> str:
        """Search function integrated into RAG module"""
        # Encode and retrieve
        query_emb = model.encode([query], batch_size=32, is_query=True)
        raw_results = retriever.retrieve(queries_embeddings=query_emb, k=k)[0]
        # Collect top-k document texts
        selected = [doc_texts[res['id']] for res in raw_results]
        context = "\n\n".join(f"Document {i+1}:\n{text}" for i, text in enumerate(selected))
        return context

# Function to create training dataset from CFA documents
def create_cfa_training_dataset(num_examples=50):
    """Generate training examples from CFA documents using question-answer generation"""
    import random
    
    print("Generating training dataset from CFA documents...")
    trainset = []
    
    # Sample documents for training data generation
    sample_docs = random.sample(list(doc_texts.items()), min(num_examples // 5, len(doc_texts)))
    
    # Question generation signatures
    class GenerateQuestions(dspy.Signature):
        """Generate specific, detailed questions about CFA Level 2 content from a document excerpt."""
        
        document_text: str = dspy.InputField(desc="excerpt from CFA Level 2 study material")
        questions: list[str] = dspy.OutputField(desc="list of 3-5 specific questions that can be answered from this document")
    
    class AnswerQuestion(dspy.Signature):
        """Answer a CFA Level 2 question using provided context."""
        
        context: str = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="comprehensive answer based on the context")
    
    question_generator = dspy.ChainOfThought(GenerateQuestions)
    answer_generator = dspy.ChainOfThought(AnswerQuestion)
    
    for doc_id, doc_text in sample_docs:
        try:
            # Take a reasonable chunk of the document (first 2000 chars)
            chunk = doc_text[:2000] if len(doc_text) > 2000 else doc_text
            
            # Generate questions from this chunk
            questions_result = question_generator(document_text=chunk)
            
            # Generate answers for each question
            for question in questions_result.questions[:3]:  # Limit to 3 questions per doc
                try:
                    answer_result = answer_generator(context=chunk, question=question)
                    
                    # Create training example
                    example = dspy.Example(
                        question=question,
                        answer=answer_result.answer
                    ).with_inputs("question")
                    
                    trainset.append(example)
                    print(f"Generated example {len(trainset)}: {question[:60]}...")
                    
                    if len(trainset) >= num_examples:
                        break
                        
                except Exception as e:
                    print(f"Error generating answer: {e}")
                    continue
                    
            if len(trainset) >= num_examples:
                break
                
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")
            continue
    
    print(f"Generated {len(trainset)} training examples")
    return trainset

# Define an answering function that retrieves relevant docs and uses the LLM with memory
from dspy import ChainOfThought

# Global conversation history for the fallback function
fallback_conversation_history = []

def answer_question_with_docs(query: str, top_k: int = 3) -> str:
    global fallback_conversation_history
    
    # Encode and retrieve
    query_emb = model.encode([query], batch_size=32, is_query=True)
    raw_results = retriever.retrieve(queries_embeddings=query_emb, k=top_k)[0]
    # Collect top-k document texts
    selected = [doc_texts[res['id']] for res in raw_results]

    context = "\n\n".join(f"Document {i+1}:\n{text}" for i, text in enumerate(selected))
    
    # Format conversation history
    history_text = ""
    if fallback_conversation_history:
        history_parts = []
        for i, (q, a) in enumerate(fallback_conversation_history[-5:], 1):  # Last 5 exchanges
            history_parts.append(f"Turn {i}: User: {q}")
            history_parts.append(f"Turn {i}: Assistant: {a}")
        history_text = "\n".join(history_parts) + "\n\n"
    
    # Build prompt for LLM
    prompt = f"""Previous conversation:
{history_text}Use the following documents to answer the current question:
{context}

Current question: {query}
Answer:"""
    
    cot = ChainOfThought('question -> response')
    res = cot(question=prompt)
    
    # Store in fallback history
    fallback_conversation_history.append((query, res.response))
    if len(fallback_conversation_history) > 10:  # Keep last 10 exchanges
        fallback_conversation_history = fallback_conversation_history[-10:]
    
    return res.response

# Initialize modules
draft_article = DraftArticle()
rag_module = RAG(num_docs=3)

# Training and optimization setup
def setup_optimized_rag():
    """Set up and optimize the RAG pipeline with CFA training data"""
    print("Setting up optimized RAG pipeline...")
    
    # Generate training dataset from CFA documents
    trainset = create_cfa_training_dataset(num_examples=30)
    if len(trainset) == 0:
        print("No training data generated, using unoptimized RAG")
        return rag_module
    
    # Split into train/validation
    train_size = int(0.8 * len(trainset))
    train_data = trainset[:train_size]
    val_data = trainset[train_size:]
    
    print(f"Training with {len(train_data)} examples, validating with {len(val_data)} examples")
    
    try:
        # Windows-specific compatibility settings
        import os
        import multiprocessing
        
        # Set single-threaded environment
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Force multiprocessing to use spawn method (Windows default)
        if hasattr(multiprocessing, 'set_start_method'):
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
        
        print("Initializing MIPROv2 optimizer with Windows compatibility settings...")
        

        
        # Create a simple optimization by using few-shot examples
        from dspy.teleprompt import BootstrapFewShot
        
        # Override input to automatically accept
        import builtins
        original_input = builtins.input
        builtins.input = lambda prompt="": "y"
        
        try:
            # Use BootstrapFewShot as a more stable alternative to MIPROv2
            optimizer = BootstrapFewShot(
                metric=lambda gold, pred, trace=None: len(pred.answer.split()) > 5,  # Simple metric
                max_bootstrapped_demos=2,
                max_labeled_demos=2,

            )
            
            print("Compiling optimized RAG model...")
            optimized_rag = optimizer.compile(RAG(num_docs=3), trainset=train_data[:5])  # Use smaller subset
            
        finally:
            # Restore original input function
            builtins.input = original_input
        
        # Simple evaluation
        if val_data and len(val_data) > 0:
            print("Testing optimized model on validation data...")
            test_example = val_data[0]
            try:
                result = optimized_rag(test_example.question)
                print(f"Sample result: {result.answer[:100]}...")
            except Exception as eval_e:
                print(f"Evaluation failed: {eval_e}")
        
        print("RAG optimization completed successfully!")
        return optimized_rag
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Falling back to unoptimized RAG")
        return rag_module

# Main interactive loop
if __name__ == "__main__":
    print("Interactive QA over indexed PDFs with conversation memory. Commands:")
    print("- 'article:<topic>' to generate a structured article")
    print("- 'optimize' to train and optimize the RAG pipeline")
    print("- 'clear' to clear conversation history")
    print("- 'history' to view conversation history")
    print("- Any other input for Q&A")
    print("- Blank input to exit")
    
    # Start with unoptimized RAG
    current_rag = rag_module
    
    while True:
        user_input = input("\nEnter command or question: ")
        if not user_input:
            break
            
        if user_input.strip().lower() == "optimize":
            print("\nStarting RAG optimization with CFA training data...")
            current_rag = setup_optimized_rag()
            
        elif user_input.strip().lower() == "clear":
            current_rag.clear_history()
            fallback_conversation_history.clear()
            print("All conversation history cleared.")
            
        elif user_input.strip().lower() == "history":
            print("\n" + current_rag.get_history_summary())
            
        elif user_input.startswith("article:"):
            topic = user_input[8:].strip()
            print(f"\nGenerating article about: {topic}")
            try:
                article = draft_article(topic=topic)
                print(f"\n# {article.title}\n")
                for section in article.sections:
                    print(section)
                    print("\n" + "="*50 + "\n")
            except Exception as e:
                print(f"Error generating article: {e}")
                
        else:
            # Use current RAG for Q&A (optimized or unoptimized)
            try:
                answer = current_rag.forward(user_input)
                print("\nAnswer:\n", answer.response)
            except Exception as e:
                print(f"Error in RAG pipeline: {e}")
                # Fallback to original function
                answer = answer_question_with_docs(user_input, top_k=3)
                print("\nAnswer (fallback):\n", answer)
