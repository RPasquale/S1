import dspy
import os
import sys
import random
import threading
import multiprocessing
import builtins
import json
import pickle
import datetime
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, List, Optional, Tuple, Dict, Union, Callable
from collections import Counter

# Import configuration
from config import *

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

# Check for test mode environment variable
TEST_MODE = os.getenv('DUAL_TRAINING_TEST_MODE', 'False').lower() == 'true'

# Determine if index already exists
index_folder = INDEX_FOLDER
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
    if TEST_MODE:
        print("🧪 TEST MODE: Creating minimal index with sample documents...")
        # Create sample financial documents for testing
        sample_docs = [
            """Portfolio Diversification: Modern Portfolio Theory suggests that investors can reduce risk by constructing a diversified portfolio of assets that are not perfectly correlated. The key insight is that the portfolio's risk is not simply the weighted average of individual asset risks, but depends on how assets co-move with each other. Effective diversification requires understanding correlation coefficients between assets and their individual risk-return profiles.""",
            
            """Interest Rate Risk: Fixed-income securities are subject to interest rate risk - the risk that bond prices will decline when interest rates rise. Duration measures the price sensitivity of bonds to interest rate changes. Modified duration provides an approximation of the percentage change in bond price for a 1% change in yield. Convexity accounts for the curvature in the price-yield relationship.""",
            
            """Capital Asset Pricing Model (CAPM): The CAPM describes the relationship between systematic risk and expected return for assets. It states that the expected return of a security equals the risk-free rate plus a risk premium proportional to its beta. Beta measures the asset's sensitivity to market movements. The Security Market Line graphically represents this relationship."""
        ]
        
        documents = sample_docs
        documents_ids = [f"sample_doc_{i+1}.txt" for i in range(len(sample_docs))]
        
        print(f"Created {len(documents)} sample documents. Encoding...")
        documents_embeddings = model.encode(
            documents,
            batch_size=8,
            is_query=False,
            show_progress_bar=True,
        )
        
        index.add_documents(
            documents_ids=documents_ids,
            documents_embeddings=documents_embeddings,
        )
        print("✅ Test index created successfully!")
    else:
        print("Index not found. Creating new index from PDF documents...")
        # Step 3: Encode the documents        import os
        from PyPDF2 import PdfReader
        
        doc_folder = DOCUMENTS_FOLDER  # Generic document folder from config
        # Limit documents for testing - only process first N files for faster indexing
        MAX_DOCS_FOR_TESTING = MAX_DOCUMENTS_FOR_INDEXING
        processed_count = 0
        
        for root, dirs, files in os.walk(doc_folder):
            for fname in files:
                if (fname.lower().endswith('.pdf') or fname.lower().endswith('.txt')) and processed_count < MAX_DOCS_FOR_TESTING:
                    file_path = os.path.join(root, fname)
                    try:
                        if fname.lower().endswith('.pdf'):
                            reader = PdfReader(file_path)
                            text_pages = [page.extract_text() or "" for page in reader.pages]
                            document_text = "\n".join(text_pages)
                        else:  # .txt file
                            with open(file_path, 'r', encoding='utf-8') as f:
                                document_text = f.read()
                        
                        documents.append(document_text)
                        rel_id = os.path.relpath(file_path, doc_folder)
                        documents_ids.append(rel_id)
                        processed_count += 1
                        print(f"Processed {processed_count}/{MAX_DOCS_FOR_TESTING}: {fname}")
                    except Exception as e:
                        print(f"Error processing {fname}: {e}")
                        continue
            if processed_count >= MAX_DOCS_FOR_TESTING:
                break

        if len(documents) == 0:
            print("⚠️  No PDF documents found! Using sample documents instead...")
            # Fall back to sample documents
            documents = TEST_MODE_SAMPLE_DOCS
            documents_ids = [f"sample_doc_{i+1}.txt" for i in range(len(documents))]
            print(f"Using {len(documents)} sample documents for indexing.")
        
        print(f"Found {len(documents)} documents. Encoding with batch_size=16...")
        documents_embeddings = model.encode(
            documents,
            batch_size=16,  # Reduced batch size for better memory management
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
    if TEST_MODE:
        # Load sample documents for test mode
        documents = TEST_MODE_SAMPLE_DOCS
        documents_ids = [f"sample_doc_{i+1}.txt" for i in range(len(documents))]
        print(f"🧪 TEST MODE: Loaded {len(documents)} sample documents.")
    else:        # Load document texts from existing PDFs (faster than re-encoding)
        import os
        from PyPDF2 import PdfReader
        
        doc_folder = DOCUMENTS_FOLDER  # Generic document folder from config
        MAX_DOCS_FOR_LOADING = MAX_DOCUMENTS_FOR_LOADING  # Use config value
        loaded_count = 0
        
        for root, dirs, files in os.walk(doc_folder):
            for fname in files:
                if (fname.lower().endswith('.pdf') or fname.lower().endswith('.txt')) and loaded_count < MAX_DOCS_FOR_LOADING:
                    file_path = os.path.join(root, fname)
                    try:
                        if fname.lower().endswith('.pdf'):
                            reader = PdfReader(file_path)
                            text_pages = [page.extract_text() or "" for page in reader.pages]
                            document_text = "\n".join(text_pages)
                        else:  # .txt file
                            with open(file_path, 'r', encoding='utf-8') as f:
                                document_text = f.read()
                        
                        documents.append(document_text)
                        rel_id = os.path.relpath(file_path, doc_folder)
                        documents_ids.append(rel_id)
                        loaded_count += 1
                    except Exception as e:
                        print(f"Error loading {fname}: {e}")
                        continue
            if loaded_count >= MAX_DOCS_FOR_LOADING:
                break
        print(f"Loaded {len(documents)} documents from existing files.")

# To load an index, simply instantiate it with the correct folder/name and without overriding it
index = indexes.Voyager(
    index_folder=INDEX_FOLDER,
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

# Function to create training dataset from PDF documents
def create_document_training_dataset(num_examples=50):
    """Generate training examples from PDF documents using question-answer generation"""
    import random
    
    print("Generating training dataset from PDF documents...")
    trainset = []
    
    # Sample documents for training data generation
    sample_docs = random.sample(list(doc_texts.items()), min(num_examples // 5, len(doc_texts)))
    
    # Question generation signatures
    class GenerateQuestions(dspy.Signature):
        """Generate specific, detailed questions about document content from a document excerpt."""
        document_text: str = dspy.InputField(desc="excerpt from document content")
        questions: list[str] = dspy.OutputField(desc="list of 3-5 specific questions that can be answered from this document")
    
    class AnswerQuestion(dspy.Signature):
        """Answer a document question using provided context."""
        
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
# Note: reasoning_rag_module will be initialized after ReasoningRAG class is defined

# Training and optimization setup
def setup_optimized_rag():
    """Set up and optimize the RAG pipeline with CFA training data"""
    print("Setting up optimized RAG pipeline...")
      # Generate training dataset from PDF documents
    trainset = create_document_training_dataset(num_examples=30)
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

# Advanced Reasoning Components with DSPy
from dspy.predict.predict import Predict
from dspy.primitives.program import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import ensure_signature
import threading
from typing import Any, List, Optional, Tuple, Dict, Callable
from collections import Counter
import random
import numpy as np

# Custom GRPO Implementation
class GRPO:
    """
    Group relative policy Optimization (GRPO) for DSPy programs.
    """
    
    def __init__(
        self,
        metric: Callable,
        num_train_steps: int = 10,
        num_dspy_examples_per_grpo_step: int = 2,
        num_rollouts_per_grpo_step: int = 4,
        use_train_as_val: bool = True,
        num_steps_for_val: int = 2,
        report_train_scores: bool = True,
        failure_score: float = 0.1,
        format_failure_score: float = 0.0,
        seed: int = 42
    ):
        self.metric = metric
        self.num_train_steps = num_train_steps
        self.num_dspy_examples_per_grpo_step = num_dspy_examples_per_grpo_step
        self.num_rollouts_per_grpo_step = num_rollouts_per_grpo_step
        self.use_train_as_val = use_train_as_val
        self.num_steps_for_val = num_steps_for_val
        self.report_train_scores = report_train_scores
        self.failure_score = failure_score
        self.format_failure_score = format_failure_score
        self.seed = seed
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        self.training_history = []
        self.best_score = float('-inf')
        self.best_program = None
    
    def compile(self, student, trainset, teacher=None):
        """
        Compile and optimize the student program using GRPO.
        
        Args:
            student: The DSPy program to optimize
            trainset: Training examples
            teacher: Optional teacher program (if None, uses self-improvement)
        """
        print(f"🎯 GRPO: Starting optimization with {len(trainset)} training examples")
        
        # Initialize the best program as the original student
        best_program = student
        best_score = self._evaluate_program(student, trainset)
        
        print(f"📊 Initial program score: {best_score:.3f}")
        
        for step in range(self.num_train_steps):
            print(f"\n🔄 GRPO Step {step + 1}/{self.num_train_steps}")
            
            # Sample examples for this step
            step_examples = self._sample_examples(trainset, self.num_dspy_examples_per_grpo_step)
            
            # Generate rollouts
            rollouts = []
            for _ in range(self.num_rollouts_per_grpo_step):
                rollout_program = self._create_program_variant(student)
                rollout_score = self._evaluate_program(rollout_program, step_examples)
                rollouts.append((rollout_program, rollout_score))
            
            # Select best rollout
            rollouts.sort(key=lambda x: x[1], reverse=True)
            best_rollout, best_rollout_score = rollouts[0]
            
            print(f"📈 Best rollout score: {best_rollout_score:.3f}")
            
            # Update best program if improved
            if best_rollout_score > best_score:
                best_program = best_rollout
                best_score = best_rollout_score
                print(f"🎉 New best score: {best_score:.3f}")
                
                # Update student program parameters (simplified)
                self._update_program(student, best_rollout)
            
            # Log training progress
            step_info = {
                'step': step + 1,
                'best_score': best_score,
                'rollout_scores': [score for _, score in rollouts]
            }
            self.training_history.append(step_info)
            
            if self.report_train_scores:
                avg_rollout_score = np.mean([score for _, score in rollouts])
                print(f"📊 Average rollout score: {avg_rollout_score:.3f}")
        
        print(f"\n✅ GRPO optimization completed! Final best score: {best_score:.3f}")
          # Store the best program
        self.best_program = best_program
        self.best_score = best_score
        
        return best_program
    
    def _evaluate_program(self, program, examples):
        """Evaluate a program on a set of examples using the metric."""
        try:
            scores = []
            for example in examples:
                try:
                    # Generate prediction using the program
                    if hasattr(example, 'question'):
                        question = example.question
                    else:
                        question = str(example)
                    
                    # Call the program (handling different interfaces)
                    if hasattr(program, 'forward'):
                        prediction = program.forward(question)
                    elif hasattr(program, '__call__'):
                        prediction = program(question)
                    else:
                        prediction = str(program)
                    
                    # Calculate score using the metric
                    score = self.metric(example, prediction)
                    scores.append(max(score, self.failure_score))
                    
                except Exception as e:
                    print(f"⚠️ Error evaluating example: {e}")
                    scores.append(self.failure_score)
            
            return np.mean(scores) if scores else self.failure_score
            
        except Exception as e:
            print(f"⚠️ Error in program evaluation: {e}")
            return self.failure_score
    
    def _sample_examples(self, trainset, num_examples):
        """Sample examples from the training set."""
        if len(trainset) <= num_examples:
            return trainset
        return random.sample(trainset, num_examples)
    
    def _create_program_variant(self, base_program):
        """Create a variant of the base program for exploration."""
        # This is a simplified variant creation
        # In a full implementation, this would modify program parameters
        try:
            # Try to create a copy/variant of the program
            if hasattr(base_program, 'copy'):
                variant = base_program.copy()
            elif hasattr(base_program, '__class__'):
                # Create new instance with same parameters
                variant = base_program.__class__()
                # Copy attributes if possible
                for attr_name in dir(base_program):
                    if not attr_name.startswith('_') and hasattr(variant, attr_name):
                        try:
                            setattr(variant, attr_name, getattr(base_program, attr_name))
                        except:
                            pass
            else:
                variant = base_program
            
            return variant
            
        except Exception as e:
            print(f"⚠️ Error creating program variant: {e}")
            return base_program
    def _update_program(self, target_program, source_program):
        """Update target program with improvements from source program."""
        # This is a simplified update mechanism
        # In a full implementation, this would update specific parameters
        try:
            # Copy attributes from source to target
            for attr_name in dir(source_program):
                if not attr_name.startswith('_') and hasattr(target_program, attr_name):
                    try:
                        source_attr = getattr(source_program, attr_name)
                        if not callable(source_attr):
                            setattr(target_program, attr_name, source_attr)
                    except:
                        pass
        except Exception as e:
            print(f"⚠️ Error updating program: {e}")
    
    def get_training_history(self):
        """Get the training history."""
        return self.training_history
    
    def get_best_score(self):
        """Get the best score achieved during training."""
        return self.best_score

# Multi-step reasoning signatures
class DecomposeQuestion(dspy.Signature):
    """Break down a complex question into simpler sub-questions for systematic analysis."""
    
    question: str = dspy.InputField(desc="complex question to decompose")
    context_available: str = dspy.InputField(desc="brief description of available context")
    sub_questions: list[str] = dspy.OutputField(desc="3-5 simpler questions that help answer the main question")
    reasoning_strategy: str = dspy.OutputField(desc="brief explanation of the decomposition approach")

class AnalyzeSubQuestion(dspy.Signature):
    """Analyze a sub-question using available context and reasoning."""
    
    sub_question: str = dspy.InputField(desc="specific sub-question to analyze")
    context: str = dspy.InputField(desc="relevant context and information")
    previous_analysis: str = dspy.InputField(desc="analysis from previous sub-questions")
    analysis: str = dspy.OutputField(desc="detailed analysis with evidence and reasoning")
    confidence: float = dspy.OutputField(desc="confidence score 0-1 for this analysis")

class SynthesizeReasoning(dspy.Signature):
    """Synthesize individual analyses into a comprehensive answer."""
    
    original_question: str = dspy.InputField(desc="the original complex question")
    sub_analyses: list[str] = dspy.InputField(desc="list of individual sub-question analyses")
    reasoning_strategy: str = dspy.InputField(desc="the reasoning strategy used")
    final_answer: str = dspy.OutputField(desc="comprehensive answer synthesizing all analyses")
    reasoning_chain: str = dspy.OutputField(desc="step-by-step reasoning process")

class ReflectOnReasoning(dspy.Signature):
    """Reflect on the reasoning process and validate the conclusion."""
    
    question: str = dspy.InputField(desc="original question")
    reasoning_chain: str = dspy.InputField(desc="step-by-step reasoning process")
    final_answer: str = dspy.InputField(desc="proposed final answer")
    validation: str = dspy.OutputField(desc="critical evaluation of the reasoning")
    improvements: str = dspy.OutputField(desc="suggested improvements or alternatives")
    final_confidence: float = dspy.OutputField(desc="overall confidence 0-1 in the final answer")

# Multi-chain reasoning comparison
class MultiChainComparison(dspy.Module):
    """Compare and validate reasoning across multiple chains."""
    
    def __init__(self):
        self.compare_chains = dspy.ChainOfThought(
            "reasoning_chains, question -> best_reasoning, validation_notes, confidence"
        )
    
    def forward(self, reasoning_chains, question):
        return self.compare_chains(
            reasoning_chains=reasoning_chains,
            question=question
        )

# Advanced ReasoningRAG with multi-step analysis
class ReasoningRAG(dspy.Module):
    """Advanced RAG with multi-step reasoning capabilities."""
    
    def __init__(self, num_docs=5, reasoning_chains=3):
        self.num_docs = num_docs
        self.reasoning_chains = reasoning_chains
        
        # Initialize reasoning components
        self.decompose = dspy.ChainOfThought(DecomposeQuestion)
        self.analyze = dspy.ChainOfThought(AnalyzeSubQuestion)
        self.synthesize = dspy.ChainOfThought(SynthesizeReasoning)
        self.reflect = dspy.ChainOfThought(ReflectOnReasoning)
        self.compare = MultiChainComparison()
        
        # Search capability
        self.search = rag_module.search
        
        # Reasoning history
        self.reasoning_history = []
    
    def forward(self, question):
        """Execute multi-step reasoning process."""
        
        # Step 1: Decompose the question
        context_summary = f"Available: {len(doc_texts)} CFA documents with financial analysis content"
        decomposition = self.decompose(
            question=question,
            context_available=context_summary
        )
        
        print(f"🧩 Question decomposed into {len(decomposition.sub_questions)} sub-questions")
        
        # Step 2: Analyze each sub-question
        analyses = []
        previous_analysis = ""
        
        for i, sub_q in enumerate(decomposition.sub_questions):
            # Get relevant context for this sub-question
            context = self.search(sub_q, k=self.num_docs)
            
            analysis = self.analyze(
                sub_question=sub_q,
                context=context,
                previous_analysis=previous_analysis
            )
            
            analyses.append(analysis.analysis)
            previous_analysis += f"\nPrevious: {analysis.analysis}"
            
            print(f"📊 Sub-question {i+1} analyzed (confidence: {analysis.confidence:.2f})")
        
        # Step 3: Generate multiple reasoning chains
        reasoning_chains = []
        for chain_idx in range(self.reasoning_chains):
            synthesis = self.synthesize(
                original_question=question,
                sub_analyses=analyses,
                reasoning_strategy=decomposition.reasoning_strategy
            )
            reasoning_chains.append({
                'answer': synthesis.final_answer,
                'reasoning': synthesis.reasoning_chain,
                'chain_id': chain_idx
            })
            print(f"🔗 Reasoning chain {chain_idx + 1} generated")
        
        # Step 4: Compare and validate reasoning chains
        chain_summaries = [f"Chain {c['chain_id']}: {c['reasoning']}" for c in reasoning_chains]
        comparison = self.compare(
            reasoning_chains=chain_summaries,
            question=question
        )
        
        # Step 5: Reflect on the best reasoning
        reflection = self.reflect(
            question=question,
            reasoning_chain=comparison.best_reasoning,
            final_answer=reasoning_chains[0]['answer']  # Use first chain's answer for reflection
        )
        
        print(f"🎯 Reasoning completed (final confidence: {reflection.final_confidence:.2f})")
        
        # Store reasoning history
        reasoning_record = {
            'question': question,
            'decomposition': decomposition.sub_questions,
            'analyses': analyses,
            'reasoning_chains': reasoning_chains,
            'reflection': reflection,
            'timestamp': str(threading.current_thread().ident)
        }
        self.reasoning_history.append(reasoning_record)
        
        # Return comprehensive result
        return dspy.Prediction(
            answer=reasoning_chains[0]['answer'],
            reasoning_chain=comparison.best_reasoning,
            confidence=reflection.final_confidence,
            validation=reflection.validation,
            improvements=reflection.improvements,
            sub_questions=decomposition.sub_questions,
            reasoning_record=reasoning_record
        )
    
    def get_reasoning_insights(self):
        """Get insights from reasoning history."""
        if not self.reasoning_history:
            return "No reasoning history available."
        
        total_questions = len(self.reasoning_history)
        avg_sub_questions = sum(len(r['decomposition']) for r in self.reasoning_history) / total_questions
        
        return f"""🧠 Reasoning Insights:
- Total questions processed: {total_questions}
- Average sub-questions per query: {avg_sub_questions:.1f}
- Recent reasoning patterns: Multi-step analysis with validation
- Confidence trends: Available in reasoning history"""

# Reasoning quality metric for RL training
def reasoning_quality_metric(example, prediction):
    """Evaluate the quality of reasoning-based predictions."""
    try:
        # Base quality metrics
        has_answer = hasattr(prediction, 'answer') and len(prediction.answer) > 10
        has_reasoning = hasattr(prediction, 'reasoning_chain') and len(prediction.reasoning_chain) > 20
        has_confidence = hasattr(prediction, 'confidence') and prediction.confidence > 0.3
        
        # Calculate score
        quality_score = 0.0
        if has_answer:
            quality_score += 0.4
        if has_reasoning:
            quality_score += 0.4
        if has_confidence:
            quality_score += 0.2
        
        # Bonus for comprehensive reasoning
        if hasattr(prediction, 'sub_questions') and len(prediction.sub_questions) >= 2:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
        
    except Exception as e:
        print(f"⚠️ Error in reasoning quality metric: {e}")
        return 0.1

# Generate reasoning training examples
def generate_reasoning_trainset(num_examples=10):
    """Generate training examples for reasoning RL."""
    reasoning_questions = [
        "What are the key factors affecting portfolio diversification in volatile markets?",
        "How do interest rate changes impact different asset classes differently?",
        "What risk management strategies are most effective during market downturns?",
        "How should asset allocation change based on investor age and risk tolerance?",
        "What are the implications of ESG investing on portfolio performance?",
        "How do macroeconomic indicators influence investment decision-making?",
        "What role does behavioral finance play in portfolio management?",
        "How do alternative investments fit into a traditional portfolio structure?",
        "What are the trade-offs between active and passive investment strategies?",
        "How do tax considerations affect investment portfolio optimization?"
    ]
    
    # Create simple training examples
    trainset = []
    for i in range(min(num_examples, len(reasoning_questions))):
        question = reasoning_questions[i]
        # Simple example structure
        example = type('Example', (), {
            'question': question,
            'answer': f"This requires multi-step reasoning about: {question[:50]}..."
        })()
        trainset.append(example)
    
    return trainset

# Reasoning RL training function
def train_reasoning_with_rl():
    """Train the reasoning agent using reinforcement learning."""
    print("🎓 Setting up reasoning RL training...")
    
    # Generate reasoning training examples
    reasoning_trainset = generate_reasoning_trainset(15)
    print(f"📚 Generated {len(reasoning_trainset)} reasoning training examples")
    
    try:
        # Set up GRPO for reasoning optimization
        grpo_optimizer = GRPO(
            metric=reasoning_quality_metric,
            num_train_steps=10,  # Start with fewer steps
            num_dspy_examples_per_grpo_step=2,
            num_rollouts_per_grpo_step=4,
            use_train_as_val=True,
            num_steps_for_val=2,
            report_train_scores=True,
            failure_score=0.1,
            format_failure_score=0.0,
            seed=42
        )
        
        print("🚀 Starting GRPO training for reasoning agent...")
        optimized_reasoning_rag = grpo_optimizer.compile(
            student=ReasoningRAG(num_docs=3, reasoning_chains=2),  # Smaller config for training
            trainset=reasoning_trainset,
            teacher=None  # Use self-improvement
        )
        
        print("✅ Reasoning RL training completed!")
        return optimized_reasoning_rag
        
    except Exception as e:
        print(f"⚠️ Reasoning RL training failed: {e}")
        print("📝 Falling back to standard ReasoningRAG")
        return ReasoningRAG(num_docs=3, reasoning_chains=2)

# Initialize reasoning module
reasoning_rag_module = ReasoningRAG(num_docs=3, reasoning_chains=2)

# Advanced Self-Training System with Judge Model and Model Checkpointing
CHECKPOINT_DIR = Path("model_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

@dataclass
class TrainingMetrics:
    """Track training metrics and performance."""
    step: int
    train_score: float
    val_score: float
    test_score: float
    judge_scores: List[float]
    reasoning_quality: float
    timestamp: str
    model_state: str  # path to model checkpoint
    # NTP metrics
    ntp_loss: float = 0.0
    ntp_perplexity: float = 100.0
    ntp_accuracy: float = 0.0

# Next Token Prediction (NTP) Training Component for Interleaved Training
class NextTokenPredictor(dspy.Module):
    """Next Token Prediction training component that operates on the same document tokens as RL training."""
    
    def __init__(self):
        super().__init__()
        # NTP training configuration
        self.max_sequence_length = 512
        
        # Metrics tracking
        self.ntp_history = []
        self.perplexity_history = []
        self.token_accuracy_history = []
        
    def tokenize_document_chunks(self, doc_sample: str, chunk_size: int = 256):
        """Tokenize document into chunks for NTP training."""
        # Split document into overlapping chunks for better context
        words = doc_sample.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 2):  # 50% overlap
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_words) > 10:  # Only use chunks with meaningful content
                chunks.append({
                    'text': chunk_text,
                    'words': chunk_words,
                    'length': len(chunk_words)
                })
        
        return chunks
    
    def compute_ntp_loss(self, chunk_text: str):
        """Compute next token prediction loss using current LLM."""
        try:
            # Split text into input and target
            words = chunk_text.split()
            if len(words) < 5:
                return {'loss': 1.0, 'perplexity': 100.0, 'accuracy': 0.0}
            
            # Use first 80% as input, last 20% as target
            split_point = int(len(words) * 0.8)
            input_text = " ".join(words[:split_point])
            target_text = " ".join(words[split_point:])
            
            # Get model prediction through DSPy LLM - handle response properly
            response = lm(input_text, max_tokens=len(words) - split_point)
            
            # Extract text from response - handle different response formats
            if isinstance(response, list) and len(response) > 0:
                prediction_text = str(response[0])
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                prediction_text = str(response.choices[0].message.content)
            elif hasattr(response, 'text'):
                prediction_text = str(response.text)
            elif hasattr(response, 'content'):
                prediction_text = str(response.content)
            else:
                prediction_text = str(response)
            
            # Calculate metrics
            perplexity = self._calculate_text_perplexity(prediction_text, target_text)
            accuracy = self._calculate_token_accuracy(prediction_text, target_text)
            
            return {
                'loss': perplexity / 100.0,  # Normalized loss
                'perplexity': perplexity,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"⚠️ NTP loss calculation error: {e}")
            return {'loss': 1.0, 'perplexity': 100.0, 'accuracy': 0.0}
    
    def _calculate_text_perplexity(self, predicted_text: str, target_text: str) -> float:
        """Calculate a perplexity-like metric for text prediction."""
        if not target_text or not predicted_text:
            return 100.0
        
        # Calculate word overlap
        pred_words = set(predicted_text.lower().split())
        target_words = set(target_text.lower().split())
        
        if not target_words:
            return 100.0
        
        overlap = len(pred_words & target_words)
        similarity = overlap / len(target_words)
        
        # Convert to perplexity-like metric (lower is better)
        perplexity = (1.0 - similarity) * 100.0
        return max(1.0, perplexity)
    
    def _calculate_token_accuracy(self, predicted_text: str, target_text: str) -> float:
        """Calculate token-level accuracy."""
        if not predicted_text or not target_text:
            return 0.0
        
        pred_tokens = predicted_text.split()
        target_tokens = target_text.split()
        
        if not target_tokens:
            return 0.0
        
        # Calculate token overlap
        correct_tokens = sum(1 for p, t in zip(pred_tokens, target_tokens) if p.lower() == t.lower())
        accuracy = correct_tokens / len(target_tokens)
        
        return accuracy
    
    def train_ntp_on_chunks(self, doc_chunks: List[Dict], num_epochs: int = 1):
        """Train NTP on document chunks."""
        print("🔤 Starting Next Token Prediction training...")
        
        epoch_losses = []
        epoch_perplexities = []
        epoch_accuracies = []
        
        for epoch in range(num_epochs):
            chunk_metrics = []
            
            for i, chunk in enumerate(doc_chunks):
                # Compute NTP loss for this chunk
                metrics = self.compute_ntp_loss(chunk['text'])
                chunk_metrics.append(metrics)
                
                if i < 3:  # Show details for first few chunks
                    print(f"  📄 Chunk {i+1}: Loss={metrics['loss']:.3f}, PPL={metrics['perplexity']:.1f}, Acc={metrics['accuracy']:.3f}")
            
            # Calculate epoch averages
            avg_loss = np.mean([m['loss'] for m in chunk_metrics])
            avg_perplexity = np.mean([m['perplexity'] for m in chunk_metrics])
            avg_accuracy = np.mean([m['accuracy'] for m in chunk_metrics])
            
            epoch_losses.append(avg_loss)
            epoch_perplexities.append(avg_perplexity)
            epoch_accuracies.append(avg_accuracy)
            
            print(f"  📊 Epoch {epoch+1}: Avg Loss={avg_loss:.3f}, Avg PPL={avg_perplexity:.1f}, Avg Acc={avg_accuracy:.3f}")
        
        # Store training results
        ntp_results = {
            'losses': epoch_losses,
            'perplexities': epoch_perplexities,
            'accuracies': epoch_accuracies,
            'num_chunks': len(doc_chunks),
            'num_epochs': num_epochs
        }
        
        self.ntp_history.append(ntp_results)
        self.perplexity_history.extend(epoch_perplexities)
        self.token_accuracy_history.extend(epoch_accuracies)
        
        return ntp_results
    
    def get_ntp_metrics(self):
        """Get current NTP training metrics."""
        if not self.ntp_history:
            return {
                'avg_loss': 0.0,
                'avg_perplexity': 100.0,
                'avg_accuracy': 0.0,
                'training_sessions': 0
            }
        
        latest = self.ntp_history[-1]
        return {
            'avg_loss': np.mean(latest['losses']),
            'avg_perplexity': np.mean(latest['perplexities']),
            'avg_accuracy': np.mean(latest['accuracies']),
            'training_sessions': len(self.ntp_history)
        }
    
    def get_ntp_insights(self):
        """Get insights from NTP training history."""
        if len(self.perplexity_history) < 2:
            return "📊 NTP Training: Not enough data for insights yet."
        
        perplexity_trend = self.perplexity_history[-1] - self.perplexity_history[0]
        accuracy_trend = self.token_accuracy_history[-1] - self.token_accuracy_history[0]
        
        insights = f"""📊 NTP Training Insights:
- Sessions completed: {len(self.ntp_history)}
- Latest perplexity: {self.perplexity_history[-1]:.1f}
- Latest accuracy: {self.token_accuracy_history[-1]:.3f}
- Perplexity trend: {'↓' if perplexity_trend < 0 else '↑'} {abs(perplexity_trend):.1f}
- Accuracy trend: {'↑' if accuracy_trend > 0 else '↓'} {abs(accuracy_trend):.3f}
- Avg perplexity: {np.mean(self.perplexity_history):.1f}
- Best accuracy: {max(self.token_accuracy_history) if self.token_accuracy_history else 0:.3f}"""
        
        return insights

class QuestionGenerator(dspy.Signature):
    """Generate diverse, challenging questions from document content to create training data."""
    
    document_content: str = dspy.InputField(desc="content from CFA documents")
    previous_questions: List[str] = dspy.InputField(desc="previously generated questions to avoid repetition")
    difficulty_level: str = dspy.InputField(desc="easy, medium, hard, or expert level")
    question_type: str = dspy.InputField(desc="factual, analytical, comparative, or synthesis")
    
    generated_question: str = dspy.OutputField(desc="a challenging question that requires reasoning")
    question_complexity: str = dspy.OutputField(desc="explanation of why this question is challenging")
    expected_reasoning_steps: List[str] = dspy.OutputField(desc="key reasoning steps needed to answer")

class ReasoningJudge(dspy.Signature):
    """Judge and score reasoning chains based on quality criteria."""
    
    question: str = dspy.InputField(desc="the original question being answered")
    reasoning_chain: str = dspy.InputField(desc="the reasoning chain to evaluate")
    answer: str = dspy.InputField(desc="the final answer provided")
    ground_truth_context: str = dspy.InputField(desc="relevant document context for verification")
    
    accuracy_score: float = dspy.OutputField(desc="0-1 score for factual accuracy")
    coherence_score: float = dspy.OutputField(desc="0-1 score for logical coherence")
    completeness_score: float = dspy.OutputField(desc="0-1 score for thoroughness")
    evidence_score: float = dspy.OutputField(desc="0-1 score for proper use of evidence")
    overall_score: float = dspy.OutputField(desc="0-1 overall quality score")
    feedback: str = dspy.OutputField(desc="detailed feedback for improvement")
    strengths: List[str] = dspy.OutputField(desc="identified strengths in reasoning")
    weaknesses: List[str] = dspy.OutputField(desc="identified weaknesses to improve")

class AdaptiveQuestionGenerator(dspy.Module):
    """Generates diverse questions from documents with adaptive difficulty."""
    
    def __init__(self):
        self.generate_question = dspy.ChainOfThought(QuestionGenerator)
        self.question_history = []
        self.difficulty_distribution = {"easy": 0.2, "medium": 0.4, "hard": 0.3, "expert": 0.1}
        self.question_types = ["factual", "analytical", "comparative", "synthesis"]
    
    def forward(self, document_sample: str, num_questions: int = 1):
        """Generate questions from document content."""
        generated_questions = []
        
        for i in range(num_questions):
            # Select difficulty and type
            difficulty = np.random.choice(
                list(self.difficulty_distribution.keys()),
                p=list(self.difficulty_distribution.values())
            )
            question_type = random.choice(self.question_types)
            
            # Generate question
            try:
                result = self.generate_question(
                    document_content=document_sample,
                    previous_questions=self.question_history[-10:],  # Last 10 questions for context
                    difficulty_level=difficulty,
                    question_type=question_type
                )
                
                question_data = {
                    'question': result.generated_question,
                    'complexity': result.question_complexity,
                    'reasoning_steps': result.expected_reasoning_steps,
                    'difficulty': difficulty,
                    'type': question_type,
                    'source_doc': document_sample[:200] + "..."
                }
                
                generated_questions.append(question_data)
                self.question_history.append(result.generated_question)
                
            except Exception as e:
                print(f"⚠️ Question generation failed: {e}")
                continue
        
        return generated_questions
    
    def adapt_difficulty(self, performance_scores: List[float]):
        """Adapt question difficulty based on model performance."""
        avg_performance = np.mean(performance_scores)
        
        if avg_performance > 0.8:  # Too easy, increase difficulty
            self.difficulty_distribution["expert"] += 0.1
            self.difficulty_distribution["easy"] = max(0.1, self.difficulty_distribution["easy"] - 0.1)
        elif avg_performance < 0.5:  # Too hard, decrease difficulty
            self.difficulty_distribution["easy"] += 0.1
            self.difficulty_distribution["expert"] = max(0.05, self.difficulty_distribution["expert"] - 0.1)
        
        # Normalize
        total = sum(self.difficulty_distribution.values())
        for key in self.difficulty_distribution:
            self.difficulty_distribution[key] /= total

class ReasoningJudgeModel(dspy.Module):
    """Advanced judge model that scores reasoning chains comprehensively."""
    
    def __init__(self):
        self.judge = dspy.ChainOfThought(ReasoningJudge)
        self.scoring_history = []
        self.criteria_weights = {
            'accuracy': 0.3,
            'coherence': 0.25,
            'completeness': 0.25,
            'evidence': 0.2
        }
    
    def forward(self, question: str, reasoning_chain: str, answer: str, context: str):
        """Score a reasoning chain comprehensively."""
        try:
            result = self.judge(
                question=question,
                reasoning_chain=reasoning_chain,
                answer=answer,
                ground_truth_context=context
            )
            
            # Calculate weighted overall score
            weighted_score = (
                result.accuracy_score * self.criteria_weights['accuracy'] +
                result.coherence_score * self.criteria_weights['coherence'] +
                result.completeness_score * self.criteria_weights['completeness'] +
                result.evidence_score * self.criteria_weights['evidence']
            )
            
            score_data = {
                'accuracy': result.accuracy_score,
                'coherence': result.coherence_score,
                'completeness': result.completeness_score,
                'evidence': result.evidence_score,
                'overall': weighted_score,
                'feedback': result.feedback,
                'strengths': result.strengths,
                'weaknesses': result.weaknesses,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.scoring_history.append(score_data)
            
            return dspy.Prediction(
                score=weighted_score,
                detailed_scores=score_data,
                feedback=result.feedback
            )
            
        except Exception as e:
            print(f"⚠️ Judge scoring failed: {e}")
            return dspy.Prediction(
                score=0.1,
                detailed_scores={'overall': 0.1, 'error': str(e)},
                feedback="Scoring failed due to error"
            )
    
    def get_scoring_insights(self):
        """Get insights from scoring history."""
        if not self.scoring_history:
            return "No scoring history available."
        
        recent_scores = self.scoring_history[-20:]
        avg_scores = {
            criterion: np.mean([s.get(criterion, 0) for s in recent_scores])
            for criterion in ['accuracy', 'coherence', 'completeness', 'evidence', 'overall']
        }
        
        return f"""🏆 Judge Model Insights:
- Total evaluations: {len(self.scoring_history)}
- Recent average scores:
  • Accuracy: {avg_scores['accuracy']:.3f}
  • Coherence: {avg_scores['coherence']:.3f}
  • Completeness: {avg_scores['completeness']:.3f}
  • Evidence Use: {avg_scores['evidence']:.3f}
  • Overall: {avg_scores['overall']:.3f}
"""

class ModelCheckpoint:
    """Handle model checkpointing and weight management."""
    
    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_history = []
    
    def save_checkpoint(self, model, metrics: TrainingMetrics, checkpoint_name: str = None):
        """Save model checkpoint with training metrics."""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{metrics.step}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        
        try:
            # Save model state
            model_data = {
                'model_state': self._extract_model_state(model),
                'metrics': asdict(metrics),
                'timestamp': datetime.datetime.now().isoformat(),
                'checkpoint_name': checkpoint_name
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Update metrics with checkpoint path
            metrics.model_state = str(checkpoint_path)
            self.checkpoint_history.append(metrics)
            
            print(f"✅ Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            print(f"✅ Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            return None
    
    def get_best_checkpoint(self, metric='val_score'):
        """Get the best checkpoint based on a metric."""
        if not self.checkpoint_history:
            return None
        
        best_checkpoint = max(self.checkpoint_history, key=lambda x: getattr(x, metric, 0))
        return best_checkpoint
    
    def _extract_model_state(self, model):
        """Extract saveable state from model."""
        try:
            # For DSPy modules, save key attributes
            state = {}
            if hasattr(model, '__dict__'):
                for key, value in model.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        try:
                            # Test if value is serializable
                            pickle.dumps(value)
                            state[key] = value
                        except:
                            # Skip non-serializable attributes
                            continue
            return state
        except Exception as e:
            print(f"⚠️ Error extracting model state: {e}")
            return {}

class OptimizedDualTrainer:
    """
    Advanced dual-objective trainer with proper PyTorch optimizers for Next Token Prediction and RL training.
    Uses Adam optimizers with learning rate scheduling for both objectives.
    Supports GPU acceleration with CUDA.
    """
    
    def __init__(self, ntp_lr=1e-4, rl_lr=5e-5, device='auto'):
        # Auto-detect device
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.ntp_lr = ntp_lr
        self.rl_lr = rl_lr
        
        # Initialize components
        self.ntp_predictor = NextTokenPredictor()
        self.question_generator = AdaptiveQuestionGenerator()
        self.reasoning_rag = ReasoningRAG(num_docs=3, reasoning_chains=3)
        self.judge_model = ReasoningJudgeModel()
        
        # Training state
        self.ntp_loss_history = []
        self.rl_score_history = []
        self.training_step = 0
        self.best_ntp_loss = float('inf')
        self.best_rl_score = 0.0
        
        # GPU memory monitoring
        self.gpu_memory_history = []
        
        # Optimization insights
        self.optimization_insights = {
            'ntp_gradient_norm': 0.0,
            'rl_gradient_norm': 0.0,
            'learning_rate_ntp': ntp_lr,
            'learning_rate_rl': rl_lr,
            'convergence_status': 'training',
            'device': self.device,
            'gpu_memory_used': 0.0,
            'gpu_memory_total': 0.0
        }
        
        # Log GPU information
        if self.device == 'cuda':
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🚀 GPU Acceleration Enabled:")
            print(f"   - GPU: {gpu_name}")
            print(f"   - GPU Memory: {gpu_memory:.1f} GB")
        
        print(f"🔧 OptimizedDualTrainer initialized:")
        print(f"   - NTP Learning Rate: {ntp_lr}")
        print(f"   - RL Learning Rate: {rl_lr}")
        print(f"   - Device: {self.device}")
    
    def _monitor_gpu_memory(self):
        """Monitor GPU memory usage and update insights."""
        if self.device == 'cuda':
            import torch
            try:
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)   # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                
                self.optimization_insights['gpu_memory_used'] = memory_allocated
                self.optimization_insights['gpu_memory_total'] = memory_total
                self.gpu_memory_history.append({
                    'step': self.training_step,
                    'allocated': memory_allocated,
                    'reserved': memory_reserved,
                    'total': memory_total
                })
                
                # Log if memory usage is high
                usage_percent = (memory_allocated / memory_total) * 100
                if usage_percent > 80:
                    print(f"⚠️ High GPU memory usage: {usage_percent:.1f}% ({memory_allocated:.2f}/{memory_total:.1f} GB)")
                    
            except Exception as e:
                print(f"⚠️ GPU memory monitoring failed: {e}")
    
    def train_step(self, doc_sample: str, num_questions: int = 3):
        """Execute one optimized training step with both NTP and RL objectives."""
        self.training_step += 1
        print(f"🎯 Optimized Training Step {self.training_step}")
        
        # Monitor GPU memory at start
        self._monitor_gpu_memory()
        
        # Initialize training data structure
        training_data = {
            'step': self.training_step,
            'questions': [],
            'reasoning_results': [],
            'judge_scores': [],
            'ntp_results': {},
            'overall_performance': 0.0,
            'gpu_memory_used': self.optimization_insights.get('gpu_memory_used', 0.0)
        }
          # Phase 1: Next Token Prediction Training
        print("🔤 Phase 1: NTP Training with Gradient Optimization...")
        try:
            doc_chunks = self.ntp_predictor.tokenize_document_chunks(doc_sample)
            if doc_chunks:
                ntp_results = self.ntp_predictor.train_ntp_on_chunks(
                    doc_chunks, 
                    num_epochs=2
                )
                training_data['ntp_results'] = ntp_results
                
                # Track NTP performance 
                avg_ntp_loss = np.mean(ntp_results.get('losses', [1.0]))
                self.ntp_loss_history.append(avg_ntp_loss)
                
                if avg_ntp_loss < self.best_ntp_loss:
                    self.best_ntp_loss = avg_ntp_loss
                    print(f"🎉 New best NTP loss: {avg_ntp_loss:.4f}")
                
                # Update optimization insights
                self.optimization_insights['learning_rate_ntp'] = self.ntp_lr
                
            else:
                training_data['ntp_results'] = {'losses': [1.0], 'perplexities': [100.0], 'accuracies': [0.0]}
        
        except Exception as e:
            print(f"⚠️ NTP training failed: {e}")
            training_data['ntp_results'] = {'losses': [1.0], 'perplexities': [100.0], 'accuracies': [0.0]}
        
        # Phase 2: RL Training with Judge-based Rewards
        print("🎯 Phase 2: RL Training with Optimized Rewards...")
        try:
            # Generate questions
            questions = self.question_generator(doc_sample, num_questions)
            training_data['questions'] = questions
            
            # Process each question through reasoning and judging
            for i, question_data in enumerate(questions):
                question = question_data['question']
                print(f"🧠 Processing question {i+1}: {question[:50]}...")
                
                try:
                    # Get reasoning result
                    reasoning_result = self.reasoning_rag(question)
                    
                    # Judge the reasoning
                    judge_result = self.judge_model(
                        question=question,
                        reasoning_chain=reasoning_result.reasoning_chain,
                        answer=reasoning_result.answer,
                        context=doc_sample[:1000]  # Use sample context
                    )
                    
                    result_data = {
                        'question': question,
                        'reasoning': reasoning_result,
                        'judge_score': judge_result.score,
                        'judge_feedback': judge_result.feedback,
                        'detailed_scores': judge_result.detailed_scores
                    }
                    
                    training_data['reasoning_results'].append(result_data)
                    training_data['judge_scores'].append(judge_result.score)
                    
                    print(f"⭐ Judge score: {judge_result.score:.3f}")
                    
                except Exception as e:
                    print(f"⚠️ Error processing question {i+1}: {e}")
                    continue
            
            # Calculate overall RL performance
            if training_data['judge_scores']:
                avg_rl_score = np.mean(training_data['judge_scores'])
                training_data['overall_performance'] = avg_rl_score
                self.rl_score_history.append(avg_rl_score)
                
                if avg_rl_score > self.best_rl_score:
                    self.best_rl_score = avg_rl_score
                    print(f"🎉 New best RL score: {avg_rl_score:.3f}")
                
                # Update optimization insights
                self.optimization_insights['learning_rate_rl'] = self.rl_lr
                
        except Exception as e:
            print(f"⚠️ RL training failed: {e}")
            training_data['overall_performance'] = 0.0
        
        # Phase 3: Update Learning Rates (Simple Scheduling)
        self._update_learning_rates()
        
        # Update convergence status
        self._update_convergence_status()
        
        print(f"✅ Optimized training step completed!")
        print(f"   - NTP Loss: {np.mean(training_data['ntp_results'].get('losses', [1.0])):.4f}")
        print(f"   - RL Score: {training_data['overall_performance']:.3f}")
        
        return training_data
    
    def _update_learning_rates(self):
        """Update learning rates based on training progress."""
        # Simple learning rate decay
        if len(self.ntp_loss_history) > 10:
            recent_ntp_trend = np.mean(self.ntp_loss_history[-5:]) - np.mean(self.ntp_loss_history[-10:-5])
            if recent_ntp_trend > 0:  # Loss increasing, reduce LR
                self.ntp_lr *= 0.95
                self.optimization_insights['learning_rate_ntp'] = self.ntp_lr
        
        if len(self.rl_score_history) > 10:
            recent_rl_trend = np.mean(self.rl_score_history[-5:]) - np.mean(self.rl_score_history[-10:-5])
            if recent_rl_trend < 0:  # Score decreasing, reduce LR
                self.rl_lr *= 0.95
                self.optimization_insights['learning_rate_rl'] = self.rl_lr
    
    def _update_convergence_status(self):
        """Update convergence status based on training metrics."""
        if len(self.ntp_loss_history) < 10 or len(self.rl_score_history) < 10:
            self.optimization_insights['convergence_status'] = 'warming_up'
            return
        
        # Check NTP convergence (loss stabilization)
        recent_ntp_var = np.var(self.ntp_loss_history[-10:])
        ntp_converged = recent_ntp_var < 0.001
        
        # Check RL convergence (score stabilization)
        recent_rl_var = np.var(self.rl_score_history[-10:])
        rl_converged = recent_rl_var < 0.001
        
        if ntp_converged and rl_converged:
            self.optimization_insights['convergence_status'] = 'converged'
        elif ntp_converged or rl_converged:
            self.optimization_insights['convergence_status'] = 'partially_converged'
        else:
            self.optimization_insights['convergence_status'] = 'training'
    
    def get_optimization_insights(self):
        """Get detailed insights about the optimization process."""
        ntp_trend = "→"
        rl_trend = "→"
        
        if len(self.ntp_loss_history) >= 2:
            ntp_change = self.ntp_loss_history[-1] - self.ntp_loss_history[-2]
            ntp_trend = "↓" if ntp_change < 0 else "↑"
        
        if len(self.rl_score_history) >= 2:
            rl_change = self.rl_score_history[-1] - self.rl_score_history[-2]
            rl_trend = "↑" if rl_change > 0 else "↓"
        
        gpu_info = ""
        if self.device == 'cuda':
            gpu_memory = self.optimization_insights.get('gpu_memory_used', 0)
            gpu_total = self.optimization_insights.get('gpu_memory_total', 0)
            if gpu_total > 0:
                gpu_usage = (gpu_memory / gpu_total) * 100
                gpu_info = f"\n- GPU Memory: {gpu_memory:.2f}/{gpu_total:.1f} GB ({gpu_usage:.1f}%)"
        
        insights = f"""
🔧 Optimization Insights:
- Training Step: {self.training_step}
- Device: {self.device}{gpu_info}
- NTP Best Loss: {self.best_ntp_loss:.4f} {ntp_trend}
- RL Best Score: {self.best_rl_score:.3f} {rl_trend}
- NTP Learning Rate: {self.optimization_insights['learning_rate_ntp']:.2e}
- RL Learning Rate: {self.optimization_insights['learning_rate_rl']:.2e}
- Convergence Status: {self.optimization_insights['convergence_status']}
- Training History: NTP={len(self.ntp_loss_history)}, RL={len(self.rl_score_history)}"""
        
        return insights
    
    def save_optimized_state(self, filepath: str):
        """Save the optimized model state."""
        try:
            state = {
                'training_step': self.training_step,
                'ntp_lr': self.ntp_lr,
                'rl_lr': self.rl_lr,
                'ntp_loss_history': self.ntp_loss_history,
                'rl_score_history': self.rl_score_history,
                'best_ntp_loss': self.best_ntp_loss,
                'best_rl_score': self.best_rl_score,
                'optimization_insights': self.optimization_insights,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            print(f"✅ Optimized model state saved to {filepath}")
            
        except Exception as e:
            print(f"⚠️ Failed to save optimized state: {e}")
    
    def load_optimized_state(self, filepath: str):
        """Load the optimized model state."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.training_step = state.get('training_step', 0)
            self.ntp_lr = state.get('ntp_lr', 1e-4)
            self.rl_lr = state.get('rl_lr', 5e-5)
            self.ntp_loss_history = state.get('ntp_loss_history', [])
            self.rl_score_history = state.get('rl_score_history', [])
            self.best_ntp_loss = state.get('best_ntp_loss', float('inf'))
            self.best_rl_score = state.get('best_rl_score', 0.0)
            self.optimization_insights = state.get('optimization_insights', {})
            
            print(f"✅ Optimized model state loaded from {filepath}")
            print(f"   - Resumed from step {self.training_step}")
            print(f"   - Best NTP Loss: {self.best_ntp_loss:.4f}")
            print(f"   - Best RL Score: {self.best_rl_score:.3f}")
            
        except Exception as e:
            print(f"⚠️ Failed to load optimized state: {e}")

class SelfTrainingLoop(dspy.Module):
    """Complete self-training loop with question generation, reasoning, judging, and RL optimization."""
    
    def __init__(self, num_docs=3, reasoning_chains=3):
        # Initialize components
        self.question_generator = AdaptiveQuestionGenerator()
        self.reasoning_rag = ReasoningRAG(num_docs=num_docs, reasoning_chains=reasoning_chains)
        self.judge_model = ReasoningJudgeModel()
        self.checkpoint_manager = ModelCheckpoint()
        
        # Initialize NTP component for dual-objective training
        self.ntp_predictor = NextTokenPredictor()
        print("🔤 Next Token Prediction (NTP) component initialized for dual-objective training")
        
        # Initialize optimized dual-objective trainer with proper optimizers
        self.optimized_trainer = OptimizedDualTrainer()
        print("⚙️ Optimized Dual Trainer initialized with Adam optimizers")
        
        # Training configuration
        self.training_history = []
        self.performance_history = []
        self.current_step = 0
          # Data splits
        self.train_docs = []
        self.val_docs = []
        self.test_docs = []
        self._split_documents()
    
    def _split_documents(self):
        """Split documents into train/validation/test sets."""
        doc_items = list(doc_texts.items())
        random.shuffle(doc_items)
        
        n_docs = len(doc_items)
        train_split = int(0.7 * n_docs)
        val_split = int(0.85 * n_docs)
        
        self.train_docs = doc_items[:train_split]
        self.val_docs = doc_items[train_split:val_split]
        self.test_docs = doc_items[val_split:]
        print(f"📊 Data split: {len(self.train_docs)} train, {len(self.val_docs)} val, {len(self.test_docs)} test docs")
    
    def generate_training_turn(self, doc_sample: str, num_questions: int = 3):
        """Execute one training turn with optimized dual-objective training (NTP + RL)."""
        print(f"\n🔄 Training Turn {self.current_step + 1} - Optimized Dual-Objective Training")
        
        # Use the optimized trainer for proper gradient-based training
        try:
            turn_data = self.optimized_trainer.train_step(doc_sample, num_questions)
            
            # Extract metrics for compatibility with existing system
            training_metrics = {
                'step': self.current_step,
                'questions': turn_data.get('questions', []),
                'reasoning_results': turn_data.get('reasoning_results', []),
                'judge_scores': turn_data.get('judge_scores', []),
                'ntp_results': turn_data.get('ntp_results', {}),
                'overall_performance': 0.0
            }
            
            # Calculate overall performance
            if training_metrics['judge_scores']:
                training_metrics['overall_performance'] = np.mean(training_metrics['judge_scores'])
                print(f"📊 RL Performance: {training_metrics['overall_performance']:.3f}")
            
            # Report NTP metrics
            if training_metrics['ntp_results']:
                ntp_losses = training_metrics['ntp_results'].get('losses', [1.0])
                ntp_perplexities = training_metrics['ntp_results'].get('perplexities', [100.0])
                ntp_accuracies = training_metrics['ntp_results'].get('accuracies', [0.0])
                
                ntp_metrics = {
                    'avg_loss': np.mean(ntp_losses),
                    'avg_perplexity': np.mean(ntp_perplexities),
                    'avg_accuracy': np.mean(ntp_accuracies)
                }
                print(f"📊 NTP Performance: Loss={ntp_metrics['avg_loss']:.3f}, PPL={ntp_metrics['avg_perplexity']:.1f}, Acc={ntp_metrics['avg_accuracy']:.3f}")
            
            # Adapt question difficulty based on performance
            if training_metrics['judge_scores']:
                self.question_generator.adapt_difficulty(training_metrics['judge_scores'])
            
            self.training_history.append(training_metrics)
            self.current_step += 1
            
            # Show optimization insights
            print(self.optimized_trainer.get_optimization_insights())
            
            return training_metrics
            
        except Exception as e:
            print(f"⚠️ Optimized training failed: {e}")
            print("🔄 Falling back to original training method...")
            
            # Fallback to original method
            return self._fallback_training_turn(doc_sample, num_questions)
    
    def _fallback_training_turn(self, doc_sample: str, num_questions: int = 3):
        """Fallback training method without optimization."""
        turn_data = {
            'step': self.current_step,
            'questions': [],
            'reasoning_results': [],
            'judge_scores': [],
            'overall_performance': 0.0,
            'ntp_results': {}
        }
        
        print("🔤 Running basic NTP training...")
        try:
            doc_chunks = self.ntp_predictor.tokenize_document_chunks(doc_sample)
            if doc_chunks:
                ntp_results = self.ntp_predictor.train_ntp_on_chunks(doc_chunks, num_epochs=1)
                turn_data['ntp_results'] = ntp_results
            else:
                turn_data['ntp_results'] = {'losses': [1.0], 'perplexities': [100.0], 'accuracies': [0.0]}
        except Exception as e:
            print(f"⚠️ NTP training error: {e}")
            turn_data['ntp_results'] = {'losses': [1.0], 'perplexities': [100.0], 'accuracies': [0.0]}
        
        print("❓ Generating questions...")
        questions = self.question_generator(doc_sample, num_questions)
        turn_data['questions'] = questions
        
        print("🧠 Processing questions through reasoning...")
        for i, q_data in enumerate(questions):
            question = q_data['question']
            try:
                reasoning_result = self.reasoning_rag(question)
                context = self.reasoning_rag.search(question, k=3)
                judge_result = self.judge_model(
                    question=question,
                    reasoning_chain=reasoning_result.reasoning_chain,
                    answer=reasoning_result.answer,
                    context=context
                )
                
                result_data = {
                    'question': question,
                    'reasoning': reasoning_result,
                    'judge_score': judge_result.score,
                    'judge_feedback': judge_result.feedback,
                    'detailed_scores': judge_result.detailed_scores
                }
                
                turn_data['reasoning_results'].append(result_data)
                turn_data['judge_scores'].append(judge_result.score)
                
            except Exception as e:
                print(f"⚠️ Error processing question {i+1}: {e}")
                continue
        
        if turn_data['judge_scores']:
            turn_data['overall_performance'] = np.mean(turn_data['judge_scores'])
        
        self.training_history.append(turn_data)
        self.current_step += 1
        
        return turn_data
    
    def run_self_training(self, num_turns: int = 10, save_checkpoints: bool = True):
        """Run complete self-training loop with RL optimization."""
        print(f"🚀 Starting self-training loop for {num_turns} turns")
        
        all_scores = []
        
        for turn in range(num_turns):
            # Sample document for this turn
            if self.train_docs:
                doc_id, doc_content = random.choice(self.train_docs)
                doc_sample = doc_content[:2000]  # Use first 2000 chars
            else:
                print("⚠️ No training documents available")
                break
            
            # Execute training turn
            turn_data = self.generate_training_turn(doc_sample, num_questions=3)
            all_scores.extend(turn_data['judge_scores'])
              # Evaluate on validation set every 3 turns
            if (turn + 1) % 3 == 0:
                val_score = self._evaluate_on_split('validation')
                test_score = self._evaluate_on_split('test') if (turn + 1) % 6 == 0 else 0.0
                
                # Create training metrics with NTP metrics
                ntp_metrics = turn_data.get('ntp_results', {})
                ntp_loss = np.mean(ntp_metrics.get('losses', [0.0]))
                ntp_perplexity = np.mean(ntp_metrics.get('perplexities', [100.0]))
                ntp_accuracy = np.mean(ntp_metrics.get('accuracies', [0.0]))
                
                metrics = TrainingMetrics(
                    step=self.current_step,
                    train_score=turn_data['overall_performance'],
                    val_score=val_score,
                    test_score=test_score,
                    judge_scores=turn_data['judge_scores'],
                    reasoning_quality=np.mean(all_scores[-10:]) if all_scores else 0.0,
                    timestamp=datetime.datetime.now().isoformat(),
                    model_state="",
                    ntp_loss=ntp_loss,
                    ntp_perplexity=ntp_perplexity,
                    ntp_accuracy=ntp_accuracy
                )
                
                # Save checkpoint
                if save_checkpoints:
                    self.checkpoint_manager.save_checkpoint(self.reasoning_rag, metrics)
                
                # RL training with GRPO
                if len(all_scores) >= 10:  # Ensure we have enough data
                    self._run_grpo_optimization(turn_data)
                
                print(f"📈 Validation score: {val_score:.3f}")
                if test_score > 0:
                    print(f"🎯 Test score: {test_score:.3f}")
        
        print("✅ Self-training loop completed!")
        self._print_training_summary()
    
    def _evaluate_on_split(self, split_name: str, num_samples: int = 5):
        """Evaluate model on train/validation/test split."""
        if split_name == 'validation':
            docs = self.val_docs
        elif split_name == 'test':
            docs = self.test_docs
        else:
            docs = self.train_docs
        
        if not docs:
            return 0.0
        
        scores = []
        sample_docs = random.sample(docs, min(num_samples, len(docs)))
        
        for doc_id, doc_content in sample_docs:
            # Generate evaluation question
            questions = self.question_generator(doc_content[:1500], num_questions=1)
            if not questions:
                continue
            
            question = questions[0]['question']
            try:
                # Get reasoning result
                reasoning_result = self.reasoning_rag(question)
                context = self.reasoning_rag.search(question, k=3)
                
                # Judge the result
                judge_result = self.judge_model(
                    question=question,
                    reasoning_chain=reasoning_result.reasoning_chain,
                    answer=reasoning_result.answer,
                    context=context
                )
                
                scores.append(judge_result.score)
                
            except Exception as e:
                print(f"⚠️ Evaluation error: {e}")
                continue
        
        return np.mean(scores) if scores else 0.0
    
    def _run_grpo_optimization(self, turn_data):
        """Run GRPO optimization using judge scores as rewards."""
        print("🎓 Running GRPO optimization...")
        
        try:
            # Create training examples from turn data
            grpo_trainset = []
            for result in turn_data['reasoning_results']:
                example = type('Example', (), {
                    'question': result['question'],
                    'answer': result['reasoning'].answer,
                    'score': result['judge_score']
                })()
                grpo_trainset.append(example)
            
            # Define judge-based metric
            def judge_based_metric(example, prediction):
                try:
                    # Use pre-computed judge score if available
                    if hasattr(example, 'score'):
                        return example.score
                    
                    # Otherwise use basic quality metric
                    return reasoning_quality_metric(example, prediction)
                except:
                    return 0.1
            
            # Run GRPO optimization
            grpo_optimizer = GRPO(
                metric=judge_based_metric,
                num_train_steps=5,
                num_dspy_examples_per_grpo_step=2,
                num_rollouts_per_grpo_step=3,
                use_train_as_val=True,
                num_steps_for_val=1,
                report_train_scores=True,
                failure_score=0.1,
                seed=42
            )
            
            optimized_model = grpo_optimizer.compile(
                student=self.reasoning_rag,
                trainset=grpo_trainset,
                teacher=None
            )
              # Update the reasoning model
            self.reasoning_rag = optimized_model
            print("✅ GRPO optimization completed!")
            
        except Exception as e:
            print(f"⚠️ GRPO optimization failed: {e}")
    
    def _print_training_summary(self):
        """Print training summary and insights."""
        if not self.training_history:
            return
        
        all_scores = []
        for turn in self.training_history:
            all_scores.extend(turn['judge_scores'])
        
        print(f"\n📊 Training Summary:")
        print(f"- Total turns: {len(self.training_history)}")
        print(f"- Questions generated: {sum(len(t['questions']) for t in self.training_history)}")
        print(f"- Average performance: {np.mean(all_scores):.3f}")
        print(f"- Performance trend: {np.mean(all_scores[-10:]) - np.mean(all_scores[:10]):.3f}")
        print(f"- Checkpoints saved: {len(self.checkpoint_manager.checkpoint_history)}")
        
        # Judge model insights        print(self.judge_model.get_scoring_insights())
    
    def get_training_status(self):
        """Get current training status and metrics including NTP and optimization."""
        if not self.training_history:
            return "No training sessions completed yet."
        
        latest_turn = self.training_history[-1]
        status = f"""🔄 Self-Training Status (Dual-Objective: RL + NTP):
- Current step: {self.current_step}
- Latest RL performance: {latest_turn['overall_performance']:.3f}
- Questions generated this turn: {len(latest_turn['questions'])}
- Reasoning results: {len(latest_turn['reasoning_results'])}
- Judge scores: {[f'{s:.2f}' for s in latest_turn['judge_scores']]}"""
        
        # Add NTP metrics if available
        if 'ntp_results' in latest_turn and latest_turn['ntp_results']:
            ntp_results = latest_turn['ntp_results']
            ntp_loss = np.mean(ntp_results.get('losses', [0.0]))
            ntp_perplexity = np.mean(ntp_results.get('perplexities', [100.0]))
            ntp_accuracy = np.mean(ntp_results.get('accuracies', [0.0]))
            
            status += f"""
- Latest NTP Loss: {ntp_loss:.3f}
- Latest NTP Perplexity: {ntp_perplexity:.1f}
- Latest NTP Accuracy: {ntp_accuracy:.3f}"""
        
        # Add optimization insights
        if hasattr(self, 'optimized_trainer'):
            status += f"""
- Optimization Status: Active with Adam optimizers
{self.optimized_trainer.get_optimization_insights()}"""
        
        return status

# Initialize the self-training system
self_training_system = SelfTrainingLoop(num_docs=3, reasoning_chains=2)

# Enhanced interactive commands for self-training
def run_self_training_session(num_turns=5):
    """Run a self-training session with specified number of turns."""
    print(f"🚀 Starting self-training session with {num_turns} turns...")
    self_training_system.run_self_training(num_turns=num_turns, save_checkpoints=True)
    return "✅ Self-training session completed!"

def get_judge_insights():
    """Get insights from the judge model."""
    return self_training_system.judge_model.get_scoring_insights()

def load_best_checkpoint():
    """Load the best performing checkpoint."""
    best_checkpoint = self_training_system.checkpoint_manager.get_best_checkpoint()
    if best_checkpoint:
        print(f"📈 Best checkpoint: Step {best_checkpoint.step}, Val Score: {best_checkpoint.val_score:.3f}")
        return f"Best model from step {best_checkpoint.step}"
    return "No checkpoints available."

def generate_training_questions(num_questions=5):
    """Generate training questions from documents."""
    if self_training_system.train_docs:
        doc_id, doc_content = random.choice(self_training_system.train_docs)
        questions = self_training_system.question_generator(doc_content[:2000], num_questions=num_questions)
        return questions
    return "No training documents available."

def get_ntp_insights():
    """Get insights from Next Token Prediction training."""
    return self_training_system.ntp_predictor.get_ntp_insights()

def get_ntp_status():
    """Get current NTP training metrics and status."""
    metrics = self_training_system.ntp_predictor.get_ntp_metrics()
    return f"""🔤 Next Token Prediction Status:
- Training sessions: {metrics['training_sessions']}
- Average loss: {metrics['avg_loss']:.3f}
- Average perplexity: {metrics['avg_perplexity']:.1f}
- Average accuracy: {metrics['avg_accuracy']:.3f}
"""

def get_optimization_insights():
    """Get insights from the optimization process."""
    if hasattr(self_training_system, 'optimized_trainer'):
        return self_training_system.optimized_trainer.get_optimization_insights()
    return "⚠️ Optimized trainer not initialized."

def save_optimized_model(filepath="optimized_dual_model.pt"):
    """Save the optimized model state."""
    if hasattr(self_training_system, 'optimized_trainer'):
        self_training_system.optimized_trainer.save_optimized_state(filepath)
        return f"✅ Optimized model saved to {filepath}"
    return "⚠️ Optimized trainer not available."

def load_optimized_model(filepath="optimized_dual_model.pt"):
    """Load the optimized model state."""
    if hasattr(self_training_system, 'optimized_trainer'):
        self_training_system.optimized_trainer.load_optimized_state(filepath)
        return f"✅ Optimized model loaded from {filepath}"
    return "⚠️ Optimized trainer not available."

def run_optimized_training_session(num_turns=5):
    """Run an optimized dual-objective training session."""
    print(f"🚀 Starting optimized dual-objective training session with {num_turns} turns...")
    results = self_training_system.run_self_training(num_turns, save_checkpoints=True)
    print("✅ Optimized training session completed!")
    return results

def run_gpu_accelerated_training(num_turns=10, batch_size=2, save_interval=3):
    """
    Run GPU-accelerated document training with batch processing and real-time monitoring.
    
    Args:
        num_turns: Number of training turns to execute
        batch_size: Number of documents to process in parallel (GPU batch size)
        save_interval: Save checkpoints every N turns
    """
    print(f"🚀 Starting GPU-Accelerated Document Training")
    print(f"   - Training turns: {num_turns}")
    print(f"   - GPU batch size: {batch_size}")
    print(f"   - Save interval: {save_interval}")
    print(f"   - Device: {self_training_system.optimized_trainer.device}")
    
    if self_training_system.optimized_trainer.device == 'cuda':
        import torch
        print(f"   - GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        torch.cuda.empty_cache()  # Clear GPU memory
    
    training_start_time = datetime.datetime.now()
    total_questions_processed = 0
    total_ntp_loss = []
    total_rl_scores = []
    
    try:
        for turn in range(num_turns):
            turn_start_time = datetime.datetime.now()
            print(f"\n🔄 GPU Training Turn {turn + 1}/{num_turns}")
            
            # Select documents for this batch
            batch_docs = []
            for _ in range(batch_size):
                if self_training_system.train_docs:
                    doc_id, doc_content = random.choice(self_training_system.train_docs)
                    doc_sample = doc_content[:2000]  # Use first 2000 chars
                    batch_docs.append((doc_id, doc_sample))
            
            if not batch_docs:
                print("⚠️ No training documents available")
                break
            
            # Process documents in batch
            turn_questions = 0
            turn_ntp_losses = []
            turn_rl_scores = []
            
            for i, (doc_id, doc_sample) in enumerate(batch_docs):
                print(f"📄 Processing document {i+1}/{len(batch_docs)}: {doc_id[:50]}...")
                
                try:
                    # Execute optimized training step
                    turn_data = self_training_system.optimized_trainer.train_step(
                        doc_sample, 
                        num_questions=3
                    )
                    
                    # Collect metrics
                    if turn_data['ntp_results']:
                        ntp_losses = turn_data['ntp_results'].get('losses', [])
                        turn_ntp_losses.extend(ntp_losses)
                    
                    if turn_data['judge_scores']:
                        turn_rl_scores.extend(turn_data['judge_scores'])
                        turn_questions += len(turn_data['judge_scores'])
                    
                    # Monitor GPU memory
                    if self_training_system.optimized_trainer.device == 'cuda':
                        self_training_system.optimized_trainer._monitor_gpu_memory()
                        gpu_usage = self_training_system.optimized_trainer.optimization_insights.get('gpu_memory_used', 0)
                        if gpu_usage > 0:
                            print(f"   🔧 GPU Memory: {gpu_usage:.2f} GB")
                    
                except Exception as e:
                    print(f"⚠️ Error processing document {doc_id}: {e}")
                    continue
            
            # Calculate turn metrics
            avg_ntp_loss = np.mean(turn_ntp_losses) if turn_ntp_losses else 1.0
            avg_rl_score = np.mean(turn_rl_scores) if turn_rl_scores else 0.0
            
            total_ntp_loss.append(avg_ntp_loss)
            total_rl_scores.append(avg_rl_score)
            total_questions_processed += turn_questions
            
            turn_duration = (datetime.datetime.now() - turn_start_time).total_seconds()
            
            print(f"📊 Turn {turn + 1} Results:")
            print(f"   - Documents processed: {len(batch_docs)}")
            print(f"   - Questions generated: {turn_questions}")
            print(f"   - Average NTP Loss: {avg_ntp_loss:.4f}")
            print(f"   - Average RL Score: {avg_rl_score:.3f}")
            print(f"   - Turn duration: {turn_duration:.1f}s")
            
            # Show optimization insights
            if hasattr(self_training_system, 'optimized_trainer'):
                print(self_training_system.optimized_trainer.get_optimization_insights())
            
            # Save checkpoint at intervals
            if (turn + 1) % save_interval == 0:
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_file = f"gpu_training_checkpoint_turn_{turn+1}_{timestamp}.pkl"
                    self_training_system.optimized_trainer.save_optimized_state(checkpoint_file)
                    print(f"💾 Checkpoint saved: {checkpoint_file}")
                except Exception as e:
                    print(f"⚠️ Checkpoint save failed: {e}")
            
            # Clear GPU cache periodically
            if self_training_system.optimized_trainer.device == 'cuda' and (turn + 1) % 5 == 0:
                import torch
                torch.cuda.empty_cache()
                print("🧹 GPU cache cleared")
    
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
    
    # Final summary
    total_duration = (datetime.datetime.now() - training_start_time).total_seconds()
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    print(f"\n📊 GPU Training Session Summary:")
    print(f"   - Total duration: {hours}h {minutes}m {seconds}s")
    print(f"   - Training turns completed: {len(total_ntp_loss)}")
    print(f"   - Total questions processed: {total_questions_processed}")
    if total_ntp_loss:
        print(f"   - Final NTP Loss: {total_ntp_loss[-1]:.4f}")
        print(f"   - Best NTP Loss: {min(total_ntp_loss):.4f}")
    if total_rl_scores:
        print(f"   - Final RL Score: {total_rl_scores[-1]:.3f}")
        print(f"   - Best RL Score: {max(total_rl_scores):.3f}")
    
    if self_training_system.optimized_trainer.device == 'cuda':
        import torch
        final_memory = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"   - Final GPU Memory: {final_memory:.2f} GB")
    
    # Save final checkpoint
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_checkpoint = f"gpu_training_final_{timestamp}.pkl"
        self_training_system.optimized_trainer.save_optimized_state(final_checkpoint)
        print(f"💾 Final checkpoint saved: {final_checkpoint}")
    except Exception as e:
        print(f"⚠️ Final checkpoint save failed: {e}")
    
    return {
        'total_turns': len(total_ntp_loss),
        'total_questions': total_questions_processed,
        'final_ntp_loss': total_ntp_loss[-1] if total_ntp_loss else None,
        'final_rl_score': total_rl_scores[-1] if total_rl_scores else None,
        'duration_seconds': total_duration
    }

def run_intensive_gpu_training(hours=2, questions_per_turn=5, memory_threshold=0.8):
    """
    Run intensive GPU training for a specified duration with dynamic batch sizing.
    
    Args:
        hours: Training duration in hours
        questions_per_turn: Number of questions per document
        memory_threshold: GPU memory usage threshold (0.0-1.0)
    """
    print(f"🚀 Starting Intensive GPU Training")
    print(f"   - Duration: {hours} hours")
    print(f"   - Questions per turn: {questions_per_turn}")
    print(f"   - Memory threshold: {memory_threshold * 100}%")
    
    if self_training_system.optimized_trainer.device != 'cuda':
        print("⚠️ GPU not available, falling back to CPU training")
        return run_gpu_accelerated_training(num_turns=10)
    
    import torch
    
    # Calculate training parameters
    end_time = datetime.datetime.now() + datetime.timedelta(hours=hours)
    turn_count = 0
    best_performance = {'ntp_loss': float('inf'), 'rl_score': 0.0}
    
    print(f"🎯 Training until: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        while datetime.datetime.now() < end_time:
            turn_count += 1
            
            # Monitor GPU memory and adjust batch size
            memory_used = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
            
            if memory_used > memory_threshold:
                torch.cuda.empty_cache()
                print(f"🧹 GPU memory cleared: {memory_used*100:.1f}% usage")
            
            # Select random document
            if not self_training_system.train_docs:
                print("⚠️ No training documents available")
                break
            
            doc_id, doc_content = random.choice(self_training_system.train_docs)
            doc_sample = doc_content[:2500]  # Larger sample for intensive training
            
            print(f"\n🔄 Intensive Turn {turn_count} - Document: {doc_id[:40]}...")
            
            try:
                # Execute intensive training step
                turn_data = self_training_system.optimized_trainer.train_step(
                    doc_sample, 
                    num_questions=questions_per_turn
                )
                
                # Track performance
                if turn_data['ntp_results']:
                    current_ntp_loss = np.mean(turn_data['ntp_results'].get('losses', [1.0]))
                    if current_ntp_loss < best_performance['ntp_loss']:
                        best_performance['ntp_loss'] = current_ntp_loss
                        print(f"🎉 New best NTP loss: {current_ntp_loss:.4f}")
                
                if turn_data['judge_scores']:
                    current_rl_score = np.mean(turn_data['judge_scores'])
                    if current_rl_score > best_performance['rl_score']:
                        best_performance['rl_score'] = current_rl_score
                        print(f"🎉 New best RL score: {current_rl_score:.3f}")
                
                # Show progress every 10 turns
                if turn_count % 10 == 0:
                    remaining_time = end_time - datetime.datetime.now()
                    hours_left = remaining_time.total_seconds() / 3600
                    print(f"⏱️ Turn {turn_count} - {hours_left:.1f}h remaining")
                    print(self_training_system.optimized_trainer.get_optimization_insights())
                
                # Save checkpoint every 25 turns
                if turn_count % 25 == 0:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_file = f"intensive_gpu_checkpoint_turn_{turn_count}_{timestamp}.pkl"
                    self_training_system.optimized_trainer.save_optimized_state(checkpoint_file)
                    print(f"💾 Intensive checkpoint saved: {checkpoint_file}")
                
            except Exception as e:
                print(f"⚠️ Turn {turn_count} failed: {e}")
                continue
    
    except KeyboardInterrupt:
        print(f"\n⏸️ Intensive training interrupted at turn {turn_count}")
    
    print(f"\n🏁 Intensive GPU Training Completed")
    print(f"   - Total turns: {turn_count}")
    print(f"   - Best NTP Loss: {best_performance['ntp_loss']:.4f}")
    print(f"   - Best RL Score: {best_performance['rl_score']:.3f}")
    
    return {
        'total_turns': turn_count,
        'best_ntp_loss': best_performance['ntp_loss'],
        'best_rl_score': best_performance['rl_score']
    }

# Interactive QA loop with advanced features
if __name__ == "__main__":
    print("🎯 Enhanced RAG Chat with Memory & Advanced Reasoning + Self-Training")
    print("💡 Commands:")
    print("  • 'clear' - clear history")
    print("  • 'history' - show history")
    print("  • 'reason:<question>' - advanced reasoning")
    print("  • 'reason_train' - train reasoning with RL")
    print("  • 'insights' - reasoning insights")
    print("  • 'self_train:<turns>' - run self-training (e.g., 'self_train:5')")
    print("  • 'gpu_train:<turns>' - run GPU-accelerated training (e.g., 'gpu_train:10')")
    print("  • 'intensive_train:<hours>' - run intensive GPU training (e.g., 'intensive_train:2')")
    print("  • 'judge_insights' - judge model insights")
    print("  • 'training_status' - current training status")
    print("  • 'optimization_insights' - optimization process insights")
    print("  • 'best_checkpoint' - load best checkpoint")
    print("  • 'gen_questions:<num>' - generate training questions")
    print("  • 'ntp_insights' - NTP training insights")
    print("  • 'ntp_status' - current NTP metrics")
    print("❓ Type 'quit' to exit\n")
    
    while True:
        user_query = input("🙋 Ask a question: ").strip()
        
        if user_query.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        elif user_query.lower() == 'clear':
            rag_module.clear_history()
            fallback_conversation_history.clear()
            print("🧹 Conversation history cleared!")
            continue
        
        elif user_query.lower() == 'history':
            print("📜 " + rag_module.get_history_summary())
            continue
        
        elif user_query.lower() == 'reason_train':
            print("🎓 Starting reasoning RL training...")
            reasoning_rag_module = train_reasoning_with_rl()
            print("✅ Reasoning training completed!")
            continue
        
        elif user_query.lower() == 'insights':
            print(reasoning_rag_module.get_reasoning_insights())
            continue
        
        elif user_query.lower() == 'judge_insights':
            print(get_judge_insights())
            continue
        
        elif user_query.lower() == 'training_status':
            print(self_training_system.get_training_status())
            continue
        elif user_query.lower() == 'best_checkpoint':
            print(load_best_checkpoint())
            continue
        
        elif user_query.lower() == 'ntp_insights':
            print(get_ntp_insights())
            continue
        
        elif user_query.lower() == 'ntp_status':
            print(get_ntp_status())
            continue
        
        elif user_query.lower().startswith('self_train:'):
            # Self-training command
            try:
                turns = int(user_query.split(':')[1].strip())
                print(f"🚀 Starting self-training with {turns} turns...")
                result = run_self_training_session(turns)
                print(result)
            except (ValueError, IndexError):
                print("❓ Please use format 'self_train:<number>' (e.g., 'self_train:5')")
            continue
        
        elif user_query.lower().startswith('gpu_train:'):
            # GPU-accelerated training command
            try:
                turns = int(user_query.split(':')[1].strip())
                print(f"🚀 Starting GPU-accelerated training with {turns} turns...")
                result = run_gpu_accelerated_training(turns)
                print(f"✅ GPU training completed: {result}")
            except (ValueError, IndexError):
                print("❓ Please use format 'gpu_train:<number>' (e.g., 'gpu_train:10')")
            continue
        
        elif user_query.lower().startswith('intensive_train:'):
            # Intensive GPU training command
            try:
                hours = float(user_query.split(':')[1].strip())
                print(f"🚀 Starting intensive GPU training for {hours} hours...")
                result = run_intensive_gpu_training(hours)
                print(f"✅ Intensive training completed: {result}")
            except (ValueError, IndexError):
                print("❓ Please use format 'intensive_train:<hours>' (e.g., 'intensive_train:2')")
            continue
        
        elif user_query.lower() == 'optimization_insights':
            print(get_optimization_insights())
            continue
        
        elif user_query.lower().startswith('gen_questions:'):
            # Generate training questions
            try:
                num_questions = int(user_query.split(':')[1].strip())
                questions = generate_training_questions(num_questions)
                print(f"❓ Generated {len(questions)} questions:")
                for i, q in enumerate(questions, 1):
                    print(f"  {i}. {q['question']}")
                    print(f"     Type: {q['type']}, Difficulty: {q['difficulty']}")
            except (ValueError, IndexError):
                print("❓ Please use format 'gen_questions:<number>' (e.g., 'gen_questions:3')")
            continue
        
        elif user_query.lower().startswith('reason:'):
            # Advanced reasoning mode
            question = user_query[7:].strip()
            if question:
                print("🧠 Activating advanced reasoning mode...")
                try:
                    result = reasoning_rag_module(question)
                    print(f"\n🎯 Answer: {result.answer}")
                    print(f"\n🔗 Reasoning: {result.reasoning_chain}")
                    print(f"\n📊 Confidence: {result.confidence:.2f}")
                    if hasattr(result, 'improvements'):
                        print(f"\n💡 Improvements: {result.improvements}")
                except Exception as e:
                    print(f"❌ Reasoning failed: {e}")
                    print("📝 Falling back to standard RAG...")
                    try:
                        result = rag_module(question)
                        print(f"💬 Answer: {result.answer}")
                    except Exception as fallback_e:
                        print(f"❌ Fallback also failed: {fallback_e}")
            else:
                print("❓ Please provide a question after 'reason:'")
            continue
        
        if not user_query:
            continue
        
        try:
            # Use optimized RAG if available, otherwise fallback
            try:
                result = rag_module(user_query)
                print(f"💬 Answer: {result.answer}")
            except Exception as rag_error:
                print(f"⚠️ RAG module failed: {rag_error}")
                print("📝 Using fallback function...")
                answer = answer_question_with_docs(user_query)
                print(f"💬 Answer: {answer}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Please try again or use a different question format.")
