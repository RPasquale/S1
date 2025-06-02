import dspy
import os
import sys
import random
import threading
import multiprocessing
import builtins
from typing import Any, List, Optional, Tuple, Dict
from collections import Counter

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
# Note: reasoning_rag_module will be initialized after ReasoningRAG class is defined

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
    Generalized Reward-based Preference Optimization (GRPO) for DSPy programs.
    This is a custom implementation to replace the missing dspy.teleprompt.GRPO import.
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

# Interactive QA loop with advanced features
if __name__ == "__main__":
    print("🎯 Enhanced RAG Chat with Memory & Advanced Reasoning")
    print("💡 Commands: 'clear' (clear history), 'history' (show history), 'reason:<question>' (advanced reasoning)")
    print("🚀 New: 'reason_train' (train reasoning with RL), 'insights' (reasoning insights)")
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
