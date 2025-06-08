# Advanced Self-Training RAG Agent with Judge Model

A sophisticated AI agent that can autonomously improve its reasoning capabilities through self-training, using your local LLM as both the reasoning engine and the judge model.

## 🌟 What This Agent Does

This agent creates a **self-improving RAG system** that:
- 📚 **Learns from your documents** (CFA materials, PDFs, etc.)
- 🤖 **Generates its own training questions** adaptively
- 🧠 **Reasons through multi-step analysis** with chain-of-thought
- ⚖️ **Judges its own reasoning quality** using multiple criteria
- 🎯 **Trains itself** using reinforcement learning (GRPO)
- 💾 **Saves and loads model checkpoints** automatically
- 📈 **Tracks performance** across training sessions

## 🏗️ System Architecture

```
Documents (CFA/PDFs) → Embedding Index (Voyager/ColBERT)
                              ↓
Question Generator ← Adaptive Difficulty Control
       ↓
Multi-Step Reasoning RAG ← Search & Retrieval
       ↓
Judge Model (Scoring) ← Multiple Quality Criteria
       ↓
GRPO Optimizer ← Reinforcement Learning
       ↓
Model Checkpoints ← Weight Updates & Saving
```

### Core Components

1. **🧩 Question Generator** - Creates diverse questions from your documents
2. **🧠 Reasoning RAG** - Multi-step reasoning with sub-question decomposition  
3. **⚖️ Judge Model** - Scores reasoning chains on accuracy, coherence, completeness
4. **🎓 GRPO Optimizer** - Reinforcement learning for self-improvement
5. **💾 Checkpoint Manager** - Automatic model saving and loading

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Windows OS** (PowerShell)
- **Ollama** running locally (`ollama serve`)
- **LLM Model** (e.g., `deepseek-r1:8b`)
- **PDF Documents** to learn from

### Installation

1. **Clone and setup:**
   ```powershell
   cd C:\Users\Admin\S1
   python -m venv venv
   .\venv\Scripts\Activate
   pip install pylate PyPDF2 dspy-ai numpy pathlib dataclasses matplotlib pandas psutil wandb
   ```

2. **Start Ollama server:**
   ```powershell
   ollama serve
   # In another terminal:
   ollama pull deepseek-r1:8b
   ```

3. **First run (builds document index):**
   ```powershell
   python model.py
   ```

4. **Start CFA Expert Training:**
   ```powershell
   python train_cfa_expert.py
   ```

5. **Start Unlimited Training with WnB (New!):**
   ```powershell
   python unlimited_training_wnb.py
   ```

## 🎮 How to Use

### 🎓 CFA Expert Training (New!)

**Launch the Ultimate Training Center:**
```powershell
python train_cfa_expert.py
```

**Training Options:**
- **Quick Expert Training (4 hours)** - Fast path to CFA expertise
- **Ultimate Long Training (8-72 hours)** - Maximum expertise development
- **Real-time Monitoring** - Watch weight updates and progress live

**Features:**
- ✅ **Explicit Weight Tracking** - See every model update
- ✅ **Performance Monitoring** - Real-time expertise progression
- ✅ **Automatic Checkpointing** - Never lose training progress
- ✅ **Multi-day Training** - Continuous learning capability
- ✅ **Expert Assessment** - Comprehensive final evaluation

### 🚀 Unlimited Training with WnB (New!)

**Launch Unlimited Duration Training:**
```powershell
python unlimited_training_wnb.py
```

**Revolutionary Features:**
- 🎯 **Unlimited Duration** - No time restrictions, trains until convergence
- 📊 **Full WnB Integration** - Real-time visualization in Weights & Biases
- 🔄 **Smart Convergence** - Advanced stopping criteria based on improvement
- 📈 **Live Dashboards** - Monitor loss, rewards, and metrics remotely
- 🌐 **Remote Monitoring** - Track training from anywhere
- 📋 **Experiment Management** - Compare runs and hyperparameters
- 🎨 **Custom Visualizations** - Advanced charts and analysis
- 💾 **Artifact Logging** - Automatic checkpoint and model versioning

**WnB Dashboard Features:**
- Real-time loss/reward curves
- Training dynamics visualization
- Convergence tracking charts
- Performance trend analysis
- Weight update heatmaps
- Content exploration metrics
- Judge scoring distribution

### Interactive Chat Mode

Run the main script to enter interactive mode:

```powershell
python model.py
```

**Available Commands:**

- **Basic Q&A:** Just type your question
- **Advanced Reasoning:** `reason: How do interest rates affect portfolio diversification?`
- **Self-Training:** `self_train` - Start autonomous training loop
- **Judge Demo:** `judge_demo` - See how the judge model works
- **Training Status:** `training_status` - Check current training progress
- **Load Checkpoint:** `load_best` - Load the best performing model
- **Clear History:** `clear` - Reset conversation history
- **Show History:** `history` - View conversation summary
- **Exit:** `quit`

### Self-Training Commands

```
🙋 Ask a question: self_train
🚀 Starting self-training loop for 10 turns
📊 Data split: 25 train, 5 val, 3 test docs
🔄 Training Turn 1
❓ Generating questions...
🧠 Processing question 1: What factors influence portfolio risk assessment...
⭐ Judge score: 0.785
📊 Turn performance: 0.742
📈 Validation score: 0.681
✅ Checkpoint saved: model_checkpoints/checkpoint_step_1_20250603_143022.pkl
```

### Demo Script

For a quick demonstration:

```powershell
python demo_self_training.py
```

## 🧠 Advanced Features

### Multi-Step Reasoning

The agent breaks down complex questions into sub-questions:

```
Question: "How should asset allocation change with market volatility?"

Sub-questions:
1. What is the relationship between volatility and risk?
2. How do different asset classes respond to volatility?
3. What allocation strategies minimize volatility impact?

→ Individual analysis of each sub-question
→ Synthesis into comprehensive answer
→ Reflection and validation
```

### Judge Model Criteria

The judge evaluates reasoning on 4 dimensions:

- **🎯 Accuracy (30%)** - Factual correctness vs. document content
- **🔗 Coherence (25%)** - Logical flow and consistency
- **📋 Completeness (25%)** - Thoroughness of analysis
- **📚 Evidence (20%)** - Proper use of source material

### Adaptive Question Generation

- **Difficulty Adaptation:** Increases complexity if performance > 80%, decreases if < 50%
- **Question Types:** Factual, analytical, comparative, synthesis
- **Diversity Control:** Avoids repetitive questions
- **Document-Grounded:** Questions come directly from your content

### Model Checkpointing

```python
# Automatic saving during training
Checkpoint: checkpoint_step_5_20250603_143045.pkl
- Training Score: 0.742
- Validation Score: 0.681  
- Test Score: 0.695
- Judge Scores: [0.785, 0.823, 0.756]
- Model Weights: Saved
```

## 📊 Training Data Flow

1. **Document Split:** 70% train, 15% validation, 15% test
2. **Question Generation:** Adaptive difficulty based on performance
3. **Reasoning:** Multi-step analysis with evidence gathering
4. **Judging:** Comprehensive scoring on multiple criteria
5. **Optimization:** GRPO reinforcement learning updates
6. **Checkpointing:** Save best models automatically

## 🔧 Configuration

### Model Settings

```python
# In model.py, you can adjust:
lm = dspy.LM('ollama_chat/deepseek-r1:8b')  # Change model
reasoning_chains = 3  # Number of reasoning chains to compare
num_docs = 5  # Documents retrieved per query
difficulty_distribution = {"easy": 0.2, "medium": 0.4, "hard": 0.3, "expert": 0.1}
```

### Training Parameters

```python
# Self-training configuration:
num_turns = 10  # Training iterations
questions_per_turn = 3  # Questions generated each turn
grpo_steps = 5  # RL optimization steps
checkpoint_frequency = 3  # Save every N turns
```

## 📁 Project Structure

```
C:\Users\Admin\S1\
├── model.py                    # Main agent with all components
├── unlimited_training_wnb.py   # Unlimited WnB training system (NEW!)
├── autonomous_training.py      # Long-term autonomous training system
├── train_cfa_expert.py        # Ultimate training launcher (NEW!)
├── run_long_training.py       # Extended training runner (NEW!)
├── quick_train_expert.py      # Quick 4-hour training (NEW!)
├── monitor_training.py        # Real-time training monitor (NEW!)
├── demo_self_training.py      # Training demonstrations
├── README.md                  # This documentation
├── model_checkpoints/         # Saved model weights
├── unlimited_training_*/      # Unlimited WnB training sessions (NEW!)
├── autonomous_training_state/ # Training progress data
├── training_analysis/         # Performance plots and analysis
├── ultimate_training_*/       # Extended training session data
├── pylate-index/             # Document embeddings index
│   └── index/
│       ├── index.voyager
│       └── *.sqlite
└── venv/                     # Python virtual environment
```

## 🔍 How It Works Internally

### 1. Document Processing
- PDFs → Text extraction → ColBERT embeddings → Voyager index
- Train/validation/test split for proper evaluation

### 2. Question Generation
```python
QuestionGenerator(
    document_content=sample_doc,
    difficulty_level="medium",
    question_type="analytical"
) → challenging_question
```

### 3. Multi-Step Reasoning
```python
ReasoningRAG:
  1. Decompose question into sub-questions
  2. Analyze each sub-question with context
  3. Generate multiple reasoning chains
  4. Compare and select best reasoning
  5. Reflect and validate conclusion
```

### 4. Judge Scoring
```python
ReasoningJudge:
  accuracy_score: 0.85
  coherence_score: 0.78
  completeness_score: 0.82
  evidence_score: 0.79
  → overall_score: 0.81
```

### 5. Reinforcement Learning
```python
GRPO:
  judge_scores → rewards
  multiple_rollouts → comparison
  best_performance → model_update
  checkpoint_save()
```

## 🎯 Use Cases

- **Financial Analysis:** CFA study materials, investment research
- **Research Papers:** Academic literature analysis
- **Legal Documents:** Contract and policy analysis  
- **Technical Manuals:** Engineering and scientific documentation
- **Training Materials:** Educational content processing

## 🔧 Troubleshooting

### Common Issues

**Ollama Connection Error:**
```bash
# Ensure Ollama is running:
ollama serve
# Check model availability:
ollama list
```

**Memory Issues:**
```python
# Reduce batch size in model.py:
batch_size = 16  # Instead of 32
num_docs = 3     # Instead of 5
```

**No Documents Found:**
```bash
# Ensure PDFs are in the correct directory
# Check that PyPDF2 can read your files
```

### Performance Tips

- **GPU Usage:** Ensure CUDA is available for faster embeddings
- **Batch Size:** Adjust based on available memory
- **Question Frequency:** Reduce questions per turn if training is slow
- **Checkpoint Cleanup:** Periodically clean old checkpoints

## 🚀 Advanced Usage

### Custom Judge Criteria

Modify the judge weights in `ReasoningJudgeModel`:

```python
self.criteria_weights = {
    'accuracy': 0.4,      # Increase accuracy importance
    'coherence': 0.2,     # Decrease coherence weight
    'completeness': 0.2,
    'evidence': 0.2
}
```

### Training Resumption

```python
# Load specific checkpoint:
checkpoint_data = checkpoint_manager.load_checkpoint("path/to/checkpoint.pkl")

# Load best checkpoint:
best_checkpoint = checkpoint_manager.get_best_checkpoint("val_score")
```

### Custom Question Types

Add new question types in `AdaptiveQuestionGenerator`:

```python
self.question_types = [
    "factual", "analytical", "comparative", 
    "synthesis", "evaluation", "application"  # New types
]
```

## 📈 Performance Monitoring

The system tracks:
- **Training Scores:** Performance on generated questions
- **Validation Scores:** Performance on held-out data
- **Test Scores:** Final evaluation metric
- **Judge Score Distribution:** Quality of reasoning over time
- **Question Difficulty Adaptation:** How difficulty changes with performance

## 🤝 Contributing

To extend the system:
1. **Add New Reasoning Strategies:** Extend `ReasoningRAG`
2. **Custom Judge Criteria:** Modify `ReasoningJudge` signature
3. **Different RL Algorithms:** Replace GRPO with other optimizers
4. **New Question Types:** Extend `QuestionGenerator`

## 📝 License

This project is for educational and research purposes. Please ensure compliance with your LLM provider's terms of service.

---

**Ready to train your own self-improving AI agent?** 🚀

```powershell
.\venv\Scripts\Activate
python model.py
# Type: self_train
```

After indexing, re-running the script skips re-indexing and drops you into an interactive prompt:

```powershell
python model.py
# Interactive QA over indexed PDFs. Blank input to exit.
Ask a question: <ask questions about the documents>
```

The script will retrieve the top K relevant PDFs, assemble their text, and call your LLM via dspy's `ChainOfThought` to generate a detailed answer.


## Configuration

- **LLM Model**: Change the `pylate_model_id` in `model.py` to any valid Hugging Face or Ollama model (default: `lightonai/Reason-ModernColBERT`).
- **Top-K Results**: Adjust `top_k` in `answer_question_with_docs()` for more or fewer retrieved documents.
- **Document Path**: Modify `doc_folder` in `model.py` to point to your PDF collection.


## Troubleshooting

- **File Not Found**: Ensure your PDFs are in the correct folder and accessible.
- **Indexing Again**: Delete the `pylate-index` folder if you need to rebuild from scratch.
- **LLM Errors**: Verify your Ollama server is running and `api_base`/`api_key` in `model.py` are correct.


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
