# Self-Training RAG System with Judge Model

## Overview

This system implements a sophisticated self-training Retrieval-Augmented Generation (RAG) pipeline that can autonomously improve its reasoning capabilities through reinforcement learning. The system uses your loaded LLM as a judge model to score reasoning chains and train itself iteratively.

## Key Features

### 🧠 **Judge Model System**
- **ReasoningJudge**: Scores reasoning chains on multiple criteria:
  - Accuracy (factual correctness)
  - Coherence (logical flow)
  - Completeness (thoroughness)
  - Evidence Usage (proper citation of sources)
- **Weighted Scoring**: Combines individual scores into overall quality metric
- **Detailed Feedback**: Provides specific strengths and weaknesses

### ❓ **Adaptive Question Generator**
- **Dynamic Difficulty**: Adjusts question complexity based on model performance
- **Question Types**: Generates factual, analytical, comparative, and synthesis questions
- **Diversity Control**: Avoids repetitive questions using history tracking
- **Document-Grounded**: Creates questions directly from your CFA documents

### 🎯 **Self-Training Loop**
- **Automated Turns**: Each turn includes question generation → reasoning → judging → optimization
- **Data Splits**: Proper train/validation/test splits from your documents
- **Performance Tracking**: Monitors improvement across training sessions
- **Adaptive Learning**: Adjusts difficulty based on model capabilities

### 💾 **Model Checkpointing**
- **Automatic Saving**: Saves model weights and training metrics at regular intervals
- **Best Model Tracking**: Identifies and loads best-performing checkpoints
- **Training History**: Maintains detailed logs of training progress
- **Recovery Support**: Can resume training from any checkpoint

### 🚀 **Reinforcement Learning with GRPO**
- **Judge-Based Rewards**: Uses judge scores as rewards for RL optimization
- **Custom GRPO Implementation**: Generalized Reward-based Preference Optimization
- **Multi-Chain Reasoning**: Compares multiple reasoning approaches
- **Self-Improvement**: Model learns to generate better reasoning chains

## Architecture

```
Documents → Question Generator → Reasoning RAG → Judge Model → GRPO Optimizer
     ↓              ↓                ↓             ↓            ↓
  Train/Val/Test   Adaptive      Multi-step    Scoring &    Weight Updates
   Data Split    Difficulty      Reasoning     Feedback    & Checkpoints
```

## Training Process

### Single Training Turn
1. **Document Sampling**: Select document from training set
2. **Question Generation**: Generate 3+ challenging questions
3. **Reasoning Execution**: Process each question through multi-step reasoning
4. **Judge Evaluation**: Score each reasoning chain on multiple criteria
5. **Performance Adaptation**: Adjust question difficulty based on scores
6. **RL Optimization**: Use judge scores to optimize model weights via GRPO

### Multi-Turn Training
- **Validation**: Evaluate on validation set every 3 turns
- **Testing**: Evaluate on test set every 6 turns  
- **Checkpointing**: Save model weights with training metrics
- **Performance Tracking**: Monitor trends and improvements
- **Adaptive Difficulty**: Continuously adjust challenge level

## Usage

### Interactive Commands

```bash
python model.py
```

**Available Commands:**
- `self_train:5` - Run 5 training turns with automatic checkpointing
- `judge_insights` - View judge model performance analytics
- `gen_questions:3` - Generate 3 training questions from documents
- `training_status` - Check current training progress and metrics
- `best_checkpoint` - Load the best performing model checkpoint
- `reason:<question>` - Use advanced multi-step reasoning mode

### Programmatic Usage

```python
from model import self_training_system, run_self_training_session

# Run training session
run_self_training_session(num_turns=10)

# Generate questions
questions = generate_training_questions(5)

# Check judge insights
insights = get_judge_insights()

# Load best model
load_best_checkpoint()
```

### Demo Script

```bash
python demo_self_training.py
```

Runs comprehensive demonstrations of all system components.

## Configuration

### Training Parameters
- **Number of Documents**: 3 retrieved per reasoning step
- **Reasoning Chains**: 2-3 parallel reasoning chains per question
- **Training Turns**: Configurable (5-20 recommended)
- **Questions per Turn**: 3 questions per training turn
- **GRPO Steps**: 5 optimization steps per RL session

### Judge Model Criteria Weights
- **Accuracy**: 30% (factual correctness)
- **Coherence**: 25% (logical consistency)  
- **Completeness**: 25% (thoroughness)
- **Evidence**: 20% (proper source usage)

### Difficulty Distribution (Adaptive)
- **Easy**: 20% (basic factual questions)
- **Medium**: 40% (analytical questions)
- **Hard**: 30% (complex reasoning)
- **Expert**: 10% (synthesis & evaluation)

## Data Management

### Document Splits
- **Training**: 70% of documents for question generation and training
- **Validation**: 15% for performance monitoring during training
- **Testing**: 15% for final evaluation and comparison

### Checkpointing
- **Location**: `model_checkpoints/` directory
- **Format**: Pickle files with model state and training metrics
- **Frequency**: Every 3 training turns
- **Retention**: All checkpoints saved with timestamp and performance scores

## Performance Monitoring

### Training Metrics
- **Train Score**: Performance on training questions
- **Validation Score**: Performance on held-out validation set
- **Test Score**: Performance on final test set
- **Judge Scores**: Individual question-level scores
- **Reasoning Quality**: Overall reasoning capability trend

### Judge Model Analytics
- **Score Distribution**: How scores are distributed across criteria
- **Performance Trends**: Improvement over time
- **Feedback Analysis**: Common strengths and weaknesses identified
- **Difficulty Adaptation**: How question difficulty adjusts to performance

## Example Training Session

```
🚀 Starting self-training session with 5 turns...

🔄 Training Turn 1
❓ Generating questions...
🧠 Processing question 1: What are the key risk factors in portfolio management...
⭐ Judge score: 0.750
🧠 Processing question 2: How do interest rate changes affect bond pricing...
⭐ Judge score: 0.820
🧠 Processing question 3: What role does diversification play in risk reduction...
⭐ Judge score: 0.690
📊 Turn performance: 0.753

🔄 Training Turn 2
...

📈 Validation score: 0.780
🎓 Running GRPO optimization...
✅ GRPO optimization completed!
💾 Checkpoint saved: checkpoint_step_5_20250603_143022.pkl

✅ Self-training session completed!

📊 Training Summary:
- Total turns: 5
- Questions generated: 15
- Average performance: 0.785
- Performance trend: +0.032
- Checkpoints saved: 2
```

## Technical Details

### GRPO Implementation
- **Custom Implementation**: Built specifically for DSPy programs
- **Judge-Based Rewards**: Uses multi-criteria judge scores as rewards
- **Exploration**: Creates program variants for optimization
- **Convergence**: Tracks best performance across rollouts

### Reasoning Chain Evaluation
- **Multi-Step Analysis**: Evaluates each step in reasoning process
- **Context Verification**: Checks reasoning against source documents
- **Coherence Assessment**: Ensures logical flow between steps
- **Evidence Integration**: Validates proper use of retrieved information

### Adaptive Learning
- **Performance Monitoring**: Tracks scores across different question types
- **Difficulty Adjustment**: Automatically adjusts based on model capability
- **Question Diversity**: Ensures variety in training questions
- **Continuous Improvement**: Learns from judge feedback to improve reasoning

## Benefits

1. **Autonomous Improvement**: Model trains itself without manual intervention
2. **Proper Evaluation**: Uses train/validation/test splits for robust assessment
3. **Comprehensive Scoring**: Multi-criteria judge evaluation for thorough feedback
4. **Adaptive Difficulty**: Automatically adjusts challenge level
5. **Weight Persistence**: Saves and loads trained model weights
6. **Performance Tracking**: Detailed metrics and progress monitoring
7. **Document Utilization**: Trains directly on your uploaded CFA documents

## Future Enhancements

- **Multi-Model Judging**: Use ensemble of judge models
- **Advanced RL Algorithms**: Implement PPO, DPO for optimization
- **Curriculum Learning**: Structured progression through difficulty levels
- **Meta-Learning**: Learn to learn more effectively
- **Distributed Training**: Scale across multiple GPUs/nodes
