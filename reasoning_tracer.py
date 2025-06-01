"""
Reasoning Tracer Module for Model Training Pipeline.
This module provides comprehensive tracing of the multi-stage query generation and model training process.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

class ReasoningTracer:
    """Provides detailed reasoning traces for the model training pipeline."""
    
    def __init__(self, output_dir: str = "trained-models"):
        self.output_dir = output_dir
        self.trace_dir = os.path.join(output_dir, "reasoning-traces")
        os.makedirs(self.trace_dir, exist_ok=True)
        
        self.current_trace = {
            "session_id": f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "pipeline_stages": [],
            "generator_outputs": {},
            "transformations": [],
            "quality_metrics": {},
            "decision_points": [],
            "final_selection": {},
            "errors_and_fallbacks": []
        }
        
        print(f"🔍 Reasoning Tracer initialized. Traces will be saved to: {self.trace_dir}")
    
    def log_stage(self, stage_name: str, description: str, inputs: Dict = None, outputs: Dict = None):
        """Log a pipeline stage with inputs and outputs."""
        stage_info = {
            "stage": stage_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs or {},
            "outputs": outputs or {},
            "duration_ms": 0
        }
        
        self.current_trace["pipeline_stages"].append(stage_info)
        print(f"📝 STAGE: {stage_name} - {description}")
        
        if inputs:
            print(f"   📥 INPUTS: {self._format_dict(inputs)}")
        if outputs:
            print(f"   📤 OUTPUTS: {self._format_dict(outputs)}")
        
        return len(self.current_trace["pipeline_stages"]) - 1  # Return stage index
    
    def log_generator_output(self, generator_name: str, input_doc: str, raw_output: str, 
                           parsed_output: List[str], metadata: Dict = None):
        """Log detailed output from a query generator."""
        generator_info = {
            "generator": generator_name,
            "timestamp": datetime.now().isoformat(),
            "input_document_sample": input_doc[:200] + "..." if len(input_doc) > 200 else input_doc,
            "raw_llm_output": raw_output,
            "parsed_questions": parsed_output,
            "num_questions_generated": len(parsed_output),
            "metadata": metadata or {}
        }
        
        if generator_name not in self.current_trace["generator_outputs"]:
            self.current_trace["generator_outputs"][generator_name] = []
        
        self.current_trace["generator_outputs"][generator_name].append(generator_info)
        
        print(f"🤖 GENERATOR: {generator_name}")
        print(f"   📖 INPUT SAMPLE: {input_doc[:100]}...")
        print(f"   🗣️  RAW OUTPUT: {raw_output[:150]}...")
        print(f"   ❓ PARSED QUESTIONS ({len(parsed_output)}):")
        for i, q in enumerate(parsed_output[:3]):  # Show first 3 questions
            print(f"      {i+1}. {q}")
        if len(parsed_output) > 3:
            print(f"      ... and {len(parsed_output) - 3} more")
        print()
    
    def log_transformation(self, from_stage: str, to_stage: str, input_data: Any, 
                         output_data: Any, transformation_type: str, description: str):
        """Log data transformations between pipeline stages."""
        transformation_info = {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "transformation_type": transformation_type,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "input_summary": self._summarize_data(input_data),
            "output_summary": self._summarize_data(output_data)
        }
        
        self.current_trace["transformations"].append(transformation_info)
        
        print(f"🔄 TRANSFORMATION: {from_stage} → {to_stage}")
        print(f"   🎯 TYPE: {transformation_type}")
        print(f"   📝 DESCRIPTION: {description}")
        print(f"   📊 INPUT: {transformation_info['input_summary']}")
        print(f"   📊 OUTPUT: {transformation_info['output_summary']}")
        print()
    
    def log_decision_point(self, decision_name: str, options: List[str], chosen_option: str, 
                          reasoning: str, confidence: float = None):
        """Log decision points in the pipeline."""
        decision_info = {
            "decision": decision_name,
            "options": options,
            "chosen": chosen_option,
            "reasoning": reasoning,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        self.current_trace["decision_points"].append(decision_info)
        
        print(f"🎯 DECISION: {decision_name}")
        print(f"   🔀 OPTIONS: {', '.join(options)}")
        print(f"   ✅ CHOSEN: {chosen_option}")
        print(f"   💭 REASONING: {reasoning}")
        if confidence:
            print(f"   📊 CONFIDENCE: {confidence:.2f}")
        print()
    
    def log_quality_metrics(self, stage: str, metrics: Dict[str, float]):
        """Log quality metrics for evaluation."""
        if stage not in self.current_trace["quality_metrics"]:
            self.current_trace["quality_metrics"][stage] = []
        
        metric_info = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.current_trace["quality_metrics"][stage].append(metric_info)
        
        print(f"📊 QUALITY METRICS - {stage}:")
        for metric, value in metrics.items():
            print(f"   📈 {metric}: {value:.4f}")
        print()
    
    def log_error_fallback(self, stage: str, error: str, fallback_action: str, success: bool):
        """Log errors and fallback actions."""
        error_info = {
            "stage": stage,
            "error": error,
            "fallback_action": fallback_action,
            "fallback_success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        self.current_trace["errors_and_fallbacks"].append(error_info)
        
        print(f"⚠️  ERROR in {stage}: {error}")
        print(f"   🔧 FALLBACK: {fallback_action}")
        print(f"   {'✅' if success else '❌'} RESULT: {'Success' if success else 'Failed'}")
        print()
    
    def log_generator_interaction(self, query_generators: Dict, document: str, 
                                results: Dict[str, List[str]]):
        """Log how different generators interact and build on each other."""
        print(f"🤝 GENERATOR INTERACTION ANALYSIS")
        print(f"   📖 Document sample: {document[:100]}...")
        print()
        
        # Analyze overlap and diversity
        all_questions = []
        generator_stats = {}
        
        for gen_name, questions in results.items():
            all_questions.extend(questions)
            generator_stats[gen_name] = {
                "count": len(questions),
                "avg_length": sum(len(q) for q in questions) / max(1, len(questions)),
                "unique_words": len(set(" ".join(questions).lower().split()))
            }
            
            print(f"   🤖 {gen_name}: {len(questions)} questions")
            print(f"      📏 Avg length: {generator_stats[gen_name]['avg_length']:.1f} chars")
            print(f"      🔤 Unique words: {generator_stats[gen_name]['unique_words']}")
        
        # Calculate cross-generator metrics
        total_questions = len(all_questions)
        unique_questions = len(set(all_questions))
        
        print(f"   📊 OVERALL DIVERSITY:")
        print(f"      📝 Total questions: {total_questions}")
        print(f"      🎯 Unique questions: {unique_questions}")
        print(f"      📈 Diversity ratio: {unique_questions/max(1, total_questions):.2f}")
        print()
        
        # Log this interaction
        interaction_info = {
            "document_sample": document[:200],
            "generator_results": results,
            "generator_stats": generator_stats,
            "overall_metrics": {
                "total_questions": total_questions,
                "unique_questions": unique_questions,
                "diversity_ratio": unique_questions/max(1, total_questions)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if "generator_interactions" not in self.current_trace:
            self.current_trace["generator_interactions"] = []
        self.current_trace["generator_interactions"].append(interaction_info)
    
    def finalize_trace(self, final_queries: List[str], selection_criteria: Dict):
        """Finalize the reasoning trace with final results."""
        self.current_trace["final_selection"] = {
            "selected_queries": final_queries,
            "total_selected": len(final_queries),
            "selection_criteria": selection_criteria,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save the complete trace
        trace_filename = f"reasoning_trace_{self.current_trace['session_id']}.json"
        trace_path = os.path.join(self.trace_dir, trace_filename)
        
        with open(trace_path, "w", encoding='utf-8') as f:
            json.dump(self.current_trace, f, indent=2, ensure_ascii=False)
        
        # Create a human-readable summary
        summary_path = os.path.join(self.trace_dir, f"summary_{self.current_trace['session_id']}.md")
        self._create_markdown_summary(summary_path)
        
        print(f"✅ REASONING TRACE COMPLETE")
        print(f"   💾 Detailed trace: {trace_path}")
        print(f"   📋 Summary: {summary_path}")
        print(f"   🎯 Final queries selected: {len(final_queries)}")
        print()
        
        return trace_path
    
    def _format_dict(self, data: Dict, max_items: int = 3) -> str:
        """Format dictionary for readable output."""
        if not data:
            return "{}"
        
        items = list(data.items())[:max_items]
        formatted = []
        
        for key, value in items:
            if isinstance(value, str) and len(value) > 50:
                value = value[:47] + "..."
            elif isinstance(value, list) and len(value) > 3:
                value = f"[{len(value)} items]"
            
            formatted.append(f"{key}: {value}")
        
        result = "{" + ", ".join(formatted)
        if len(data) > max_items:
            result += f", ...and {len(data) - max_items} more"
        result += "}"
        
        return result
    
    def _summarize_data(self, data: Any) -> str:
        """Create a summary of data for logging."""
        if isinstance(data, list):
            return f"List[{len(data)} items]"
        elif isinstance(data, dict):
            return f"Dict[{len(data)} keys]"
        elif isinstance(data, str):
            return f"String[{len(data)} chars]: '{data[:50]}...'" if len(data) > 50 else f"String: '{data}'"
        else:
            return f"{type(data).__name__}: {str(data)[:50]}..."
    
    def _create_markdown_summary(self, path: str):
        """Create a human-readable markdown summary of the trace."""
        trace = self.current_trace
        
        with open(path, "w", encoding='utf-8') as f:
            f.write(f"# Reasoning Trace Summary\n\n")
            f.write(f"**Session ID:** {trace['session_id']}\n")
            f.write(f"**Generated:** {trace['timestamp']}\n\n")
            
            # Pipeline stages
            f.write(f"## Pipeline Stages\n\n")
            for i, stage in enumerate(trace['pipeline_stages'], 1):
                f.write(f"### {i}. {stage['stage']}\n")
                f.write(f"**Description:** {stage['description']}\n")
                f.write(f"**Time:** {stage['timestamp']}\n\n")
            
            # Generator outputs
            if trace['generator_outputs']:
                f.write(f"## Generator Outputs\n\n")
                for gen_name, outputs in trace['generator_outputs'].items():
                    f.write(f"### {gen_name}\n")
                    total_questions = sum(len(output['parsed_questions']) for output in outputs)
                    f.write(f"**Total questions generated:** {total_questions}\n")
                    f.write(f"**Documents processed:** {len(outputs)}\n\n")
            
            # Decision points
            if trace['decision_points']:
                f.write(f"## Key Decisions\n\n")
                for decision in trace['decision_points']:
                    f.write(f"### {decision['decision']}\n")
                    f.write(f"**Options:** {', '.join(decision['options'])}\n")
                    f.write(f"**Chosen:** {decision['chosen']}\n")
                    f.write(f"**Reasoning:** {decision['reasoning']}\n\n")
            
            # Quality metrics
            if trace['quality_metrics']:
                f.write(f"## Quality Metrics\n\n")
                for stage, metrics_list in trace['quality_metrics'].items():
                    f.write(f"### {stage}\n")
                    for metrics_info in metrics_list:
                        for metric, value in metrics_info['metrics'].items():
                            f.write(f"- **{metric}:** {value:.4f}\n")
                    f.write("\n")
            
            # Final selection
            if trace['final_selection']:
                final = trace['final_selection']
                f.write(f"## Final Selection\n\n")
                f.write(f"**Queries selected:** {final['total_selected']}\n")
                f.write(f"**Selection criteria:** {final['selection_criteria']}\n\n")
                f.write(f"### Selected Queries\n\n")
                for i, query in enumerate(final['selected_queries'], 1):
                    f.write(f"{i}. {query}\n")
