"""
Custom DSPy Generator System with Reasoning Traces
==================================================

This module allows users to create custom DSPy generators with custom signatures
and see detailed reasoning traces as they interact with each other using the same model.
"""

import dspy
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, asdict
from reasoning_tracer import ReasoningTracer

# Global model configuration - ALL generators use the same model
SHARED_MODEL_CONFIG = {
    "model": "ollama_chat/deepseek-r1:8b",
    "api_base": "http://localhost:11434", 
    "api_key": ""
}

class DSPyGeneratorRegistry:
    """Registry to manage all custom DSPy generators and their reasoning traces."""
    
    def __init__(self):
        self.generators = {}
        self.signatures = {}
        self.reasoning_tracer = ReasoningTracer()
        self.shared_lm = None
        self.generator_chains = []
        self.trace_history = []
        
        # Initialize shared language model
        self._setup_shared_model()
    
    def _setup_shared_model(self):
        """Setup the shared language model that ALL generators will use."""
        try:
            self.shared_lm = dspy.LM(
                model=SHARED_MODEL_CONFIG["model"],
                api_base=SHARED_MODEL_CONFIG["api_base"], 
                api_key=SHARED_MODEL_CONFIG["api_key"]
            )
            dspy.configure(lm=self.shared_lm)
            
            self.reasoning_tracer.log_step(
                "system_init", 
                "Shared Model Setup",
                f"Configured shared model: {SHARED_MODEL_CONFIG['model']}",
                {"model_config": SHARED_MODEL_CONFIG}
            )
            print(f"✅ Shared DSPy model configured: {SHARED_MODEL_CONFIG['model']}")
            
        except Exception as e:
            self.reasoning_tracer.log_step(
                "system_init",
                "Model Setup Error", 
                f"Failed to setup shared model: {str(e)}",
                {"error": str(e)}
            )
            print(f"❌ Error setting up shared model: {e}")
    
    def register_signature(self, name: str, signature_class: Type[dspy.Signature]):
        """Register a custom DSPy signature."""
        self.signatures[name] = signature_class
        
        self.reasoning_tracer.log_step(
            "signature_registration",
            f"Registered Signature: {name}",
            f"Signature '{name}' registered with fields: {signature_class.__annotations__}",
            {
                "signature_name": name,
                "input_fields": {k: str(v) for k, v in signature_class.__annotations__.items() 
                               if hasattr(signature_class, k) and getattr(signature_class, k).__class__.__name__ == 'InputField'},
                "output_fields": {k: str(v) for k, v in signature_class.__annotations__.items() 
                                if hasattr(signature_class, k) and getattr(signature_class, k).__class__.__name__ == 'OutputField'}
            }
        )
        print(f"📝 Registered signature: {name}")
    
    def create_generator(self, name: str, signature_name: str, generator_type: str = "ChainOfThought"):
        """Create a new DSPy generator using a registered signature."""
        if signature_name not in self.signatures:
            raise ValueError(f"Signature '{signature_name}' not registered")
        
        signature_class = self.signatures[signature_name]
        
        # Create generator using the shared model
        if generator_type == "ChainOfThought":
            generator = dspy.ChainOfThought(signature_class)
        elif generator_type == "Predict":
            generator = dspy.Predict(signature_class)
        elif generator_type == "ReAct":
            generator = dspy.ReAct(signature_class)
        else:
            raise ValueError(f"Unsupported generator type: {generator_type}")
        
        # Wrap generator to add reasoning traces
        wrapped_generator = self._wrap_generator_with_tracing(generator, name, signature_name)
        
        self.generators[name] = {
            "generator": wrapped_generator,
            "signature": signature_name,
            "type": generator_type,
            "created_at": datetime.now().isoformat(),
            "call_count": 0
        }
        
        self.reasoning_tracer.log_step(
            "generator_creation",
            f"Created Generator: {name}",
            f"Generator '{name}' created with signature '{signature_name}' using {generator_type}",
            {
                "generator_name": name,
                "signature_name": signature_name,
                "generator_type": generator_type,
                "shared_model": SHARED_MODEL_CONFIG["model"]
            }
        )
        print(f"🎯 Created generator: {name} ({generator_type} with {signature_name})")
        
        return wrapped_generator
    
    def _wrap_generator_with_tracing(self, generator, name: str, signature_name: str):
        """Wrap a generator to add detailed reasoning traces."""
        
        class TracedGenerator:
            def __init__(self, original_generator, gen_name, sig_name, tracer, registry):
                self.original = original_generator
                self.name = gen_name
                self.signature_name = sig_name
                self.tracer = tracer
                self.registry = registry
            
            def __call__(self, **kwargs):
                call_id = f"{self.name}_{int(time.time() * 1000)}"
                
                # Log input
                self.tracer.log_step(
                    call_id,
                    f"Generator Input: {self.name}",
                    f"Calling generator '{self.name}' with signature '{self.signature_name}'",
                    {
                        "generator_name": self.name,
                        "signature": self.signature_name,
                        "input_data": kwargs,
                        "model_used": SHARED_MODEL_CONFIG["model"],
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                print(f"\n🔄 Executing {self.name} ({self.signature_name})")
                print(f"   Input: {json.dumps(kwargs, indent=2)}")
                
                try:
                    # Execute the original generator
                    start_time = time.time()
                    result = self.original(**kwargs)
                    execution_time = time.time() - start_time
                    
                    # Extract output data
                    output_data = {}
                    if hasattr(result, '__dict__'):
                        for key, value in result.__dict__.items():
                            if not key.startswith('_'):
                                output_data[key] = value
                    else:
                        output_data = {"result": str(result)}
                    
                    # Log successful output
                    self.tracer.log_step(
                        call_id,
                        f"Generator Output: {self.name}",
                        f"Generator '{self.name}' completed successfully",
                        {
                            "generator_name": self.name,
                            "signature": self.signature_name,
                            "output_data": output_data,
                            "execution_time_seconds": execution_time,
                            "model_used": SHARED_MODEL_CONFIG["model"],
                            "success": True
                        }
                    )
                    
                    print(f"   ✅ Output: {json.dumps(output_data, indent=2)}")
                    print(f"   ⏱️  Execution time: {execution_time:.2f}s")
                    
                    # Update call count
                    self.registry.generators[self.name]["call_count"] += 1
                    
                    return result
                    
                except Exception as e:
                    # Log error
                    self.tracer.log_step(
                        call_id,
                        f"Generator Error: {self.name}",
                        f"Generator '{self.name}' failed with error: {str(e)}",
                        {
                            "generator_name": self.name,
                            "signature": self.signature_name,
                            "error": str(e),
                            "model_used": SHARED_MODEL_CONFIG["model"],
                            "success": False
                        }
                    )
                    
                    print(f"   ❌ Error: {str(e)}")
                    raise
        
        return TracedGenerator(generator, name, signature_name, self.reasoning_tracer, self)
    
    def create_generator_chain(self, chain_name: str, generator_sequence: List[Dict[str, Any]]):
        """Create a chain of generators that pass data between each other."""
        
        if not generator_sequence:
            raise ValueError("Generator sequence cannot be empty")
        
        chain_id = f"chain_{chain_name}_{int(time.time())}"
        
        self.reasoning_tracer.log_step(
            chain_id,
            f"Chain Creation: {chain_name}",
            f"Creating generator chain '{chain_name}' with {len(generator_sequence)} steps",
            {
                "chain_name": chain_name,
                "steps": generator_sequence,
                "shared_model": SHARED_MODEL_CONFIG["model"]
            }
        )
        
        class GeneratorChain:
            def __init__(self, name, sequence, registry, tracer):
                self.name = name
                self.sequence = sequence
                self.registry = registry
                self.tracer = tracer
                self.chain_id = chain_id
            
            def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
                """Execute the generator chain with detailed tracing."""
                
                print(f"\n🔗 Executing Generator Chain: {self.name}")
                print(f"   Initial Input: {json.dumps(initial_input, indent=2)}")
                
                current_data = initial_input.copy()
                chain_results = []
                
                for step_idx, step_config in enumerate(self.sequence):
                    step_name = step_config["generator"]
                    input_mapping = step_config.get("input_mapping", {})
                    output_mapping = step_config.get("output_mapping", {})
                    
                    print(f"\n  📍 Step {step_idx + 1}: {step_name}")
                    
                    if step_name not in self.registry.generators:
                        raise ValueError(f"Generator '{step_name}' not found in registry")
                    
                    generator_info = self.registry.generators[step_name]
                    generator = generator_info["generator"]
                    
                    # Map input data according to input_mapping
                    mapped_input = {}
                    for target_field, source_field in input_mapping.items():
                        if source_field in current_data:
                            mapped_input[target_field] = current_data[source_field]
                        else:
                            print(f"   ⚠️  Warning: Source field '{source_field}' not found in current data")
                    
                    # Add any direct fields
                    for key, value in current_data.items():
                        if key not in [source for source in input_mapping.values()]:
                            mapped_input[key] = value
                    
                    # Log chain step
                    step_id = f"{self.chain_id}_step_{step_idx}"
                    self.tracer.log_step(
                        step_id,
                        f"Chain Step {step_idx + 1}: {step_name}",
                        f"Executing step {step_idx + 1} in chain '{self.name}'",
                        {
                            "chain_name": self.name,
                            "step_index": step_idx,
                            "generator_name": step_name,
                            "input_mapping": input_mapping,
                            "mapped_input": mapped_input,
                            "current_data_keys": list(current_data.keys())
                        }
                    )
                    
                    # Execute generator
                    try:
                        result = generator(**mapped_input)
                        
                        # Extract result data
                        result_data = {}
                        if hasattr(result, '__dict__'):
                            for key, value in result.__dict__.items():
                                if not key.startswith('_'):
                                    result_data[key] = value
                        
                        # Apply output mapping
                        if output_mapping:
                            for source_field, target_field in output_mapping.items():
                                if source_field in result_data:
                                    current_data[target_field] = result_data[source_field]
                        else:
                            # If no output mapping, add all result fields
                            current_data.update(result_data)
                        
                        # Store step result
                        step_result = {
                            "step": step_idx + 1,
                            "generator": step_name,
                            "input": mapped_input,
                            "output": result_data,
                            "success": True
                        }
                        chain_results.append(step_result)
                        
                        print(f"     ✅ Step completed: {list(result_data.keys())}")
                        
                    except Exception as e:
                        step_result = {
                            "step": step_idx + 1,
                            "generator": step_name,
                            "input": mapped_input,
                            "error": str(e),
                            "success": False
                        }
                        chain_results.append(step_result)
                        
                        print(f"     ❌ Step failed: {str(e)}")
                        raise
                
                # Log chain completion
                self.tracer.log_step(
                    self.chain_id,
                    f"Chain Completed: {self.name}",
                    f"Generator chain '{self.name}' completed successfully",
                    {
                        "chain_name": self.name,
                        "total_steps": len(self.sequence),
                        "final_data": current_data,
                        "step_results": chain_results,
                        "shared_model": SHARED_MODEL_CONFIG["model"]
                    }
                )
                
                print(f"\n🎉 Chain '{self.name}' completed successfully!")
                print(f"   Final Output: {json.dumps(current_data, indent=2)}")
                
                return {
                    "final_data": current_data,
                    "step_results": chain_results,
                    "chain_name": self.name
                }
        
        chain = GeneratorChain(chain_name, generator_sequence, self, self.reasoning_tracer)
        self.generator_chains.append(chain)
        
        print(f"🔗 Created generator chain: {chain_name}")
        return chain
    
    def get_reasoning_trace(self, format_type: str = "detailed") -> Dict[str, Any]:
        """Get the complete reasoning trace for all generator interactions."""
        
        trace_data = self.reasoning_tracer.get_trace()
        
        if format_type == "summary":
            # Provide a summary view
            summary = {
                "total_steps": len(trace_data["steps"]),
                "generators_used": list(self.generators.keys()),
                "chains_created": len(self.generator_chains),
                "shared_model": SHARED_MODEL_CONFIG["model"],
                "trace_summary": []
            }
            
            for step in trace_data["steps"]:
                summary["trace_summary"].append({
                    "step_id": step["step_id"],
                    "title": step["title"],
                    "timestamp": step["metadata"].get("timestamp", ""),
                    "success": step["metadata"].get("success", True)
                })
            
            return summary
        
        elif format_type == "detailed":
            # Provide detailed view with all data
            return {
                "reasoning_trace": trace_data,
                "generators": self.generators,
                "signatures": {name: str(sig) for name, sig in self.signatures.items()},
                "chains": [{"name": chain.name, "steps": len(chain.sequence)} for chain in self.generator_chains],
                "shared_model_config": SHARED_MODEL_CONFIG
            }
        
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
    
    def save_trace_to_file(self, filepath: str):
        """Save the complete reasoning trace to a file."""
        trace_data = self.get_reasoning_trace("detailed")
        
        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)
        
        print(f"💾 Reasoning trace saved to: {filepath}")
        
        self.reasoning_tracer.log_step(
            "trace_export",
            "Trace Saved",
            f"Complete reasoning trace exported to {filepath}",
            {"filepath": filepath, "timestamp": datetime.now().isoformat()}
        )

# Global registry instance
dspy_registry = DSPyGeneratorRegistry()

# Convenience functions for easy use
def register_signature(name: str, signature_class: Type[dspy.Signature]):
    """Register a custom DSPy signature."""
    return dspy_registry.register_signature(name, signature_class)

def create_generator(name: str, signature_name: str, generator_type: str = "ChainOfThought"):
    """Create a new DSPy generator."""
    return dspy_registry.create_generator(name, signature_name, generator_type)

def create_chain(chain_name: str, generator_sequence: List[Dict[str, Any]]):
    """Create a generator chain."""
    return dspy_registry.create_generator_chain(chain_name, generator_sequence)

def get_trace(format_type: str = "detailed"):
    """Get reasoning trace."""
    return dspy_registry.get_reasoning_trace(format_type)

def save_trace(filepath: str):
    """Save reasoning trace to file."""
    return dspy_registry.save_trace_to_file(filepath)

# Example usage and built-in signatures
def setup_example_generators():
    """Setup some example generators to demonstrate the system."""
    
    # Define example signatures
    class DocumentAnalyzer(dspy.Signature):
        """Analyze a document and extract key information."""
        document = dspy.InputField(desc="The document text to analyze")
        key_topics = dspy.OutputField(desc="List of key topics found in the document")
        summary = dspy.OutputField(desc="Brief summary of the document")
        entities = dspy.OutputField(desc="Important entities mentioned")
    
    class QuestionGenerator(dspy.Signature):
        """Generate questions based on document topics."""
        topics = dspy.InputField(desc="List of key topics")
        summary = dspy.InputField(desc="Document summary")
        questions = dspy.OutputField(desc="List of relevant questions")
    
    class AnswerValidator(dspy.Signature):
        """Validate if questions can be answered from the document."""
        questions = dspy.InputField(desc="List of questions to validate")
        document = dspy.InputField(desc="Original document text")
        validated_questions = dspy.OutputField(desc="Questions that can be answered")
        unanswerable_questions = dspy.OutputField(desc="Questions that cannot be answered")
    
    # Register signatures
    register_signature("DocumentAnalyzer", DocumentAnalyzer)
    register_signature("QuestionGenerator", QuestionGenerator)
    register_signature("AnswerValidator", AnswerValidator)
    
    # Create generators
    create_generator("analyzer", "DocumentAnalyzer", "ChainOfThought")
    create_generator("question_gen", "QuestionGenerator", "ChainOfThought")
    create_generator("validator", "AnswerValidator", "ChainOfThought")
    
    # Create a generator chain
    chain_config = [
        {
            "generator": "analyzer",
            "input_mapping": {"document": "document"},
            "output_mapping": {}
        },
        {
            "generator": "question_gen", 
            "input_mapping": {"topics": "key_topics", "summary": "summary"},
            "output_mapping": {}
        },
        {
            "generator": "validator",
            "input_mapping": {"questions": "questions", "document": "document"},
            "output_mapping": {}
        }
    ]
    
    return create_chain("document_qa_pipeline", chain_config)

if __name__ == "__main__":
    print("🚀 DSPy Generator System with Reasoning Traces")
    print("=" * 50)
    
    # Setup example generators
    example_chain = setup_example_generators()
    
    # Example usage
    sample_document = """
    The CFA Level 2 exam focuses on asset valuation and analysis. Key topics include:
    - Corporate finance and capital budgeting
    - Equity valuation methods
    - Fixed income analysis
    - Derivatives and risk management
    - Portfolio management theory
    """
    
    # Execute the chain
    result = example_chain.execute({"document": sample_document})
    
    # Show reasoning trace
    trace = get_trace("summary")
    print(f"\n📊 Reasoning Trace Summary:")
    print(f"   Total steps: {trace['total_steps']}")
    print(f"   Generators used: {', '.join(trace['generators_used'])}")
    print(f"   Shared model: {trace['shared_model']}")
