# ==============================================================================
# dev_tool_optimizer.py
#
# A full command-line tool to build a self-optimizing AI Developer Assistant.
# This unified script uses a configurable Ollama model for all workflows.
#
# WORKFLOWS:
# 1. optimize: A one-stage GEPA pipeline to evolve the prompts of the Router system.
# 2. tdd: A Test-Driven Development pipeline to generate modules from tests.
#
# USAGE:
#   > python dev_tool_optimizer.py optimize --ollama_model "llama3"
#   > python dev_tool_optimizer.py tdd --ollama_model "llama3"
# ==============================================================================

import dspy
import argparse
import logging
import os
import re
import inspect
from dspy.teleprompt.gepa import GEPA
from dspy.teleprompt import BootstrapFewShot

# --- 1. Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
GEPA_OUTPUT_PATH = "gepa_optimized_devtool.json"
TDD_OUTPUT_PATH = "tdd_generated_module.json"

# ==============================================================================
# SECTION A: DEV TOOL ROUTER SYSTEM (for the 'optimize' command)
# ==============================================================================

# --- Mock Data for the Dev Tool ---
MOCK_FILE_SYSTEM = {
    "auth_service.py": "import flask\ndef login(user, password):\n    if not user or not password:\n        return None\n    return 'Login successful'",
}
MOCK_LOG_DATA = {
    "app.log": "ERROR:root:Exception on /api/login [POST]\nAttributeError: 'NoneType' object has no attribute 'is_valid'",
}

# --- Signatures and Modules for the Dev Tool ---
class ProposeDiffSignature(dspy.Signature):
    """Given a task, context, and a plan, propose a code modification as a diff."""
    task_description = dspy.InputField()
    context = dspy.InputField(desc="Relevant code snippets or log entries.")
    modification_plan = dspy.InputField()
    code_diff = dspy.OutputField(desc="A code diff in the standard git diff format.")

class CodeAnalysisPipeline(dspy.Module):
    def __init__(self): super().__init__(); self.plan = dspy.ChainOfThought("task_description, code_context -> modification_plan"); self.propose_diff = dspy.Predict(ProposeDiffSignature)
    def forward(self, task_description): code_context = MOCK_FILE_SYSTEM["auth_service.py"]; plan = self.plan(task_description=task_description, code_context=code_context).modification_plan; return self.propose_diff(task_description=task_description, context=code_context, modification_plan=plan)

class LogAnalysisPipeline(dspy.Module):
    def __init__(self): super().__init__(); self.plan = dspy.ChainOfThought("task_description, log_context -> root_cause, modification_plan"); self.propose_diff = dspy.Predict(ProposeDiffSignature)
    def forward(self, task_description): log_context = MOCK_LOG_DATA["app.log"]; analysis = self.plan(task_description=task_description, log_context=log_context); full_context = f"LOGS:\n{log_context}\n\nCODE:\n{MOCK_FILE_SYSTEM['auth_service.py']}"; return self.propose_diff(task_description=task_description, context=full_context, modification_plan=analysis.modification_plan)

class DirectCodingPipeline(dspy.Module):
    def __init__(self): super().__init__(); self.propose_diff = dspy.ChainOfThought("task_description -> code_diff")
    def forward(self, task_description): return self.propose_diff(task_description=task_description)

class DevToolRouterSignature(dspy.Signature):
    """Select the best strategy: 'CodeAnalysisPipeline', 'LogAnalysisPipeline', or 'DirectCodingPipeline'."""
    task_description = dspy.InputField()
    route = dspy.OutputField(desc="Must be one of the available pipeline options.")

class RouterSystem(dspy.Module):
    def __init__(self):
        super().__init__()
        self.router = dspy.Predict(DevToolRouterSignature)
        self.specialist_modules = {"CodeAnalysisPipeline": CodeAnalysisPipeline(), "LogAnalysisPipeline": LogAnalysisPipeline(), "DirectCodingPipeline": DirectCodingPipeline()}
    def forward(self, task_description):
        prediction = self.router(task_description=task_description)
        chosen_route = prediction.route
        dspy.Suggest(chosen_route in self.specialist_modules, f"Router chose an invalid route: {chosen_route}")
        return self.specialist_modules[chosen_route](task_description=task_description)

# --- Training Data for the Dev Tool ---
dev_tool_trainset = [
    dspy.Example(task_description="The login function crashes on null user. Check logs and fix.", gold_route="LogAnalysisPipeline", gold_diff="...").with_inputs("task_description"),
    dspy.Example(task_description="Add a health check endpoint.", gold_route="DirectCodingPipeline", gold_diff="...").with_inputs("task_description"),
]

# ==============================================================================
# SECTION B: TEST-DRIVEN DEVELOPMENT (TDD-AI) WORKFLOW (for the 'tdd' command)
# ==============================================================================

class TestEmailExtractor:
    """A collection of test cases that define the behavior of an email extractor."""
    def test_simple_case(self):
        text = "You can reach me at test.user@dspy.ai for more info."
        assert self.extract_email(text) == "test.user@dspy.ai"

    def test_with_name(self):
        text = "The user is John Doe <john.d@company.org>"
        assert self.extract_email(text) == "john.d@company.org"
        
    def test_no_email(self):
        text = "There is no contact information here."
        assert self.extract_email(text) == "null"
        
    def extract_email(self, text: str) -> str:
        # This is a placeholder for DSPy to implement.
        pass

def convert_tests_to_dspy_spec(test_class):
    """Parses a test class and converts its test cases into a DSPy specification."""
    print(f"--- 📝 Converting tests from '{test_class.__name__}' into a DSPy specification... ---")
    signature = dspy.Signature("text -> email_address", "Extract an email address from text. If no email is found, return 'null'.")
    trainset = []
    test_methods = [member for member in inspect.getmembers(test_class, predicate=inspect.isfunction) if member[0].startswith('test_')]
    for name, method in test_methods:
        source = inspect.getsource(method)
        text_match = re.search(r'text = "(.*?)"', source)
        expected_match = re.search(r'assert.*?== "(.*?)"', source)
        if text_match and expected_match:
            text = text_match.group(1)
            expected = expected_match.group(1)
            trainset.append(dspy.Example(text=text, email_address=expected).with_inputs("text"))
            print(f"  - Found test '{name}': input='{text[:30]}...', expected='{expected}'")
    def metric(gold, pred, trace=None): return gold.email_address == pred.email_address
    return signature, trainset, metric

# ==============================================================================
# SECTION C: WORKFLOW ORCHESTRATION & CLI
# ==============================================================================

def run_gepa_optimization(args):
    """Runs the GEPA prompt evolution pipeline."""
    print("\n--- 🚀 Starting GEPA Optimization for the Dev Assistant ---")
    
    def gepa_feedback_metric(gold, pred, trace, pred_name, pred_trace):
        route_correct = gold.gold_route == trace[0][2]['route']
        answer_good = "--- a/" in pred.code_diff
        score = (0.5 if route_correct else 0.0) + (0.5 if answer_good else 0.0)
        return dspy.Prediction(score=score, feedback=f"Route correct: {route_correct}, Diff good: {answer_good}")
    
    optimizer = GEPA(metric=gepa_feedback_metric, reflection_lm=dspy.settings.lm, auto="light")
    compiled_system = optimizer.compile(RouterSystem(), trainset=dev_tool_trainset)
    
    compiled_system.save(GEPA_OUTPUT_PATH)
    print(f"\n--- ✅ GEPA Finished! Optimized program saved to '{GEPA_OUTPUT_PATH}' ---")
    
    print("\n--- 🧪 Testing the optimized system ---")
    test_task = "The login service is failing on null inputs. Review the logs and patch the vulnerability."
    print(f"Test Task: {test_task}")
    response = compiled_system(task_description=test_task)
    print("Proposed Diff:")
    print(response.code_diff)

def run_tdd_workflow(args):
    """Runs the Test-Driven Development workflow to generate a module from tests."""
    print("\n--- 💡 Starting Test-Driven Development (TDD-AI) Workflow ---")
    
    signature, trainset, metric = convert_tests_to_dspy_spec(TestEmailExtractor)
    
    class GeneratedModule(dspy.Module):
        def __init__(self):
            super().__init__(); self.generate = dspy.Predict(signature)
        def forward(self, text):
            return self.generate(text=text)

    optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
    compiled_module = optimizer.compile(GeneratedModule(), trainset=trainset)
    
    compiled_module.save(TDD_OUTPUT_PATH)
    print(f"\n--- ✅ TDD Finished! Compiled module saved to '{TDD_OUTPUT_PATH}' ---")
    
    print("\n--- 🧪 Testing the generated module ---")
    test_input = "Please forward this to human.resources@mycorp.com, thanks!"
    print(f"Input: '{test_input}'")
    print(f"Extracted Email: {compiled_module(text=test_input).email_address}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An advanced, multi-purpose DSPy development and optimization tool using Ollama.")
    
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Subparser for the 'optimize' command ---
    parser_optimize = subparsers.add_parser('optimize', help="Run the GEPA optimization on the main Dev Assistant.")
    parser_optimize.add_argument("--ollama_model", type=str, default="llama3",
                                 help="The Ollama model to use for the optimize workflow.")

    # --- Subparser for the 'tdd' command ---
    parser_tdd = subparsers.add_parser('tdd', help="Generate a new DSPy module from a class of test cases.")
    parser_tdd.add_argument("--ollama_model", type=str, default="llama3",
                            help="The Ollama model to use for the TDD workflow.")
    
    args = parser.parse_args()

    # --- Unified Ollama Model Configuration ---
    print(f"--- ⚙️ Configuring Ollama Model: {args.ollama_model} ---")
    try:
        ollama_lm = dspy.Ollama(model=args.ollama_model)
        dspy.settings.configure(lm=ollama_lm)
    except Exception as e:
        print(f"\nCould not configure Ollama. Please check the following:")
        print("1. Ollama is installed and running on your system.")
        print(f"2. You have pulled the specified model via 'ollama pull {args.ollama_model}'.")
        print(f"Error: {e}")
        exit()

    # --- Dispatch to the correct workflow ---
    if args.command == 'optimize':
        run_gepa_optimization(args)
    elif args.command == 'tdd':
        run_tdd_workflow(args)