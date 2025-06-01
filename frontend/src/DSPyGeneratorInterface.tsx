import React, { useState, useEffect } from 'react';
import './DSPyGeneratorInterface.css';

interface Signature {
  name: string;
  description: string;
  input_fields: { [key: string]: string };
  output_fields: { [key: string]: string };
}

interface Generator {
  name: string;
  signature_name: string;
  type: string;
  call_count: number;
  created_at: string;
}

interface Chain {
  name: string;
  steps: number;
  generators: string[];
}

interface ChainStep {
  generator: string;
  input_mapping?: { [key: string]: string };
  output_mapping?: { [key: string]: string };
}

interface ReasoningTrace {
  total_steps: number;
  generators_used: string[];
  shared_model: string;
  trace_summary: Array<{
    step_id: string;
    title: string;
    timestamp: string;
    success: boolean;
  }>;
}

const DSPyGeneratorInterface: React.FC = () => {
  const [signatures, setSignatures] = useState<string[]>([]);
  const [generators, setGenerators] = useState<{ [key: string]: Generator }>({});
  const [chains, setChains] = useState<Chain[]>([]);
  const [sharedModel, setSharedModel] = useState<string>('');
  const [reasoningTrace, setReasoningTrace] = useState<ReasoningTrace | null>(null);
  
  // Form states
  const [activeTab, setActiveTab] = useState<'signatures' | 'generators' | 'chains' | 'execute' | 'trace'>('signatures');
  const [newSignature, setNewSignature] = useState<Signature>({
    name: '',
    description: '',
    input_fields: {},
    output_fields: {}
  });
  const [newGenerator, setNewGenerator] = useState({
    name: '',
    signature_name: '',
    generator_type: 'ChainOfThought'
  });
  const [newChain, setNewChain] = useState({
    name: '',
    steps: [] as ChainStep[]
  });
  const [executeForm, setExecuteForm] = useState({
    type: 'generator' as 'generator' | 'chain',
    target: '',
    inputs: {}
  });
  
  // Dynamic field inputs
  const [inputFieldName, setInputFieldName] = useState('');
  const [inputFieldDesc, setInputFieldDesc] = useState('');
  const [outputFieldName, setOutputFieldName] = useState('');
  const [outputFieldDesc, setOutputFieldDesc] = useState('');
  
  // Load data on component mount
  useEffect(() => {
    loadGenerators();
  }, []);

  const loadGenerators = async () => {
    try {
      const response = await fetch('/api/dspy/generators');
      const data = await response.json();
      
      setGenerators(data.generators || {});
      setSignatures(data.signatures || []);
      setChains(data.chains || []);
      setSharedModel(data.shared_model || 'Not configured');
    } catch (error) {
      console.error('Error loading generators:', error);
    }
  };

  const loadReasoningTrace = async () => {
    try {
      const response = await fetch('/api/dspy/trace?format_type=summary');
      const data = await response.json();
      setReasoningTrace(data);
    } catch (error) {
      console.error('Error loading reasoning trace:', error);
    }
  };

  const addInputField = () => {
    if (inputFieldName && inputFieldDesc) {
      setNewSignature(prev => ({
        ...prev,
        input_fields: {
          ...prev.input_fields,
          [inputFieldName]: inputFieldDesc
        }
      }));
      setInputFieldName('');
      setInputFieldDesc('');
    }
  };

  const addOutputField = () => {
    if (outputFieldName && outputFieldDesc) {
      setNewSignature(prev => ({
        ...prev,
        output_fields: {
          ...prev.output_fields,
          [outputFieldName]: outputFieldDesc
        }
      }));
      setOutputFieldName('');
      setOutputFieldDesc('');
    }
  };

  const createSignature = async () => {
    try {
      const response = await fetch('/api/dspy/signature', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSignature)
      });
      
      if (response.ok) {
        alert('Signature created successfully!');
        setNewSignature({ name: '', description: '', input_fields: {}, output_fields: {} });
        loadGenerators();
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      alert(`Error creating signature: ${error}`);
    }
  };

  const createGenerator = async () => {
    try {
      const response = await fetch('/api/dspy/generator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newGenerator)
      });
      
      if (response.ok) {
        alert('Generator created successfully!');
        setNewGenerator({ name: '', signature_name: '', generator_type: 'ChainOfThought' });
        loadGenerators();
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      alert(`Error creating generator: ${error}`);
    }
  };

  const createChain = async () => {
    try {
      const response = await fetch('/api/dspy/chain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chain_name: newChain.name,
          generator_sequence: newChain.steps
        })
      });
      
      if (response.ok) {
        alert('Chain created successfully!');
        setNewChain({ name: '', steps: [] });
        loadGenerators();
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      alert(`Error creating chain: ${error}`);
    }
  };

  const executeGenerator = async () => {
    try {
      const endpoint = executeForm.type === 'generator' ? '/api/dspy/execute' : '/api/dspy/chain/execute';
      const body = executeForm.type === 'generator' 
        ? { generator_name: executeForm.target, inputs: executeForm.inputs }
        : { chain_name: executeForm.target, initial_input: executeForm.inputs };
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`Execution completed! Check console for details.`);
        console.log('Execution result:', result);
        loadReasoningTrace(); // Refresh reasoning trace
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      alert(`Error executing: ${error}`);
    }
  };

  const addChainStep = () => {
    setNewChain(prev => ({
      ...prev,
      steps: [...prev.steps, { generator: '', input_mapping: {}, output_mapping: {} }]
    }));
  };

  const updateChainStep = (index: number, field: keyof ChainStep, value: any) => {
    setNewChain(prev => ({
      ...prev,
      steps: prev.steps.map((step, i) => 
        i === index ? { ...step, [field]: value } : step
      )
    }));
  };

  const saveReasoningTrace = async () => {
    try {
      const response = await fetch('/api/dspy/trace/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`Reasoning trace saved to: ${result.filepath}`);
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      alert(`Error saving trace: ${error}`);
    }
  };

  return (
    <div className="dspy-interface">
      <div className="header">
        <h2>🎯 DSPy Generator System</h2>
        <div className="model-info">
          <strong>Shared Model:</strong> {sharedModel}
        </div>
      </div>

      <div className="tabs">
        {['signatures', 'generators', 'chains', 'execute', 'trace'].map(tab => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab as any)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      <div className="tab-content">
        {activeTab === 'signatures' && (
          <div className="signatures-tab">
            <h3>Create Custom Signature</h3>
            <div className="form-group">
              <label>Name:</label>
              <input
                type="text"
                value={newSignature.name}
                onChange={(e) => setNewSignature(prev => ({ ...prev, name: e.target.value }))}
                placeholder="MyCustomSignature"
              />
            </div>
            
            <div className="form-group">
              <label>Description:</label>
              <textarea
                value={newSignature.description}
                onChange={(e) => setNewSignature(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Describe what this signature does..."
              />
            </div>

            <div className="fields-section">
              <h4>Input Fields</h4>
              <div className="field-input">
                <input
                  type="text"
                  placeholder="Field name"
                  value={inputFieldName}
                  onChange={(e) => setInputFieldName(e.target.value)}
                />
                <input
                  type="text"
                  placeholder="Field description"
                  value={inputFieldDesc}
                  onChange={(e) => setInputFieldDesc(e.target.value)}
                />
                <button onClick={addInputField}>Add Input Field</button>
              </div>
              <div className="field-list">
                {Object.entries(newSignature.input_fields).map(([name, desc]) => (
                  <div key={name} className="field-item">
                    <strong>{name}:</strong> {desc}
                  </div>
                ))}
              </div>
            </div>

            <div className="fields-section">
              <h4>Output Fields</h4>
              <div className="field-input">
                <input
                  type="text"
                  placeholder="Field name"
                  value={outputFieldName}
                  onChange={(e) => setOutputFieldName(e.target.value)}
                />
                <input
                  type="text"
                  placeholder="Field description"
                  value={outputFieldDesc}
                  onChange={(e) => setOutputFieldDesc(e.target.value)}
                />
                <button onClick={addOutputField}>Add Output Field</button>
              </div>
              <div className="field-list">
                {Object.entries(newSignature.output_fields).map(([name, desc]) => (
                  <div key={name} className="field-item">
                    <strong>{name}:</strong> {desc}
                  </div>
                ))}
              </div>
            </div>

            <button className="create-btn" onClick={createSignature}>
              Create Signature
            </button>

            <div className="existing-signatures">
              <h4>Existing Signatures</h4>
              {signatures.map(sig => (
                <div key={sig} className="signature-item">{sig}</div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'generators' && (
          <div className="generators-tab">
            <h3>Create Generator</h3>
            <div className="form-group">
              <label>Generator Name:</label>
              <input
                type="text"
                value={newGenerator.name}
                onChange={(e) => setNewGenerator(prev => ({ ...prev, name: e.target.value }))}
                placeholder="my_generator"
              />
            </div>

            <div className="form-group">
              <label>Signature:</label>
              <select
                value={newGenerator.signature_name}
                onChange={(e) => setNewGenerator(prev => ({ ...prev, signature_name: e.target.value }))}
              >
                <option value="">Select a signature</option>
                {signatures.map(sig => (
                  <option key={sig} value={sig}>{sig}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Generator Type:</label>
              <select
                value={newGenerator.generator_type}
                onChange={(e) => setNewGenerator(prev => ({ ...prev, generator_type: e.target.value }))}
              >
                <option value="ChainOfThought">Chain of Thought</option>
                <option value="Predict">Predict</option>
                <option value="ReAct">ReAct</option>
              </select>
            </div>

            <button className="create-btn" onClick={createGenerator}>
              Create Generator
            </button>

            <div className="existing-generators">
              <h4>Existing Generators</h4>
              {Object.entries(generators).map(([name, gen]) => (
                <div key={name} className="generator-item">
                  <strong>{name}</strong> ({gen.type}) - {gen.signature_name}
                  <br />
                  <small>Calls: {gen.call_count} | Created: {new Date(gen.created_at).toLocaleDateString()}</small>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'execute' && (
          <div className="execute-tab">
            <h3>Execute Generator or Chain</h3>
            <div className="form-group">
              <label>Type:</label>
              <select
                value={executeForm.type}
                onChange={(e) => setExecuteForm(prev => ({ ...prev, type: e.target.value as any }))}
              >
                <option value="generator">Generator</option>
                <option value="chain">Chain</option>
              </select>
            </div>

            <div className="form-group">
              <label>Target:</label>
              <select
                value={executeForm.target}
                onChange={(e) => setExecuteForm(prev => ({ ...prev, target: e.target.value }))}
              >
                <option value="">Select target</option>
                {executeForm.type === 'generator' 
                  ? Object.keys(generators).map(name => (
                      <option key={name} value={name}>{name}</option>
                    ))
                  : chains.map(chain => (
                      <option key={chain.name} value={chain.name}>{chain.name}</option>
                    ))
                }
              </select>
            </div>

            <div className="form-group">
              <label>Input Data (JSON):</label>
              <textarea
                value={JSON.stringify(executeForm.inputs, null, 2)}
                onChange={(e) => {
                  try {
                    setExecuteForm(prev => ({ ...prev, inputs: JSON.parse(e.target.value) }));
                  } catch {}
                }}
                placeholder='{"field1": "value1", "field2": "value2"}'
                rows={6}
              />
            </div>

            <button className="execute-btn" onClick={executeGenerator}>
              Execute
            </button>
          </div>
        )}

        {activeTab === 'trace' && (
          <div className="trace-tab">
            <h3>Reasoning Trace</h3>
            <div className="trace-actions">
              <button onClick={loadReasoningTrace}>Refresh Trace</button>
              <button onClick={saveReasoningTrace}>Save Trace to File</button>
            </div>

            {reasoningTrace && (
              <div className="trace-summary">
                <div className="trace-stats">
                  <div><strong>Total Steps:</strong> {reasoningTrace.total_steps}</div>
                  <div><strong>Generators Used:</strong> {reasoningTrace.generators_used.join(', ')}</div>
                  <div><strong>Shared Model:</strong> {reasoningTrace.shared_model}</div>
                </div>

                <div className="trace-steps">
                  <h4>Step-by-Step Execution</h4>
                  {reasoningTrace.trace_summary.map((step, index) => (
                    <div key={step.step_id} className={`trace-step ${step.success ? 'success' : 'error'}`}>
                      <div className="step-header">
                        <strong>Step {index + 1}:</strong> {step.title}
                      </div>
                      <div className="step-meta">
                        <small>{step.timestamp} | {step.success ? '✅ Success' : '❌ Error'}</small>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DSPyGeneratorInterface;
