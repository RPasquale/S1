import React, { useState, useEffect } from 'react';
import './DSPyFunctionCreator.css';

interface DSPyFunction {
  id: string;
  name: string;
  description: string;
  signature: {
    inputs: Array<{
      name: string;
      type: string;
      description: string;
      required: boolean;
    }>;
    outputs: Array<{
      name: string;
      type: string;
      description: string;
    }>;
  };
  implementation: string;
  category: string;
  tags: string[];
  examples: Array<{
    inputs: Record<string, any>;
    expected_output: Record<string, any>;
    description: string;
  }>;
  performance_metrics: {
    accuracy?: number;
    latency?: number;
    token_usage?: number;
  };
  training_data: Array<{
    input: Record<string, any>;
    output: Record<string, any>;
    feedback: 'positive' | 'negative' | 'neutral';
  }>;
}

interface DSPyFunctionCreatorProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (func: DSPyFunction) => void;
  editingFunction?: DSPyFunction | null;
}

const DSPyFunctionCreator: React.FC<DSPyFunctionCreatorProps> = ({
  isOpen,
  onClose,
  onSave,
  editingFunction
}) => {
  const [activeTab, setActiveTab] = useState('definition');
  const [functionData, setFunctionData] = useState<DSPyFunction>({
    id: '',
    name: '',
    description: '',
    signature: {
      inputs: [{ name: '', type: 'string', description: '', required: true }],
      outputs: [{ name: '', type: 'string', description: '' }]
    },
    implementation: '',
    category: 'general',
    tags: [],
    examples: [],
    performance_metrics: {},
    training_data: []
  });

  const [currentTag, setCurrentTag] = useState('');
  const [currentExample, setCurrentExample] = useState({
    inputs: {},
    expected_output: {},
    description: ''
  });
  const [testResults, setTestResults] = useState<any>(null);
  const [isDeploying, setIsDeploying] = useState(false);

  const categories = [
    'general',
    'text_analysis',
    'question_answering',
    'summarization',
    'classification',
    'generation',
    'reasoning',
    'data_extraction',
    'agent_coordination',
    'self_improvement'
  ];

  const dataTypes = [
    'string',
    'number',
    'boolean',
    'array',
    'object',
    'document',
    'query',
    'response',
    'embedding'
  ];

  useEffect(() => {
    if (editingFunction) {
      setFunctionData(editingFunction);
    } else {
      // Reset for new function
      setFunctionData({
        id: '',
        name: '',
        description: '',
        signature: {
          inputs: [{ name: '', type: 'string', description: '', required: true }],
          outputs: [{ name: '', type: 'string', description: '' }]
        },
        implementation: '',
        category: 'general',
        tags: [],
        examples: [],
        performance_metrics: {},
        training_data: []
      });
    }
  }, [editingFunction]);

  const handleSave = () => {
    if (!functionData.name || !functionData.description) {
      alert('Please provide at least a name and description');
      return;
    }

    const functionToSave = {
      ...functionData,
      id: functionData.id || `dspy_func_${Date.now()}`
    };

    onSave(functionToSave);
  };

  const addInput = () => {
    setFunctionData(prev => ({
      ...prev,
      signature: {
        ...prev.signature,
        inputs: [...prev.signature.inputs, { name: '', type: 'string', description: '', required: true }]
      }
    }));
  };

  const removeInput = (index: number) => {
    setFunctionData(prev => ({
      ...prev,
      signature: {
        ...prev.signature,
        inputs: prev.signature.inputs.filter((_, i) => i !== index)
      }
    }));
  };

  const addOutput = () => {
    setFunctionData(prev => ({
      ...prev,
      signature: {
        ...prev.signature,
        outputs: [...prev.signature.outputs, { name: '', type: 'string', description: '' }]
      }
    }));
  };

  const removeOutput = (index: number) => {
    setFunctionData(prev => ({
      ...prev,
      signature: {
        ...prev.signature,
        outputs: prev.signature.outputs.filter((_, i) => i !== index)
      }
    }));
  };

  const addTag = () => {
    if (currentTag && !functionData.tags.includes(currentTag)) {
      setFunctionData(prev => ({
        ...prev,
        tags: [...prev.tags, currentTag]
      }));
      setCurrentTag('');
    }
  };

  const removeTag = (tag: string) => {
    setFunctionData(prev => ({
      ...prev,
      tags: prev.tags.filter(t => t !== tag)
    }));
  };

  const addExample = () => {
    if (currentExample.description) {
      setFunctionData(prev => ({
        ...prev,
        examples: [...prev.examples, { ...currentExample }]
      }));
      setCurrentExample({
        inputs: {},
        expected_output: {},
        description: ''
      });
    }
  };

  const testFunction = async () => {
    try {
      const response = await fetch('/api/dspy/test-function', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          function_definition: functionData,
          test_cases: functionData.examples
        }),
      });

      const results = await response.json();
      setTestResults(results);
    } catch (error) {
      console.error('Error testing function:', error);
      setTestResults({ error: 'Failed to test function' });
    }
  };

  const deployFunction = async () => {
    setIsDeploying(true);
    try {
      const response = await fetch('/api/dspy/deploy-function', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(functionData),
      });

      if (response.ok) {
        alert('Function deployed successfully!');
        handleSave();
      } else {
        alert('Failed to deploy function');
      }
    } catch (error) {
      console.error('Error deploying function:', error);
      alert('Error deploying function');
    } finally {
      setIsDeploying(false);
    }
  };

  const generateImplementation = async () => {
    try {
      const response = await fetch('/api/dspy/generate-implementation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: functionData.name,
          description: functionData.description,
          signature: functionData.signature,
          examples: functionData.examples
        }),
      });

      const result = await response.json();
      if (result.implementation) {
        setFunctionData(prev => ({
          ...prev,
          implementation: result.implementation
        }));
      }
    } catch (error) {
      console.error('Error generating implementation:', error);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="dspy-function-creator-overlay">
      <div className="dspy-function-creator">
        <div className="dspy-function-creator-header">
          <h2>{editingFunction ? 'Edit DSPy Function' : 'Create DSPy Function'}</h2>
          <button onClick={onClose} className="close-btn">×</button>
        </div>

        <div className="dspy-function-creator-tabs">
          <button
            className={activeTab === 'definition' ? 'active' : ''}
            onClick={() => setActiveTab('definition')}
          >
            Definition
          </button>
          <button
            className={activeTab === 'signature' ? 'active' : ''}
            onClick={() => setActiveTab('signature')}
          >
            Signature
          </button>
          <button
            className={activeTab === 'implementation' ? 'active' : ''}
            onClick={() => setActiveTab('implementation')}
          >
            Implementation
          </button>
          <button
            className={activeTab === 'examples' ? 'active' : ''}
            onClick={() => setActiveTab('examples')}
          >
            Examples
          </button>
          <button
            className={activeTab === 'testing' ? 'active' : ''}
            onClick={() => setActiveTab('testing')}
          >
            Testing
          </button>
          <button
            className={activeTab === 'training' ? 'active' : ''}
            onClick={() => setActiveTab('training')}
          >
            Training Data
          </button>
        </div>

        <div className="dspy-function-creator-content">
          {activeTab === 'definition' && (
            <div className="definition-tab">
              <div className="form-group">
                <label>Function Name</label>
                <input
                  type="text"
                  value={functionData.name}
                  onChange={(e) => setFunctionData(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="e.g., DocumentSummarizer"
                />
              </div>

              <div className="form-group">
                <label>Description</label>
                <textarea
                  value={functionData.description}
                  onChange={(e) => setFunctionData(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Describe what this function does and how it helps the agent..."
                  rows={4}
                />
              </div>

              <div className="form-group">
                <label>Category</label>
                <select
                  value={functionData.category}
                  onChange={(e) => setFunctionData(prev => ({ ...prev, category: e.target.value }))}
                >
                  {categories.map(cat => (
                    <option key={cat} value={cat}>{cat.replace('_', ' ')}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Tags</label>
                <div className="tags-input">
                  <input
                    type="text"
                    value={currentTag}
                    onChange={(e) => setCurrentTag(e.target.value)}
                    placeholder="Add tag..."
                    onKeyPress={(e) => e.key === 'Enter' && addTag()}
                  />
                  <button onClick={addTag}>Add</button>
                </div>
                <div className="tags-list">
                  {functionData.tags.map(tag => (
                    <span key={tag} className="tag">
                      {tag}
                      <button onClick={() => removeTag(tag)}>×</button>
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'signature' && (
            <div className="signature-tab">
              <div className="signature-section">
                <h3>Input Parameters</h3>
                {functionData.signature.inputs.map((input, index) => (
                  <div key={index} className="signature-item">
                    <input
                      type="text"
                      placeholder="Parameter name"
                      value={input.name}
                      onChange={(e) => {
                        const newInputs = [...functionData.signature.inputs];
                        newInputs[index].name = e.target.value;
                        setFunctionData(prev => ({
                          ...prev,
                          signature: { ...prev.signature, inputs: newInputs }
                        }));
                      }}
                    />
                    <select
                      value={input.type}
                      onChange={(e) => {
                        const newInputs = [...functionData.signature.inputs];
                        newInputs[index].type = e.target.value;
                        setFunctionData(prev => ({
                          ...prev,
                          signature: { ...prev.signature, inputs: newInputs }
                        }));
                      }}
                    >
                      {dataTypes.map(type => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                    <input
                      type="text"
                      placeholder="Description"
                      value={input.description}
                      onChange={(e) => {
                        const newInputs = [...functionData.signature.inputs];
                        newInputs[index].description = e.target.value;
                        setFunctionData(prev => ({
                          ...prev,
                          signature: { ...prev.signature, inputs: newInputs }
                        }));
                      }}
                    />
                    <label>
                      <input
                        type="checkbox"
                        checked={input.required}
                        onChange={(e) => {
                          const newInputs = [...functionData.signature.inputs];
                          newInputs[index].required = e.target.checked;
                          setFunctionData(prev => ({
                            ...prev,
                            signature: { ...prev.signature, inputs: newInputs }
                          }));
                        }}
                      />
                      Required
                    </label>
                    <button onClick={() => removeInput(index)}>Remove</button>
                  </div>
                ))}
                <button onClick={addInput} className="add-btn">Add Input</button>
              </div>

              <div className="signature-section">
                <h3>Output Parameters</h3>
                {functionData.signature.outputs.map((output, index) => (
                  <div key={index} className="signature-item">
                    <input
                      type="text"
                      placeholder="Output name"
                      value={output.name}
                      onChange={(e) => {
                        const newOutputs = [...functionData.signature.outputs];
                        newOutputs[index].name = e.target.value;
                        setFunctionData(prev => ({
                          ...prev,
                          signature: { ...prev.signature, outputs: newOutputs }
                        }));
                      }}
                    />
                    <select
                      value={output.type}
                      onChange={(e) => {
                        const newOutputs = [...functionData.signature.outputs];
                        newOutputs[index].type = e.target.value;
                        setFunctionData(prev => ({
                          ...prev,
                          signature: { ...prev.signature, outputs: newOutputs }
                        }));
                      }}
                    >
                      {dataTypes.map(type => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                    <input
                      type="text"
                      placeholder="Description"
                      value={output.description}
                      onChange={(e) => {
                        const newOutputs = [...functionData.signature.outputs];
                        newOutputs[index].description = e.target.value;
                        setFunctionData(prev => ({
                          ...prev,
                          signature: { ...prev.signature, outputs: newOutputs }
                        }));
                      }}
                    />
                    <button onClick={() => removeOutput(index)}>Remove</button>
                  </div>
                ))}
                <button onClick={addOutput} className="add-btn">Add Output</button>
              </div>
            </div>
          )}

          {activeTab === 'implementation' && (
            <div className="implementation-tab">
              <div className="implementation-header">
                <h3>DSPy Implementation</h3>
                <button onClick={generateImplementation} className="generate-btn">
                  Generate Implementation
                </button>
              </div>
              <textarea
                value={functionData.implementation}
                onChange={(e) => setFunctionData(prev => ({ ...prev, implementation: e.target.value }))}
                placeholder="Enter DSPy function implementation..."
                rows={20}
                className="implementation-textarea"
              />
              <div className="implementation-tips">
                <h4>Implementation Tips:</h4>
                <ul>
                  <li>Use dspy.Signature to define the function interface</li>
                  <li>Implement with dspy.ChainOfThought or dspy.Predict</li>
                  <li>Include proper error handling and validation</li>
                  <li>Consider using dspy.Module for complex functions</li>
                </ul>
              </div>
            </div>
          )}

          {activeTab === 'examples' && (
            <div className="examples-tab">
              <h3>Function Examples</h3>
              <div className="example-form">
                <h4>Add New Example</h4>
                <div className="form-group">
                  <label>Description</label>
                  <input
                    type="text"
                    value={currentExample.description}
                    onChange={(e) => setCurrentExample(prev => ({ ...prev, description: e.target.value }))}
                    placeholder="Describe this example..."
                  />
                </div>
                <div className="form-group">
                  <label>Input Data (JSON)</label>
                  <textarea
                    value={JSON.stringify(currentExample.inputs, null, 2)}
                    onChange={(e) => {
                      try {
                        const inputs = JSON.parse(e.target.value);
                        setCurrentExample(prev => ({ ...prev, inputs }));
                      } catch (err) {
                        // Handle invalid JSON
                      }
                    }}
                    placeholder='{"param1": "value1", "param2": "value2"}'
                    rows={4}
                  />
                </div>
                <div className="form-group">
                  <label>Expected Output (JSON)</label>
                  <textarea
                    value={JSON.stringify(currentExample.expected_output, null, 2)}
                    onChange={(e) => {
                      try {
                        const expected_output = JSON.parse(e.target.value);
                        setCurrentExample(prev => ({ ...prev, expected_output }));
                      } catch (err) {
                        // Handle invalid JSON
                      }
                    }}
                    placeholder='{"result": "expected result"}'
                    rows={4}
                  />
                </div>
                <button onClick={addExample} className="add-btn">Add Example</button>
              </div>

              <div className="examples-list">
                <h4>Current Examples</h4>
                {functionData.examples.map((example, index) => (
                  <div key={index} className="example-item">
                    <h5>{example.description}</h5>
                    <div className="example-data">
                      <div>
                        <strong>Input:</strong>
                        <pre>{JSON.stringify(example.inputs, null, 2)}</pre>
                      </div>
                      <div>
                        <strong>Expected Output:</strong>
                        <pre>{JSON.stringify(example.expected_output, null, 2)}</pre>
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        const newExamples = functionData.examples.filter((_, i) => i !== index);
                        setFunctionData(prev => ({ ...prev, examples: newExamples }));
                      }}
                      className="remove-btn"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'testing' && (
            <div className="testing-tab">
              <h3>Function Testing</h3>
              <div className="testing-controls">
                <button onClick={testFunction} className="test-btn">Run Tests</button>
                <button onClick={deployFunction} disabled={isDeploying} className="deploy-btn">
                  {isDeploying ? 'Deploying...' : 'Deploy Function'}
                </button>
              </div>

              {testResults && (
                <div className="test-results">
                  <h4>Test Results</h4>
                  <pre>{JSON.stringify(testResults, null, 2)}</pre>
                </div>
              )}

              <div className="performance-metrics">
                <h4>Performance Metrics</h4>
                <div className="metrics-grid">
                  <div className="metric">
                    <label>Accuracy (%)</label>
                    <input
                      type="number"
                      value={functionData.performance_metrics.accuracy || ''}
                      onChange={(e) => setFunctionData(prev => ({
                        ...prev,
                        performance_metrics: {
                          ...prev.performance_metrics,
                          accuracy: parseFloat(e.target.value) || undefined
                        }
                      }))}
                    />
                  </div>
                  <div className="metric">
                    <label>Latency (ms)</label>
                    <input
                      type="number"
                      value={functionData.performance_metrics.latency || ''}
                      onChange={(e) => setFunctionData(prev => ({
                        ...prev,
                        performance_metrics: {
                          ...prev.performance_metrics,
                          latency: parseFloat(e.target.value) || undefined
                        }
                      }))}
                    />
                  </div>
                  <div className="metric">
                    <label>Token Usage</label>
                    <input
                      type="number"
                      value={functionData.performance_metrics.token_usage || ''}
                      onChange={(e) => setFunctionData(prev => ({
                        ...prev,
                        performance_metrics: {
                          ...prev.performance_metrics,
                          token_usage: parseFloat(e.target.value) || undefined
                        }
                      }))}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'training' && (
            <div className="training-tab">
              <h3>Training Data</h3>
              <p>Collect training data to improve this function's performance over time.</p>
              
              <div className="training-data-list">
                {functionData.training_data.map((data, index) => (
                  <div key={index} className="training-data-item">
                    <div className="training-input">
                      <strong>Input:</strong>
                      <pre>{JSON.stringify(data.input, null, 2)}</pre>
                    </div>
                    <div className="training-output">
                      <strong>Output:</strong>
                      <pre>{JSON.stringify(data.output, null, 2)}</pre>
                    </div>
                    <div className="training-feedback">
                      <strong>Feedback:</strong>
                      <span className={`feedback ${data.feedback}`}>{data.feedback}</span>
                    </div>
                  </div>
                ))}
              </div>

              {functionData.training_data.length === 0 && (
                <div className="no-training-data">
                  <p>No training data collected yet. Training data will be automatically collected when this function is used in production.</p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="dspy-function-creator-footer">
          <button onClick={onClose} className="cancel-btn">Cancel</button>
          <button onClick={handleSave} className="save-btn">
            {editingFunction ? 'Update Function' : 'Save Function'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DSPyFunctionCreator;
