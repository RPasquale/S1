import React, { useState, useEffect } from 'react';
import './TrainingModal.css';

interface TrainingModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface TrainingCapabilities {
  training_types: {
    [key: string]: {
      available: boolean;
      description: string;
      requirements: string[];
    };
  };
  supported_formats: string[];
  dependencies: {
    [key: string]: boolean;
  };
}

interface TrainingConfig {
  training_type: string;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  ppo_steps: number;
  base_model_path: string;
  documents: string[];
  file_paths: string[];
}

interface TrainingStatus {
  is_training: boolean;
  progress: number;
  status_message: string;
  model_path?: string;
  error?: string;
}

interface DSPySignature {
  name: string;
  description: string;
  input_fields: { name: string; description: string }[];
  output_fields: { name: string; description: string }[];
}

const TrainingModal: React.FC<TrainingModalProps> = ({ isOpen, onClose }) => {
  const [capabilities, setCapabilities] = useState<TrainingCapabilities | null>(null);
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    training_type: 'next_token_prediction',
    epochs: 3,
    learning_rate: 0.00005,
    batch_size: 4,
    ppo_steps: 100,
    base_model_path: '',
    documents: [],
    file_paths: []
  });
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState<'config' | 'dspy' | 'embeddings' | 'status'>('config');
  const [customSignatures, setCustomSignatures] = useState<DSPySignature[]>([]);
  const [documentText, setDocumentText] = useState('');

  // Predefined DSPy signatures
  const predefinedSignatures: DSPySignature[] = [
    {
      name: 'DocumentAnalyzer',
      description: 'Analyze document content for key insights and information',
      input_fields: [{ name: 'document', description: 'Document text to analyze' }],
      output_fields: [{ name: 'analysis', description: 'Comprehensive analysis of the document' }]
    },
    {
      name: 'QueryResponder', 
      description: 'Respond to queries using document context',
      input_fields: [
        { name: 'query', description: 'User query' },
        { name: 'context', description: 'Relevant document context' }
      ],
      output_fields: [{ name: 'response', description: 'Detailed response based on context' }]
    },
    {
      name: 'FactualQueryGenerator',
      description: 'Generate specific factual questions from documents',
      input_fields: [{ name: 'document', description: 'Text from a document' }],
      output_fields: [{ name: 'questions', description: 'Five specific factual questions' }]
    },
    {
      name: 'EntityExtractor',
      description: 'Extract important entities and concepts from documents',
      input_fields: [{ name: 'document', description: 'Text from a document' }],
      output_fields: [{ name: 'entities', description: 'List of important entities and concepts' }]
    }
  ];

  useEffect(() => {
    if (isOpen) {
      fetchCapabilities();
    }
  }, [isOpen]);

  const fetchCapabilities = async () => {
    try {
      const response = await fetch('/api/training/capabilities');
      if (response.ok) {
        const data = await response.json();
        setCapabilities(data);
      }
    } catch (error) {
      console.error('Error fetching training capabilities:', error);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    setUploading(true);
    const formData = new FormData();
    
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const response = await fetch('/api/upload-training-documents', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        setTrainingConfig(prev => ({
          ...prev,
          file_paths: [...prev.file_paths, ...data.file_paths]
        }));
      }
    } catch (error) {
      console.error('Error uploading files:', error);
    } finally {
      setUploading(false);
    }
  };

  const addDocumentText = () => {
    if (documentText.trim()) {
      setTrainingConfig(prev => ({
        ...prev,
        documents: [...prev.documents, documentText.trim()]
      }));
      setDocumentText('');
    }
  };

  const removeDocument = (index: number, type: 'documents' | 'file_paths') => {
    setTrainingConfig(prev => ({
      ...prev,
      [type]: prev[type].filter((_, i) => i !== index)
    }));
  };

  const startTraining = async () => {
    try {
      const response = await fetch('/api/training/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          training_type: trainingConfig.training_type,
          documents: trainingConfig.documents,
          file_paths: trainingConfig.file_paths,
          model_config: {
            epochs: trainingConfig.epochs,
            learning_rate: trainingConfig.learning_rate,
            batch_size: trainingConfig.batch_size,
            ppo_steps: trainingConfig.ppo_steps,
            base_model_path: trainingConfig.base_model_path
          }
        })
      });

      if (response.ok) {
        const data = await response.json();
        setTrainingStatus(data);
        setActiveTab('status');
      }
    } catch (error) {
      console.error('Error starting training:', error);
    }
  };

  const addCustomSignature = () => {
    const newSignature: DSPySignature = {
      name: 'CustomSignature',
      description: 'Custom signature description',
      input_fields: [{ name: 'input', description: 'Input description' }],
      output_fields: [{ name: 'output', description: 'Output description' }]
    };
    setCustomSignatures(prev => [...prev, newSignature]);
  };

  const updateSignature = (index: number, field: keyof DSPySignature, value: any) => {
    setCustomSignatures(prev => prev.map((sig, i) => 
      i === index ? { ...sig, [field]: value } : sig
    ));
  };

  const addSignatureField = (signatureIndex: number, fieldType: 'input_fields' | 'output_fields') => {
    setCustomSignatures(prev => prev.map((sig, i) => 
      i === signatureIndex ? {
        ...sig,
        [fieldType]: [...sig[fieldType], { name: '', description: '' }]
      } : sig
    ));
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content training-modal">
        <div className="modal-header">
          <h2>Advanced Model Training</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="training-tabs">
          <button 
            className={`tab ${activeTab === 'config' ? 'active' : ''}`}
            onClick={() => setActiveTab('config')}
          >
            Training Config
          </button>
          <button 
            className={`tab ${activeTab === 'dspy' ? 'active' : ''}`}
            onClick={() => setActiveTab('dspy')}
          >
            DSPy Signatures
          </button>
          <button 
            className={`tab ${activeTab === 'embeddings' ? 'active' : ''}`}
            onClick={() => setActiveTab('embeddings')}
          >
            Embeddings
          </button>
          <button 
            className={`tab ${activeTab === 'status' ? 'active' : ''}`}
            onClick={() => setActiveTab('status')}
          >
            Training Status
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'config' && (
            <div className="config-tab">
              <div className="form-section">
                <h3>Training Type</h3>
                <select 
                  value={trainingConfig.training_type}
                  onChange={(e) => setTrainingConfig(prev => ({ ...prev, training_type: e.target.value }))}
                >
                  {capabilities?.training_types && Object.entries(capabilities.training_types).map(([key, info]) => (
                    <option key={key} value={key} disabled={!info.available}>
                      {key.replace('_', ' ').toUpperCase()} 
                      {!info.available && ' (Not Available)'}
                    </option>
                  ))}
                </select>
                {capabilities?.training_types[trainingConfig.training_type] && (
                  <div className="training-type-info">
                    <p>{capabilities.training_types[trainingConfig.training_type].description}</p>
                    <p><strong>Requirements:</strong> {capabilities.training_types[trainingConfig.training_type].requirements.join(', ')}</p>
                  </div>
                )}
              </div>

              <div className="form-section">
                <h3>Training Parameters</h3>
                <div className="parameter-grid">
                  <div className="parameter-item">
                    <label>Epochs:</label>
                    <input 
                      type="number" 
                      value={trainingConfig.epochs}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) }))}
                      min="1" max="20"
                    />
                  </div>
                  <div className="parameter-item">
                    <label>Learning Rate:</label>
                    <input 
                      type="number" 
                      value={trainingConfig.learning_rate}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                      step="0.00001" min="0.00001" max="0.001"
                    />
                  </div>
                  <div className="parameter-item">
                    <label>Batch Size:</label>
                    <input 
                      type="number" 
                      value={trainingConfig.batch_size}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                      min="1" max="16"
                    />
                  </div>
                  {trainingConfig.training_type === 'reinforcement_learning' && (
                    <div className="parameter-item">
                      <label>PPO Steps:</label>
                      <input 
                        type="number" 
                        value={trainingConfig.ppo_steps}
                        onChange={(e) => setTrainingConfig(prev => ({ ...prev, ppo_steps: parseInt(e.target.value) }))}
                        min="10" max="1000"
                      />
                    </div>
                  )}
                </div>
              </div>

              <div className="form-section">
                <h3>Training Data</h3>
                
                <div className="upload-section">
                  <label className="file-upload-label">
                    <input 
                      type="file" 
                      multiple 
                      accept=".txt,.md,.pdf,.py,.js,.html,.json"
                      onChange={handleFileUpload}
                      disabled={uploading}
                    />
                    {uploading ? 'Uploading...' : 'Upload Documents'}
                  </label>
                  <p>Supported formats: {capabilities?.supported_formats.join(', ')}</p>
                </div>

                <div className="text-input-section">
                  <textarea 
                    value={documentText}
                    onChange={(e) => setDocumentText(e.target.value)}
                    placeholder="Or paste document text here..."
                    rows={4}
                  />
                  <button onClick={addDocumentText} disabled={!documentText.trim()}>
                    Add Document Text
                  </button>
                </div>

                <div className="document-list">
                  <h4>Uploaded Files ({trainingConfig.file_paths.length})</h4>
                  {trainingConfig.file_paths.map((path, index) => (
                    <div key={index} className="document-item">
                      <span>{path}</span>
                      <button onClick={() => removeDocument(index, 'file_paths')}>Remove</button>
                    </div>
                  ))}

                  <h4>Text Documents ({trainingConfig.documents.length})</h4>
                  {trainingConfig.documents.map((doc, index) => (
                    <div key={index} className="document-item">
                      <span>{doc.substring(0, 100)}...</span>
                      <button onClick={() => removeDocument(index, 'documents')}>Remove</button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'dspy' && (
            <div className="dspy-tab">
              <div className="form-section">
                <h3>DSPy Signatures & Generators</h3>
                <p>Configure DSPy signatures for specialized prompt engineering and reasoning chains.</p>

                <div className="predefined-signatures">
                  <h4>Predefined Signatures</h4>
                  {predefinedSignatures.map((signature, index) => (
                    <div key={index} className="signature-card">
                      <h5>{signature.name}</h5>
                      <p>{signature.description}</p>
                      <div className="signature-fields">
                        <div className="input-fields">
                          <strong>Inputs:</strong>
                          {signature.input_fields.map((field, i) => (
                            <span key={i} className="field-tag">{field.name}</span>
                          ))}
                        </div>
                        <div className="output-fields">
                          <strong>Outputs:</strong>
                          {signature.output_fields.map((field, i) => (
                            <span key={i} className="field-tag">{field.name}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="custom-signatures">
                  <h4>Custom Signatures</h4>
                  <button onClick={addCustomSignature} className="add-signature-btn">
                    Add Custom Signature
                  </button>
                  
                  {customSignatures.map((signature, sigIndex) => (
                    <div key={sigIndex} className="custom-signature-editor">
                      <div className="signature-header">
                        <input 
                          value={signature.name}
                          onChange={(e) => updateSignature(sigIndex, 'name', e.target.value)}
                          placeholder="Signature Name"
                          className="signature-name-input"
                        />
                        <button onClick={() => setCustomSignatures(prev => prev.filter((_, i) => i !== sigIndex))}>
                          Remove
                        </button>
                      </div>
                      
                      <textarea 
                        value={signature.description}
                        onChange={(e) => updateSignature(sigIndex, 'description', e.target.value)}
                        placeholder="Signature description..."
                        rows={2}
                      />

                      <div className="fields-editor">
                        <div className="input-fields-editor">
                          <h5>Input Fields</h5>
                          {signature.input_fields.map((field, fieldIndex) => (
                            <div key={fieldIndex} className="field-editor">
                              <input 
                                value={field.name}
                                onChange={(e) => {
                                  const newFields = [...signature.input_fields];
                                  newFields[fieldIndex].name = e.target.value;
                                  updateSignature(sigIndex, 'input_fields', newFields);
                                }}
                                placeholder="Field name"
                              />
                              <input 
                                value={field.description}
                                onChange={(e) => {
                                  const newFields = [...signature.input_fields];
                                  newFields[fieldIndex].description = e.target.value;
                                  updateSignature(sigIndex, 'input_fields', newFields);
                                }}
                                placeholder="Field description"
                              />
                            </div>
                          ))}
                          <button onClick={() => addSignatureField(sigIndex, 'input_fields')}>
                            Add Input Field
                          </button>
                        </div>

                        <div className="output-fields-editor">
                          <h5>Output Fields</h5>
                          {signature.output_fields.map((field, fieldIndex) => (
                            <div key={fieldIndex} className="field-editor">
                              <input 
                                value={field.name}
                                onChange={(e) => {
                                  const newFields = [...signature.output_fields];
                                  newFields[fieldIndex].name = e.target.value;
                                  updateSignature(sigIndex, 'output_fields', newFields);
                                }}
                                placeholder="Field name"
                              />
                              <input 
                                value={field.description}
                                onChange={(e) => {
                                  const newFields = [...signature.output_fields];
                                  newFields[fieldIndex].description = e.target.value;
                                  updateSignature(sigIndex, 'output_fields', newFields);
                                }}
                                placeholder="Field description"
                              />
                            </div>
                          ))}
                          <button onClick={() => addSignatureField(sigIndex, 'output_fields')}>
                            Add Output Field
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'embeddings' && (
            <div className="embeddings-tab">
              <div className="form-section">
                <h3>Embedding Model Configuration</h3>
                <p>Configure embedding models for better document retrieval and semantic search.</p>

                <div className="embedding-options">
                  <div className="option-card">
                    <h4>Document-Query Matching</h4>
                    <p>Train embeddings to better match queries with relevant documents</p>
                    <label>
                      <input type="checkbox" defaultChecked />
                      Enable contrastive learning
                    </label>
                    <label>
                      <input type="checkbox" />
                      Use hard negative mining
                    </label>
                  </div>

                  <div className="option-card">
                    <h4>Semantic Clustering</h4>
                    <p>Group similar documents for better topic coverage</p>
                    <label>
                      <input type="checkbox" defaultChecked />
                      Apply hierarchical clustering
                    </label>
                    <label>
                      Number of clusters: 
                      <input type="number" defaultValue={3} min={1} max={10} />
                    </label>
                  </div>

                  <div className="option-card">
                    <h4>Entity Recognition</h4>
                    <p>Extract and embed important entities and concepts</p>
                    <label>
                      <input type="checkbox" defaultChecked />
                      Extract named entities
                    </label>
                    <label>
                      <input type="checkbox" />
                      Generate entity-based questions
                    </label>
                  </div>
                </div>

                <div className="embedding-models">
                  <h4>Available Embedding Models</h4>
                  <select defaultValue="sentence-transformers/all-MiniLM-L6-v2">
                    <option value="sentence-transformers/all-MiniLM-L6-v2">
                      all-MiniLM-L6-v2 (Fast, 384 dim)
                    </option>
                    <option value="sentence-transformers/all-mpnet-base-v2">
                      all-mpnet-base-v2 (Balanced, 768 dim)
                    </option>
                    <option value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2">
                      paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)
                    </option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'status' && (
            <div className="status-tab">
              <div className="form-section">
                <h3>Training Status</h3>
                {trainingStatus ? (
                  <div className="status-display">
                    <div className="status-header">
                      <span className={`status-indicator ${trainingStatus.is_training ? 'training' : 'idle'}`}>
                        {trainingStatus.is_training ? 'Training' : 'Idle'}
                      </span>
                      <span className="progress-text">{trainingStatus.progress}%</span>
                    </div>
                    
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${trainingStatus.progress}%` }}
                      />
                    </div>

                    <div className="status-message">
                      {trainingStatus.status_message}
                    </div>

                    {trainingStatus.error && (
                      <div className="error-message">
                        Error: {trainingStatus.error}
                      </div>
                    )}

                    {trainingStatus.model_path && (
                      <div className="success-message">
                        Model saved to: {trainingStatus.model_path}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="no-status">
                    No training session active. Configure training parameters and start training.
                  </div>
                )}

                <div className="dependencies-status">
                  <h4>System Dependencies</h4>
                  {capabilities?.dependencies && (
                    <div className="dependencies-grid">
                      {Object.entries(capabilities.dependencies).map(([dep, available]) => (
                        <div key={dep} className={`dependency-item ${available ? 'available' : 'missing'}`}>
                          <span className="dependency-name">{dep}</span>
                          <span className="dependency-status">
                            {available ? '✓ Available' : '✗ Missing'}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button 
            onClick={startTraining}
            disabled={
              trainingConfig.documents.length === 0 && 
              trainingConfig.file_paths.length === 0 ||
              !capabilities?.training_types[trainingConfig.training_type]?.available
            }
            className="start-training-btn"
          >
            Start Training
          </button>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

export default TrainingModal;
