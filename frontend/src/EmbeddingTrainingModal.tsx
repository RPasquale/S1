import React, { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './EmbeddingTrainingModal.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface EmbeddingTrainingModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface TrainingProgress {
  epoch: number;
  step: number;
  train_loss: number;
  eval_loss?: number;
  learning_rate: number;
  timestamp: string;
}

interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
}

const EmbeddingTrainingModal: React.FC<EmbeddingTrainingModalProps> = ({ isOpen, onClose }) => {
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    epochs: 3,
    batch_size: 16,
    learning_rate: 2e-5
  });
  
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [useCFADocuments, setUseCFADocuments] = useState<boolean>(false);
  const wsRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // WebSocket connection for live updates
  useEffect(() => {
    if (isOpen && !wsRef.current) {
      connectWebSocket();
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [isOpen]);

  const connectWebSocket = () => {
    const wsUrl = `ws://localhost:8000/ws/training`;
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
    };
    
    wsRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'training_progress') {
        setTrainingProgress(prev => [...prev, message.data]);
        setStatus(`Epoch ${message.data.epoch}, Step ${message.data.step}, Loss: ${message.data.train_loss.toFixed(4)}`);
      } else if (message.type === 'training_complete') {
        setIsTraining(false);
        setStatus('Training completed successfully!');
      } else if (message.type === 'training_error') {
        setIsTraining(false);
        setError(message.error);
        setStatus('Training failed');
      } else if (message.type === 'training_stopped') {
        setIsTraining(false);
        setStatus('Training stopped by user');
      }
    };
    
    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      setUploadedFiles(Array.from(files));
    }
  };
  const startTraining = async () => {
    // If not using CFA documents, require uploads
    if (!useCFADocuments && uploadedFiles.length === 0) {
      setError('Please upload at least one document or select "Use CFA Documents"');
      return;
    }

    setError('');
    setStatus('Starting training...');
    setTrainingProgress([]);
    setIsTraining(true);

    try {
      let response;      // Use different endpoints based on document source
      if (useCFADocuments) {
        // Use CFA documents endpoint with query parameters
        const url = `/api/train/cfa?epochs=${trainingConfig.epochs}&batch_size=${trainingConfig.batch_size}&learning_rate=${trainingConfig.learning_rate}`;
        response = await fetch(url, {
          method: 'POST'
        });
      } else {
        // Use uploaded documents
        const formData = new FormData();
        uploadedFiles.forEach(file => {
          formData.append('files', file);
        });
        formData.append('epochs', trainingConfig.epochs.toString());
        formData.append('batch_size', trainingConfig.batch_size.toString());
        formData.append('learning_rate', trainingConfig.learning_rate.toString());

        response = await fetch('/api/train/embedding', {
          method: 'POST',
          body: formData
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Training failed to start');
      }

      const result = await response.json();
      setStatus(`Training started with ${result.document_count || result.num_documents} documents`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      setIsTraining(false);
      setStatus('Failed to start training');
    }
  };

  const stopTraining = async () => {
    try {
      const response = await fetch('/api/train/stop', {
        method: 'POST'
      });

      if (response.ok) {
        setStatus('Stopping training...');
      }
    } catch (err) {
      setError('Failed to stop training');
    }
  };

  // Prepare chart data
  const chartData = {
    labels: trainingProgress.map(p => `E${p.epoch}S${p.step}`),
    datasets: [
      {
        label: 'Training Loss',
        data: trainingProgress.map(p => p.train_loss),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      },
      ...(trainingProgress.some(p => p.eval_loss !== undefined) ? [{
        label: 'Validation Loss',
        data: trainingProgress.map(p => p.eval_loss || null),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1
      }] : [])
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Training Loss Over Time'
      }
    },
    scales: {
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: 'Loss'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Training Steps'
        }
      }
    },
    animation: {
      duration: 0 // Disable animation for real-time updates
    }
  };

  if (!isOpen) return null;

  return (
    <div className="embedding-training-modal-overlay">
      <div className="embedding-training-modal">
        <div className="modal-header">
          <h2>Embedding Model Training</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="modal-content">          {/* Document Source Selection */}
          <div className="section">
            <h3>Training Documents Source</h3>
            <div className="document-source-selection">
              <div className="source-option">
                <input
                  type="radio"
                  id="upload-documents"
                  name="document-source"
                  checked={!useCFADocuments}
                  onChange={() => setUseCFADocuments(false)}
                  disabled={isTraining}
                />
                <label htmlFor="upload-documents">Upload New Documents</label>
              </div>
              <div className="source-option">
                <input
                  type="radio"
                  id="use-cfa-documents"
                  name="document-source"
                  checked={useCFADocuments}
                  onChange={() => setUseCFADocuments(true)}
                  disabled={isTraining}
                />
                <label htmlFor="use-cfa-documents">Use CFA Documents (already uploaded)</label>
              </div>
            </div>
          </div>

          {/* File Upload Section */}
          {!useCFADocuments && (
            <div className="section">
              <h3>Upload Training Documents</h3>
              <div className="file-upload-area">
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".txt,.pdf,.docx,.md"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
                <button 
                  className="upload-button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isTraining}
                >
                  Select Documents
                </button>
                
                {uploadedFiles.length > 0 && (
                  <div className="uploaded-files">
                    <h4>Selected Files ({uploadedFiles.length}):</h4>
                    <ul>
                      {uploadedFiles.map((file, index) => (
                        <li key={index}>{file.name} ({(file.size / 1024).toFixed(1)} KB)</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* CFA Documents Info */}
          {useCFADocuments && (
            <div className="section">
              <h3>CFA Documents</h3>
              <div className="cfa-documents-info">
                <p>Training will use the CFA documents from the configured document folder.</p>
                <p>Documents are loaded from: <code>C:\Users\robbi\OneDrive\CFA</code></p>
              </div>
            </div>
          )}

          {/* Training Configuration */}
          <div className="section">
            <h3>Training Configuration</h3>
            <div className="config-grid">
              <div className="config-item">
                <label>Epochs:</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={trainingConfig.epochs}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    epochs: parseInt(e.target.value)
                  }))}
                  disabled={isTraining}
                />
              </div>
              
              <div className="config-item">
                <label>Batch Size:</label>
                <input
                  type="number"
                  min="1"
                  max="64"
                  value={trainingConfig.batch_size}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    batch_size: parseInt(e.target.value)
                  }))}
                  disabled={isTraining}
                />
              </div>
              
              <div className="config-item">
                <label>Learning Rate:</label>
                <input
                  type="number"
                  step="0.00001"
                  min="0.00001"
                  max="0.1"
                  value={trainingConfig.learning_rate}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    learning_rate: parseFloat(e.target.value)
                  }))}
                  disabled={isTraining}
                />
              </div>
            </div>
          </div>

          {/* Training Controls */}
          <div className="section">
            <h3>Training Controls</h3>
            <div className="training-controls">              {!isTraining ? (
                <button 
                  className="start-training-button"
                  onClick={startTraining}
                  disabled={!useCFADocuments && uploadedFiles.length === 0}
                >
                  Start Training
                </button>
              ) : (
                <button 
                  className="stop-training-button"
                  onClick={stopTraining}
                >
                  Stop Training
                </button>
              )}
            </div>
            
            {status && (
              <div className="status-message">
                <strong>Status:</strong> {status}
              </div>
            )}
            
            {error && (
              <div className="error-message">
                <strong>Error:</strong> {error}
              </div>
            )}
          </div>

          {/* Live Loss Graph */}
          {trainingProgress.length > 0 && (
            <div className="section">
              <h3>Live Training Progress</h3>
              <div className="chart-container">
                <Line data={chartData} options={chartOptions} />
              </div>
              
              <div className="training-stats">
                <div className="stat">
                  <label>Current Loss:</label>
                  <span>{trainingProgress[trainingProgress.length - 1]?.train_loss.toFixed(4) || 'N/A'}</span>
                </div>
                <div className="stat">
                  <label>Total Steps:</label>
                  <span>{trainingProgress.length}</span>
                </div>
                <div className="stat">
                  <label>Current Epoch:</label>
                  <span>{trainingProgress[trainingProgress.length - 1]?.epoch || 'N/A'}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EmbeddingTrainingModal;
