import React, { useState, useEffect } from 'react';
import './App.css';

interface ModelTrainingStatusProps {
  isOpen: boolean;
  onClose: () => void;
}

interface TrainingStatus {
  is_training: boolean;
  start_time: string | null;
  progress: number;
  status_message: string;
  completed_steps: string[];
  errors: string[];
}

const ModelTrainingStatus: React.FC<ModelTrainingStatusProps> = ({ isOpen, onClose }) => {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    is_training: false,
    start_time: null,
    progress: 0,
    status_message: '',
    completed_steps: [],
    errors: []
  });

  // Polling interval for status updates (in milliseconds)
  const STATUS_POLL_INTERVAL = 3000;

  useEffect(() => {
    let intervalId: number | null = null;

    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/training/status');
        if (response.ok) {
          const data = await response.json();
          setTrainingStatus(data);

          // Stop polling when training is complete
          if (!data.is_training && data.progress === 100) {
            if (intervalId) {
              window.clearInterval(intervalId);
              intervalId = null;
            }
          }
        }
      } catch (error) {
        console.error('Error fetching training status:', error);
      }
    };

    // Only poll if the modal is open
    if (isOpen) {
      // Fetch initial status
      fetchStatus();
      
      // Start polling
      intervalId = window.setInterval(fetchStatus, STATUS_POLL_INTERVAL);
    }

    return () => {
      // Clean up interval on component unmount or when modal closes
      if (intervalId) {
        window.clearInterval(intervalId);
      }
    };
  }, [isOpen]);

  const handleStartTraining = async () => {
    try {
      await fetch('/api/training/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      // Fetch status update immediately
      const response = await fetch('/api/training/status');
      if (response.ok) {
        const data = await response.json();
        setTrainingStatus(data);
      }
    } catch (error) {
      console.error('Error starting training:', error);
    }
  };

  const handleCancelTraining = async () => {
    try {
      await fetch('/api/training/cancel', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      // Fetch status update immediately
      const response = await fetch('/api/training/status');
      if (response.ok) {
        const data = await response.json();
        setTrainingStatus(data);
      }
    } catch (error) {
      console.error('Error cancelling training:', error);
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'N/A';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch {
      return dateString;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop">
      <div className="modal-content model-training-status">
        <div className="modal-header">
          <h2>Model Training Status</h2>
          <button className="close-button" onClick={onClose}>&times;</button>
        </div>
        
        <div className="modal-body">
          <div className="status-section">
            <div className="status-header">
              <h3>Status: {trainingStatus.is_training ? 'Training in Progress' : 'Idle'}</h3>
              {trainingStatus.start_time && (
                <p>Started: {formatDate(trainingStatus.start_time)}</p>
              )}
            </div>
            
            <div className="progress-container">
              <div 
                className="progress-bar" 
                style={{ width: `${trainingStatus.progress}%` }}
              />
              <span className="progress-text">{trainingStatus.progress}%</span>
            </div>
            
            <p className="status-message">{trainingStatus.status_message}</p>
          </div>
          
          {trainingStatus.completed_steps.length > 0 && (
            <div className="completed-steps">
              <h4>Completed Steps:</h4>
              <ul>
                {trainingStatus.completed_steps.map((step, index) => (
                  <li key={index} className="completed-step">
                    <span className="step-check">✓</span>
                    {step.replace(/_/g, ' ')}
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {trainingStatus.errors.length > 0 && (
            <div className="error-section">
              <h4>Errors:</h4>
              <ul className="error-list">
                {trainingStatus.errors.map((error, index) => (
                  <li key={index} className="error-item">{error}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
        
        <div className="modal-footer">
          {!trainingStatus.is_training && trainingStatus.progress < 100 && (
            <button 
              className="primary-button"
              onClick={handleStartTraining}
            >
              Start Training
            </button>
          )}
          
          {trainingStatus.is_training && (
            <button 
              className="danger-button"
              onClick={handleCancelTraining}
            >
              Cancel Training
            </button>
          )}
          
          <button 
            className="secondary-button"
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelTrainingStatus;
