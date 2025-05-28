import React from 'react';

interface Props {
  isOpen: boolean;
  stage: 'uploading' | 'indexing' | 'complete' | 'error';
  progress?: number;
  message?: string;
  onClose: () => void;
}

const UploadModal: React.FC<Props> = ({ 
  isOpen, 
  stage, 
  progress = 0, 
  message = '', 
  onClose 
}) => {
  if (!isOpen) return null;
  
  // Don't allow closing if we're in the middle of processing
  const canClose = stage === 'complete' || stage === 'error';
  
  // Modal title based on stage
  const getTitle = () => {
    switch (stage) {
      case 'uploading': return 'Uploading Files...';
      case 'indexing': return 'Indexing Documents...';
      case 'complete': return 'Upload Complete!';
      case 'error': return 'Upload Error';
      default: return 'Processing...';
    }
  };
  
  // Default messages if none provided
  const getDefaultMessage = () => {
    switch (stage) {
      case 'uploading': return 'Uploading your PDFs to the server...';
      case 'indexing': return 'Creating searchable index from documents...';
      case 'complete': return 'Your documents have been successfully uploaded and indexed.';
      case 'error': return 'There was an error processing your upload.';
      default: return '';
    }
  };

  const displayMessage = message || getDefaultMessage();
  
  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h3>{getTitle()}</h3>
        
        {/* Progress visualization */}
        {(stage === 'uploading' || stage === 'indexing') && (
          <div className="progress-bar-container">
            <div 
              className="progress-bar" 
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
        
        {/* Status icon */}
        <div className="status-icon">
          {stage === 'uploading' && <div className="spinner"></div>}
          {stage === 'indexing' && <div className="spinner"></div>}
          {stage === 'complete' && <div className="success-icon">✓</div>}
          {stage === 'error' && <div className="error-icon">✕</div>}
        </div>
        
        <p>{displayMessage}</p>
        
        {/* Only show close button when we can close */}
        {canClose && (
          <button 
            className="modal-button" 
            onClick={onClose}
          >
            Close
          </button>
        )}
      </div>
    </div>
  );
};

export default UploadModal;
