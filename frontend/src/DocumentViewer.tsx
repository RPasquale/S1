import React, { useState, useEffect } from 'react';
import './DocumentViewer.css';

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  children?: FileNode[];
  size?: number;
  lastModified?: number;
}

interface Props {
  isOpen: boolean;
  file: FileNode | null;
  onClose: () => void;
}

const DocumentViewer: React.FC<Props> = ({ isOpen, file, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && file) {
      loadFileContent();
    }
    return () => {
      // Clean up blob URL if it exists
      if (fileUrl) {
        URL.revokeObjectURL(fileUrl);
      }
    };
  }, [isOpen, file]);

  const loadFileContent = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setFileContent(null);
    setFileUrl(null);

    try {      // Make request to backend to get file content
      const response = await fetch(`/api/files/view?path=${encodeURIComponent(file.path)}`);
      
      if (!response.ok) {
        throw new Error(`Failed to load file: ${response.statusText}`);
      }

      const fileExtension = file.name.split('.').pop()?.toLowerCase();
      
      if (fileExtension === 'pdf') {
        // For PDFs, create a blob URL
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setFileUrl(url);
      } else if (['txt', 'md', 'json', 'xml', 'csv'].includes(fileExtension || '')) {
        // For text files, get the content as text
        const text = await response.text();
        setFileContent(text);
      } else {
        // For other files, try to create a blob URL
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setFileUrl(url);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load file');
    } finally {
      setLoading(false);
    }
  };

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf': return '📄';
      case 'txt': return '📝';
      case 'doc':
      case 'docx': return '📃';
      case 'xls':
      case 'xlsx': return '📊';
      case 'ppt':
      case 'pptx': return '📽️';
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif': return '🖼️';
      default: return '📄';
    }
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (timestamp?: number) => {
    if (!timestamp) return '';
    return new Date(timestamp).toLocaleDateString() + ' ' + new Date(timestamp).toLocaleTimeString();
  };

  const downloadFile = async () => {
    if (!file) return;
      try {
      const response = await fetch(`/api/files/download?path=${encodeURIComponent(file.path)}`);
      if (!response.ok) throw new Error('Download failed');
      
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download error:', err);
    }
  };

  const renderFileContent = () => {
    if (loading) {
      return (
        <div className="viewer-loading">
          <div className="spinner"></div>
          <p>Loading document...</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="viewer-error">
          <div className="error-icon">⚠️</div>
          <p>{error}</p>
          <button onClick={loadFileContent} className="retry-button">
            Try Again
          </button>
        </div>
      );
    }

    if (!file) return null;

    const fileExtension = file.name.split('.').pop()?.toLowerCase();

    // PDF viewer
    if (fileExtension === 'pdf' && fileUrl) {
      return (
        <div className="pdf-viewer">
          <iframe
            src={fileUrl}
            title={file.name}
            width="100%"
            height="100%"
            style={{ border: 'none' }}
          />
        </div>
      );
    }

    // Text content viewer
    if (fileContent !== null) {
      return (
        <div className="text-viewer">
          <pre className="text-content">{fileContent}</pre>
        </div>
      );
    }

    // Image viewer
    if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(fileExtension || '') && fileUrl) {
      return (
        <div className="image-viewer">
          <img src={fileUrl} alt={file.name} className="image-content" />
        </div>
      );
    }

    // Unsupported file type
    return (
      <div className="unsupported-viewer">
        <div className="unsupported-icon">📄</div>
        <p>Preview not available for this file type</p>
        <button onClick={downloadFile} className="download-button">
          Download File
        </button>
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="document-viewer-overlay">
      <div className="document-viewer">
        <div className="viewer-header">
          <div className="file-info">
            <span className="file-icon-large">
              {file ? getFileIcon(file.name) : '📄'}
            </span>
            <div className="file-details">
              <h3 className="file-name">{file?.name}</h3>
              <div className="file-meta">
                {file?.size && <span>{formatFileSize(file.size)}</span>}
                {file?.lastModified && <span>{formatDate(file.lastModified)}</span>}
              </div>
            </div>
          </div>
          
          <div className="viewer-actions">
            <button onClick={downloadFile} className="action-button download">
              ⬇️ Download
            </button>
            <button onClick={onClose} className="action-button close">
              ✕
            </button>
          </div>
        </div>

        <div className="viewer-content">
          {renderFileContent()}
        </div>
      </div>
    </div>
  );
};

export default DocumentViewer;
