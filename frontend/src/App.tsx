import { useState, useEffect } from 'react';
import './App.css';
import ChatWindow from './ChatWindow';
import MessageInput from './MessageInput';
import FileUploader from './FileUploader';
import NewConversationButton from './NewConversationButton';
import ConversationList from './ConversationList';
import UploadModal from './UploadModal';
import ModelTrainingStatus from './ModelTrainingStatus';
import TrainingModal from './TrainingModal';
import DataExtractionModal from './DataExtractionModal';
import DSPyFunctionCreator from './DSPyFunctionCreator';
import FileTree from './FileTree';
import DocumentViewer from './DocumentViewer';
import EmbeddingTrainingModal from './EmbeddingTrainingModal';

type Message = { role: 'user' | 'bot'; content: string };
type Conversation = { id: string; messages: Message[] };

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  children?: FileNode[];
  size?: number;
  lastModified?: number;
}

// Helper function to generate unique IDs
const generateUniqueId = () => {
  return Date.now().toString() + Math.random().toString(36).substring(2, 9);
};

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string>('');
  const [modalOpen, setModalOpen] = useState(false);
  const [uploadStage, setUploadStage] = useState<'uploading' | 'indexing' | 'complete' | 'error'>('uploading');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [trainingStarted, setTrainingStarted] = useState(false);
  const [trainingModalOpen, setTrainingModalOpen] = useState(false);
  const [advancedTrainingModalOpen, setAdvancedTrainingModalOpen] = useState(false);
  const [dataExtractionModalOpen, setDataExtractionModalOpen] = useState(false);
  const [dspyFunctionCreatorOpen, setDspyFunctionCreatorOpen] = useState(false);
  const [embeddingTrainingModalOpen, setEmbeddingTrainingModalOpen] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<FileNode[]>([]);
  const [documentViewerOpen, setDocumentViewerOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<FileNode | null>(null);
  
  // Load file list on component mount and initialize new conversation
  useEffect(() => { 
    newConversation(); 
    refreshFileList();
  }, []);
  
  const newConversation = () => {
    const id = generateUniqueId();
    setConversations(prev => [...prev, { id, messages: [] }]);
    setCurrentConvId(id);
  };
  
  const selectConversation = (id: string) => setCurrentConvId(id);
  const activeConversation = conversations.find(c => c.id === currentConvId);
  const currentMessages = activeConversation?.messages ?? [];

  // Function to build file tree structure from FileList
  const buildFileTree = (files: FileList): FileNode[] => {
    const fileMap: { [path: string]: FileNode } = {};
    const rootNodes: FileNode[] = [];

    Array.from(files).forEach(file => {
      const pathParts = file.webkitRelativePath.split('/');
      let currentPath = '';
      
      pathParts.forEach((part, index) => {
        const previousPath = currentPath;
        currentPath = currentPath ? `${currentPath}/${part}` : part;
        
        if (!fileMap[currentPath]) {
          const isFile = index === pathParts.length - 1;
          const node: FileNode = {
            name: part,
            path: currentPath,
            type: isFile ? 'file' : 'folder',
            children: isFile ? undefined : [],
            size: isFile ? file.size : undefined,
            lastModified: isFile ? file.lastModified : undefined
          };
          
          fileMap[currentPath] = node;
          
          if (index === 0) {
            // Root level
            rootNodes.push(node);
          } else {
            // Add to parent
            const parent = fileMap[previousPath];
            if (parent && parent.children) {
              parent.children.push(node);
            }
          }
        }
      });
    });

    return rootNodes;
  };

  const handleUpload = async (files: FileList) => {
    try {
      const form = new FormData();
      const totalFiles = files.length;
      let uploadedCount = 0;
      
      // Build file tree structure for immediate display
      const fileTree = buildFileTree(files);
      setUploadedFiles(fileTree);
      
      // Open modal in uploading state
      setModalOpen(true);
      setUploadStage('uploading');
      setUploadProgress(0);
      setTrainingStarted(false);
      
      // Add each file to the form data
      Array.from(files).forEach(file => {
        form.append('files', file);
        uploadedCount++;
        setUploadProgress(Math.round((uploadedCount / totalFiles) * 50)); // First 50% for upload
      });
      
      // Start the actual upload
      const uploadResponse = await fetch('/api/upload', { 
        method: 'POST', 
        body: form 
      });
      
      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }
      
      // Switch to indexing state
      setUploadStage('indexing');
      
      // Simulate progress for indexing (since the server doesn't report progress)
      let indexProgress = 0;
      const progressInterval = setInterval(() => {
        indexProgress += 5;
        setUploadProgress(50 + Math.min(indexProgress, 50)); // Second 50% for indexing
        
        if (indexProgress >= 50) {
          clearInterval(progressInterval);
          setUploadStage('complete');
          setUploadProgress(100);
          
          // Refresh file list from server after upload completes
          refreshFileList();
        }
      }, 500);
      
      const responseData = await uploadResponse.json();
      
      // Check if training was started
      if (responseData.training_started) {
        setTrainingStarted(true);
      }
      
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStage('error');
    }
  };

  // Function to refresh file list from server
  const refreshFileList = async () => {
    try {
      const response = await fetch('/api/files/list');
      if (response.ok) {
        const data = await response.json();
        setUploadedFiles(data.files);
      }
    } catch (error) {
      console.error('Error refreshing file list:', error);
    }
  };

  // Load file list on component mount
  useEffect(() => {
    refreshFileList();
  }, []);
  
  const closeModal = () => {
    setModalOpen(false);
    // If we were in error state, reset for next upload
    if (uploadStage === 'error') {
      setUploadProgress(0);
    }
  };
  
  const handleViewTraining = () => {
    setModalOpen(false);
    setTrainingModalOpen(true);
  };
  
  const handleCloseTrainingModal = () => {
    setTrainingModalOpen(false);
  };

  const handleOpenAdvancedTraining = () => {
    setAdvancedTrainingModalOpen(true);
  };

  const handleCloseAdvancedTraining = () => {
    setAdvancedTrainingModalOpen(false);
  };

  const handleOpenDataExtraction = () => {
    setDataExtractionModalOpen(true);
  };

  const handleCloseDataExtraction = () => {
    setDataExtractionModalOpen(false);
  };

  const handleOpenDSPyCreator = () => {
    setDspyFunctionCreatorOpen(true);
  };

  const handleCloseDSPyCreator = () => {
    setDspyFunctionCreatorOpen(false);
  };

  const handleOpenEmbeddingTraining = () => {
    setEmbeddingTrainingModalOpen(true);
  };

  const handleCloseEmbeddingTraining = () => {
    setEmbeddingTrainingModalOpen(false);
  };

  const handleFileSelect = (file: FileNode) => {
    if (file.type === 'file') {
      setSelectedFile(file);
      setDocumentViewerOpen(true);
    }
  };

  const handleCloseDocumentViewer = () => {
    setDocumentViewerOpen(false);
    setSelectedFile(null);
  };

  const handleSaveDSPyFunction = async (func: any) => {
    try {
      const response = await fetch('/api/dspy/functions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(func),
      });
      
      if (response.ok) {
        console.log('DSPy function saved successfully');
        setDspyFunctionCreatorOpen(false);
      } else {
        console.error('Failed to save DSPy function');
      }
    } catch (error) {
      console.error('Error saving DSPy function:', error);
    }
  };

  const handleSend = async (text: string) => {
    setConversations(prev => prev.map(c => c.id === currentConvId ? { ...c, messages: [...c.messages, { role: 'user', content: text }] } : c));
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, conversation_id: currentConvId }),
      });
      
      if (!resp.ok) {
        throw new Error('Failed to get answer');
      }
      
      const { answer } = await resp.json();
      setConversations(prev => prev.map(c => c.id === currentConvId ? { ...c, messages: [...c.messages, { role: 'bot', content: answer }] } : c));
    } catch (error) {
      console.error('Chat error:', error);
      // Add an error message to the conversation
      setConversations(prev => prev.map(c => 
        c.id === currentConvId ? 
        { ...c, messages: [...c.messages, { role: 'bot', content: 'Sorry, there was an error processing your request.' }] } : 
        c
      ));
    }
  };

  return (
    <div className="app-container">
      <aside className="sidebar">
        <NewConversationButton onNew={newConversation} />
        <ConversationList conversations={conversations} currentId={currentConvId} onSelect={selectConversation} />
        <FileUploader onUpload={handleUpload} />
        
        {/* File Tree Display */}
        <FileTree 
          files={uploadedFiles}
          onFileSelect={handleFileSelect}
        />
        
        {/* AI Agent Control Panel */}
        <div className="agent-controls">
          <h3>AI Agent Controls</h3>
          <button className="control-button" onClick={handleOpenAdvancedTraining}>
            🎯 Advanced Training
          </button>
          <button className="control-button" onClick={handleOpenEmbeddingTraining}>
            🧠 Embedding Training
          </button>
          <button className="control-button" onClick={handleOpenDataExtraction}>
            📊 Data Extraction
          </button>
          <button className="control-button" onClick={handleOpenDSPyCreator}>
            🛠️ DSPy Function Creator
          </button>
        </div>
      </aside>
      <main className="chat-container">
        <ChatWindow messages={currentMessages} />
        <MessageInput onSend={handleSend} />
      </main>
      
      {/* Upload progress modal */}
      <UploadModal
        isOpen={modalOpen}
        stage={uploadStage}
        progress={uploadProgress}
        onClose={closeModal}
        onViewTraining={handleViewTraining}
        trainingStarted={trainingStarted}
      />
      
      {/* Model training status modal */}
      <ModelTrainingStatus 
        isOpen={trainingModalOpen}
        onClose={handleCloseTrainingModal}
      />
      
      {/* Advanced Training Modal */}
      <TrainingModal
        isOpen={advancedTrainingModalOpen}
        onClose={handleCloseAdvancedTraining}
      />
      
      {/* Data Extraction Modal */}
      <DataExtractionModal
        isOpen={dataExtractionModalOpen}
        onClose={handleCloseDataExtraction}
      />
      
      {/* DSPy Function Creator Modal */}
      <DSPyFunctionCreator
        isOpen={dspyFunctionCreatorOpen}
        onClose={handleCloseDSPyCreator}
        onSave={handleSaveDSPyFunction}
      />
      
      {/* Document Viewer Modal */}
      <DocumentViewer
        isOpen={documentViewerOpen}
        file={selectedFile}
        onClose={handleCloseDocumentViewer}
      />
      
      {/* Embedding Training Modal */}
      <EmbeddingTrainingModal
        isOpen={embeddingTrainingModalOpen}
        onClose={handleCloseEmbeddingTraining}
      />
    </div>
  );
}

export default App;
