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

type Message = { role: 'user' | 'bot'; content: string };
type Conversation = { id: string; messages: Message[] };

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
  
  useEffect(() => { newConversation(); }, []);
  
  const newConversation = () => {
    const id = generateUniqueId();
    setConversations(prev => [...prev, { id, messages: [] }]);
    setCurrentConvId(id);
  };
  
  const selectConversation = (id: string) => setCurrentConvId(id);
  const activeConversation = conversations.find(c => c.id === currentConvId);
  const currentMessages = activeConversation?.messages ?? [];

  const handleUpload = async (files: FileList) => {
    try {
      const form = new FormData();
      const totalFiles = files.length;
      let uploadedCount = 0;
      
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
        
        {/* AI Agent Control Panel */}
        <div className="agent-controls">
          <h3>AI Agent Controls</h3>
          <button className="control-button" onClick={handleOpenAdvancedTraining}>
            🎯 Advanced Training
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
    </div>
  );
}

export default App;
