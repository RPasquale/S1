import { useState, useEffect } from 'react';
import './App.css';
import ChatWindow from './ChatWindow';
import MessageInput from './MessageInput';
import FileUploader from './FileUploader';
import NewConversationButton from './NewConversationButton';
import ConversationList from './ConversationList';
import UploadModal from './UploadModal';
import ModelTrainingStatus from './ModelTrainingStatus';

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
  
  useEffect(() => { newConversation(); }, []);
  
  const newConversation = () => {
    const id = generateUniqueId();
    setConversations(prev => [...prev, { id, messages: [] }]);
    setCurrentConvId(id);
  };
  
  const selectConversation = (id: string) => setCurrentConvId(id);
  const activeConversation = conversations.find(c => c.id === currentConvId);
  const currentMessages = activeConversation?.messages ?? [];  const handleUpload = async (files: FileList) => {
    console.log('handleUpload called with files:', files);
    console.log('Number of files:', files.length);
    console.log('File names:', Array.from(files).map(f => f.name));
    
    let progressInterval: number | null = null;
    
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
      
      // Create an AbortController for request timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
      
      try {
        // Start the actual upload
        const uploadResponse = await fetch('/api/upload', { 
          method: 'POST', 
          body: form,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!uploadResponse.ok) {
          throw new Error(`Upload failed with status: ${uploadResponse.status}`);
        }
        
        // Switch to indexing state
        setUploadStage('indexing');
        
        // Simulate progress for indexing (since the server doesn't report progress)
        let indexProgress = 0;
        progressInterval = window.setInterval(() => {
          indexProgress += 5;
          setUploadProgress(50 + Math.min(indexProgress, 50)); // Second 50% for indexing
          
          if (indexProgress >= 50) {
            if (progressInterval !== null) {
              clearInterval(progressInterval);
              progressInterval = null;
            }
            setUploadStage('complete');
            setUploadProgress(100);
          }
        }, 500);
        
        const responseData = await uploadResponse.json();
        
        // Check if training was started
        if (responseData.training_started) {
          setTrainingStarted(true);
        }
        
      } catch (fetchError) {
        clearTimeout(timeoutId);
        if (fetchError instanceof Error && fetchError.name === 'AbortError') {
          throw new Error('Upload timeout - please try again with smaller files');
        }
        throw fetchError;
      }
      
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStage('error');
      
      // Clean up progress interval if it exists
      if (progressInterval !== null) {
        clearInterval(progressInterval);
      }
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
  const handleSend = async (text: string) => {
    setConversations(prev => prev.map(c => c.id === currentConvId ? { ...c, messages: [...c.messages, { role: 'user', content: text }] } : c));
    
    // Create an AbortController for request timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout for chat
    
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, conversation_id: currentConvId }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!resp.ok) {
        throw new Error(`Failed to get answer: ${resp.status} ${resp.statusText}`);
      }
      
      const { answer } = await resp.json();
      setConversations(prev => prev.map(c => c.id === currentConvId ? { ...c, messages: [...c.messages, { role: 'bot', content: answer }] } : c));
    } catch (error) {
      clearTimeout(timeoutId);
      console.error('Chat error:', error);
      
      let errorMessage = 'Sorry, there was an error processing your request.';
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request timeout - the question took too long to process. Please try again.';
        } else if (error.message.includes('Failed to fetch')) {
          errorMessage = 'Unable to connect to the server. Please check if the backend is running.';
        }
      }
      
      // Add an error message to the conversation
      setConversations(prev => prev.map(c => 
        c.id === currentConvId ? 
        { ...c, messages: [...c.messages, { role: 'bot', content: errorMessage }] } : 
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
    </div>
  );
}

export default App;
