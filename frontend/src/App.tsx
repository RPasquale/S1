import { useState, useEffect } from 'react';
import './App.css';
import ChatWindow from './ChatWindow';
import MessageInput from './MessageInput';
import FileUploader from './FileUploader';
import NewConversationButton from './NewConversationButton';
import ConversationList from './ConversationList';
import UploadModal from './UploadModal';

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
      
      // Add each file to the form data
      Array.from(files).forEach(file => {
        form.append('files', file);
        uploadedCount++;
        setUploadProgress(Math.round((uploadedCount / totalFiles) * 50)); // First 50% for upload
      });
      
      // Start the actual upload
      const uploadResponse = await fetch('/upload', { 
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
      
      await uploadResponse.json();
      
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

  const handleSend = async (text: string) => {
    setConversations(prev => prev.map(c => c.id === currentConvId ? { ...c, messages: [...c.messages, { role: 'user', content: text }] } : c));
    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text }),
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
      />
    </div>
  );
}

export default App;
