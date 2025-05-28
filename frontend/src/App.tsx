import { useState, useEffect } from 'react';
import './App.css';
import ChatWindow from './ChatWindow';
import MessageInput from './MessageInput';
import FileUploader from './FileUploader';
import NewConversationButton from './NewConversationButton';
import ConversationList from './ConversationList';

type Message = { role: 'user' | 'bot'; content: string };
type Conversation = { id: string; messages: Message[] };

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string>('');
  useEffect(() => { newConversation(); }, []);
  const newConversation = () => {
    const id = Date.now().toString();
    setConversations(prev => [...prev, { id, messages: [] }]);
    setCurrentConvId(id);
  };
  const selectConversation = (id: string) => setCurrentConvId(id);
  const activeConversation = conversations.find(c => c.id === currentConvId);
  const currentMessages = activeConversation?.messages ?? [];

  const handleUpload = async (files: FileList) => {
    const form = new FormData();
    Array.from(files).forEach(file => form.append('files', file));
    await fetch('/upload', { method: 'POST', body: form });
    alert('Upload complete and indexed');
  };

  const handleSend = async (text: string) => {
    setConversations(prev => prev.map(c => c.id === currentConvId ? { ...c, messages: [...c.messages, { role: 'user', content: text }] } : c));
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text }),
    });
    const { answer } = await resp.json();
    setConversations(prev => prev.map(c => c.id === currentConvId ? { ...c, messages: [...c.messages, { role: 'bot', content: answer }] } : c));
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
    </div>
  );
}

export default App;
