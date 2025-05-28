import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'

export type Message = { role: 'user' | 'bot'; content: string }

interface Props { 
  messages: Message[] 
}

const ChatWindow = ({ messages }: Props) => {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  // Auto-scroll to bottom when new messages come in
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Show a welcome message if no messages yet
  if (messages.length === 0) {
    return (
      <div className="chat-window">
        <div className="welcome-message">
          <h2>Welcome to PDF QA Chatbot</h2>
          <p>Upload PDF documents and ask questions about them</p>
        </div>
        <div ref={messagesEndRef} />
      </div>
    )
  }

  return (
    <div className="chat-window">
      {messages.map((msg, idx) => (
        <div key={idx} className={`message ${msg.role}`}>
          <div className="message-content">
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        </div>
      ))}
      <div ref={messagesEndRef} />
    </div>
  )
}

export default ChatWindow