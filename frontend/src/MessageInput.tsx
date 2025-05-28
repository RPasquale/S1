import { useState, KeyboardEvent } from 'react';

interface Props {
  onSend: (text: string) => void;
}

const MessageInput = ({ onSend }: Props) => {
  const [text, setText] = useState('');

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setText('');
  };
  
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="message-input-container">
      <form className="message-input-form" onSubmit={(e) => { e.preventDefault(); handleSend(); }}>
        <textarea
          className="message-input"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          rows={1}
        />
        <button className="send-button" type="submit">Send</button>
      </form>
    </div>
  );
};

export default MessageInput;