import React from 'react';

export interface Conversation {
  id: string;
  messages: { role: 'user' | 'bot'; content: string }[];
}

interface Props {
  conversations: Conversation[];
  currentId: string;
  onSelect: (id: string) => void;
}

const ConversationList: React.FC<Props> = ({ conversations, currentId, onSelect }) => {
  // Format conversation name based on first message or timestamp
  const getConversationTitle = (conversation: Conversation) => {
    if (conversation.messages.length > 0) {
      // Use first user message as the title, truncated
      const firstUserMsg = conversation.messages.find(m => m.role === 'user');
      if (firstUserMsg) {
        const title = firstUserMsg.content.substring(0, 25);
        return title.length < firstUserMsg.content.length ? `${title}...` : title;
      }
    }
    
    // Fallback to a formatted timestamp from the ID
    const timestamp = new Date(parseInt(conversation.id));
    return `Conversation ${timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  };

  return (
    <ul className="conversation-list">
      {conversations.map(conv => (
        <li
          key={conv.id}
          className={conv.id === currentId ? 'active' : ''}
          onClick={() => onSelect(conv.id)}
        >
          {getConversationTitle(conv)}
        </li>
      ))}
    </ul>
  );
};

export default ConversationList;
