import React from 'react';

interface Props {
  onNew: () => void;
}

const NewConversationButton: React.FC<Props> = ({ onNew }) => {
  return (
    <button className="new-conversation-button" onClick={onNew}>
      <span className="button-icon">+</span>
      New Conversation
    </button>
  );
};

export default NewConversationButton;
