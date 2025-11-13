import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader } from 'lucide-react';
import Message from './Message';
import { processQuery } from '../services/api';
import './ChatArea.css';

function ChatArea({ messages, onNewMessage, filters, isLoading, setIsLoading }) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const handleSuggestionClick = (event) => {
      handleSubmit(event.detail);
    };

    window.addEventListener('suggestionClick', handleSuggestionClick);
    return () => window.removeEventListener('suggestionClick', handleSuggestionClick);
  }, [filters]);

  const handleSubmit = async (query) => {
    const queryText = query || input.trim();
    if (!queryText || isLoading) return;

    setInput('');
    setIsLoading(true);

    // Add user message
    const userMessage = { role: 'user', content: queryText };
    onNewMessage(userMessage);

    try {
      // Call API
      const response = await processQuery({
        query: queryText,
        year_range: filters.yearRange,
        tort_types: filters.tortTypes,
        max_sources: filters.maxSources,
        faiss_weight: filters.faissWeight
      });

      // Add assistant message
      const assistantMessage = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        searchType: response.search_type
      };
      onNewMessage(assistantMessage);

    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.message || 'Failed to process query'}`,
        isError: true
      };
      onNewMessage(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="chat-area">
      <div className="messages-container">
        {messages.map((message, index) => (
          <Message key={index} message={message} />
        ))}
        {isLoading && (
          <div className="loading-indicator">
            <Loader className="spinner" size={24} />
            <span>Searching case law database...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about tort law cases..."
          disabled={isLoading}
          className="chat-input"
        />
        <button 
          onClick={() => handleSubmit()}
          disabled={!input.trim() || isLoading}
          className="send-button"
        >
          <Send size={20} />
        </button>
      </div>
    </div>
  );
}

export default ChatArea;
