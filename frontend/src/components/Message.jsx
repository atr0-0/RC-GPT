import React, { useState } from 'react';
import { User, Bot, ChevronDown, ChevronUp, BookOpen } from 'lucide-react';
import './Message.css';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function Message({ message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';
  const isError = message.isError;

  const getSectionBadge = (sectionType) => {
    const badges = {
      'facts': { label: '📋 Facts', color: '#3b82f6' },
      'judgment': { label: '⚖️ Judgment', color: '#8b5cf6' },
      'reasoning': { label: '💡 Reasoning', color: '#f59e0b' },
      'general': { label: '📄 General', color: '#6b7280' }
    };
    return badges[sectionType] || badges['general'];
  };

  return (
    <div className={`message ${isUser ? 'user-message' : 'assistant-message'} ${isError ? 'error-message' : ''}`}>
      <div className="message-header">
        <div className="message-avatar">
          {isUser ? <User size={20} /> : <Bot size={20} />}
        </div>
        <span className="message-sender">{isUser ? 'You' : 'CaseLawGPT'}</span>
      </div>
      
      <div className="message-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content || ''}</ReactMarkdown>
      </div>

      {message.sources && message.sources.length > 0 && (
        <div className="sources-section">
          <button 
            className="sources-toggle"
            onClick={() => setShowSources(!showSources)}
          >
            <BookOpen size={18} />
            <span>View {message.sources.length} source excerpts</span>
            {showSources ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>

          {showSources && (
            <div className="sources-list">
              {message.sources.map((source, index) => {
                const badge = getSectionBadge(source.section_type);
                return (
                  <div key={index} className="source-card">
                    <div className="source-header">
                      <div className="source-title">
                        <span className="source-number">Source {index + 1}</span>
                        <span className="source-name">{source.case_name}</span>
                      </div>
                      <span className="confidence-badge">{source.confidence}</span>
                    </div>

                    {source.citation && source.citation !== 'N/A' && (
                      <div className="citation-badge">
                        📖 {source.citation}
                      </div>
                    )}

                    <span 
                      className="section-badge"
                      style={{ background: badge.color }}
                    >
                      {badge.label}
                    </span>

                    <div className="source-excerpt">
                      {source.excerpt}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {message.searchType && (
        <div className="search-type-badge">
          🔬 {message.searchType}
        </div>
      )}
    </div>
  );
}

export default Message;
