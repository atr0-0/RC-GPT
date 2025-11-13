import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import StatsBar from './components/StatsBar';
import ChatArea from './components/ChatArea';
import Sidebar from './components/Sidebar';
import QuerySuggestions from './components/QuerySuggestions';
import { getStats } from './services/api';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [stats, setStats] = useState(null);
  const [filters, setFilters] = useState({
    yearRange: [1950, 2025],
    tortTypes: [],
    maxSources: 5,
    faissWeight: 0.7
  });
  const [searchHistory, setSearchHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const data = await getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const handleNewMessage = (message) => {
    setMessages(prev => [...prev, message]);
    
    // Add to search history if it's a user message
    if (message.role === 'user' && !searchHistory.includes(message.content)) {
      setSearchHistory(prev => [...prev.slice(-9), message.content]);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  const handleFilterChange = (newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  };

  return (
    <div className="app">
      <Header />
      {stats && <StatsBar stats={stats} />}
      
      <div className="app-container">
        <Sidebar 
          filters={filters}
          onFilterChange={handleFilterChange}
          searchHistory={searchHistory}
          onClearChat={handleClearChat}
          onHistoryClick={(query) => {
            // Trigger new search with historical query
            const event = new CustomEvent('suggestionClick', { detail: query });
            window.dispatchEvent(event);
          }}
        />
        
        <main className="main-content">
          {(() => {
            // Show suggestions until we have at least one substantive assistant message with sources
            const hasSubstantiveAnswer = messages.some(m => m.role === 'assistant' && m.sources && m.sources.length > 0);
            if (!hasSubstantiveAnswer) {
              return (
                <QuerySuggestions onSuggestionClick={(query) => {
                  const event = new CustomEvent('suggestionClick', { detail: query });
                  window.dispatchEvent(event);
                }} />
              );
            }
            return null;
          })()}
          
          <ChatArea 
            messages={messages}
            onNewMessage={handleNewMessage}
            filters={filters}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
