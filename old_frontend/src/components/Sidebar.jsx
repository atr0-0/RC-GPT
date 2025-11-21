import React from 'react';
import { Info, Filter, Settings, History, Trash2 } from 'lucide-react';
import './Sidebar.css';

const tortTypeOptions = [
  'negligence', 'defamation', 'trespass', 'nuisance', 'assault',
  'battery', 'false_imprisonment', 'malicious_prosecution',
  'conversion', 'strict_liability', 'vicarious_liability'
];

function Sidebar({ filters, onFilterChange, searchHistory, onClearChat, onHistoryClick }) {
  return (
    <aside className="sidebar">
      {/* About Section */}
      <div className="sidebar-section">
        <div className="section-header">
          <Info size={20} />
          <h3>About</h3>
        </div>
        <div className="section-content">
          <p><strong>CaseLawGPT</strong> helps lawyers find relevant Supreme Court tort law judgments.</p>
          <ul className="feature-list">
            <li>🔍 Hybrid search (semantic + keyword)</li>
            <li>💬 Conversational memory</li>
            <li>📚 690+ case database</li>
            <li>🎯 Precise citations</li>
            <li>🎚️ Advanced filtering</li>
          </ul>
        </div>
      </div>

      {/* Filters Section */}
      <div className="sidebar-section">
        <div className="section-header">
          <Filter size={20} />
          <h3>Filters</h3>
        </div>
        <div className="section-content">
          <label className="filter-label">
            📅 Case Year Range
            <div className="year-range-display">
              {filters.yearRange[0]} - {filters.yearRange[1]}
            </div>
            <input
              type="range"
              min="1950"
              max="2025"
              value={filters.yearRange[0]}
              onChange={(e) => onFilterChange({ 
                yearRange: [parseInt(e.target.value), filters.yearRange[1]] 
              })}
              className="range-slider"
            />
            <input
              type="range"
              min="1950"
              max="2025"
              value={filters.yearRange[1]}
              onChange={(e) => onFilterChange({ 
                yearRange: [filters.yearRange[0], parseInt(e.target.value)] 
              })}
              className="range-slider"
            />
          </label>

          <label className="filter-label">
            ⚖️ Tort Types
            <select
              multiple
              value={filters.tortTypes}
              onChange={(e) => {
                const selected = Array.from(e.target.selectedOptions, option => option.value);
                onFilterChange({ tortTypes: selected });
              }}
              className="tort-select"
              size="5"
            >
              {tortTypeOptions.map(type => (
                <option key={type} value={type}>
                  {type.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {/* Settings Section */}
      <div className="sidebar-section">
        <div className="section-header">
          <Settings size={20} />
          <h3>Settings</h3>
        </div>
        <div className="section-content">
          <label className="filter-label">
            Max Sources
            <input
              type="number"
              min="1"
              max="10"
              value={filters.maxSources}
              onChange={(e) => onFilterChange({ maxSources: parseInt(e.target.value) })}
              className="number-input"
            />
          </label>

          <label className="filter-label">
            Semantic Weight: {(filters.faissWeight * 100).toFixed(0)}%
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filters.faissWeight}
              onChange={(e) => onFilterChange({ faissWeight: parseFloat(e.target.value) })}
              className="range-slider"
            />
            <span className="range-hint">
              Keyword: {((1 - filters.faissWeight) * 100).toFixed(0)}%
            </span>
          </label>
        </div>
      </div>

      {/* Search History */}
      <div className="sidebar-section">
        <div className="section-header">
          <History size={20} />
          <h3>Recent Searches</h3>
        </div>
        <div className="section-content">
          {searchHistory.length > 0 ? (
            <div className="history-list">
              {searchHistory.slice(-5).reverse().map((query, index) => (
                <button
                  key={index}
                  className="history-item"
                  onClick={() => onHistoryClick(query)}
                >
                  ↩️ {query.substring(0, 40)}{query.length > 40 ? '...' : ''}
                </button>
              ))}
            </div>
          ) : (
            <p className="empty-state">No recent searches</p>
          )}
        </div>
      </div>

      {/* Clear Button */}
      <button className="clear-button" onClick={onClearChat}>
        <Trash2 size={18} />
        Clear Chat History
      </button>
    </aside>
  );
}

export default Sidebar;
