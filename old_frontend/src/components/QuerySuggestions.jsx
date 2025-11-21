import React from 'react';
import { Hospital, Shield, Newspaper, Briefcase, Zap, Car } from 'lucide-react';
import './QuerySuggestions.css';

const suggestions = [
  { icon: Hospital, text: 'Cases on medical negligence compensation', query: 'Cases on medical negligence compensation' },
  { icon: Shield, text: 'Police custody torture and damages', query: 'Police custody torture and damages' },
  { icon: Newspaper, text: 'Defamation by media houses', query: 'Defamation by media houses' },
  { icon: Briefcase, text: 'Vicarious liability of employers', query: 'Vicarious liability of employers' },
  { icon: Zap, text: 'Strict liability in hazardous activities', query: 'Strict liability in hazardous activities' },
  { icon: Car, text: 'Motor vehicle accident compensation', query: 'Motor vehicle accident compensation' }
];

function QuerySuggestions({ onSuggestionClick }) {
  return (
    <div className="query-suggestions">
      <h3 className="suggestions-title">💡 Try these example queries:</h3>
      <div className="suggestions-grid">
        {suggestions.map((suggestion, index) => {
          const Icon = suggestion.icon;
          return (
            <button
              key={index}
              className="suggestion-chip"
              onClick={() => onSuggestionClick(suggestion.query)}
            >
              <Icon size={18} />
              <span>{suggestion.text}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default QuerySuggestions;
