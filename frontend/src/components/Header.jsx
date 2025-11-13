import React from 'react';
import { Scale } from 'lucide-react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="header-title">
          <Scale className="header-icon" size={36} />
          <h1>CaseLawGPT</h1>
        </div>
        <p className="header-subtitle">AI Legal Research Assistant for Indian Tort Law</p>
      </div>
    </header>
  );
}

export default Header;
