import React from 'react';
import { FileText, Database, Calendar, Activity } from 'lucide-react';
import './StatsBar.css';

function StatsBar({ stats }) {
  return (
    <div className="stats-bar">
      <div className="stat-card">
        <FileText className="stat-icon" size={24} />
        <div className="stat-content">
          <div className="stat-number">{stats.total_cases}</div>
          <div className="stat-label">SC Cases</div>
        </div>
      </div>
      
      <div className="stat-card">
        <Database className="stat-icon" size={24} />
        <div className="stat-content">
          <div className="stat-number">{stats.total_chunks?.toLocaleString()}</div>
          <div className="stat-label">Legal Chunks</div>
        </div>
      </div>
      
      <div className="stat-card">
        <Calendar className="stat-icon" size={24} />
        <div className="stat-content">
          <div className="stat-number">{stats.year_range}</div>
          <div className="stat-label">Year Range</div>
        </div>
      </div>
      
      <div className="stat-card">
        <Activity className="stat-icon" size={24} />
        <div className="stat-content">
          <div className="stat-number">{stats.uptime}</div>
          <div className="stat-label">Uptime</div>
        </div>
      </div>
    </div>
  );
}

export default StatsBar;
