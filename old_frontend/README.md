# CaseLawGPT React Frontend

Modern, responsive web interface for the CaseLawGPT legal research assistant.

## Features

✨ **Modern UI/UX**
- Gradient design with dark theme
- Smooth animations and transitions
- Responsive layout for all devices
- Real-time chat interface

🎯 **Advanced Functionality**
- Query suggestion chips for quick start
- Live statistics dashboard
- Advanced filtering (year range, tort types)
- Search history tracking
- Expandable source citations
- Confidence scoring

🚀 **Tech Stack**
- React 18.3
- Vite (fast build tool)
- Axios for API calls
- Lucide React icons
- CSS Modules

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Backend API running (FastAPI on port 8000)

### Installation

1. **Install Dependencies**
   ```powershell
   cd frontend
   npm install
   ```

2. **Start Development Server**
   ```powershell
   npm run dev
   ```

   The app will be available at `http://localhost:3000`

### Quick Start (Full Stack)

Run both backend and frontend together:
```powershell
cd scripts
.\start_fullstack.ps1
```

This will:
- Start FastAPI backend on port 8000
- Start React frontend on port 3000
- Open in separate PowerShell windows

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # Main header with gradient
│   │   ├── StatsBar.jsx        # Statistics dashboard
│   │   ├── ChatArea.jsx        # Chat interface
│   │   ├── Message.jsx         # Individual messages
│   │   ├── Sidebar.jsx         # Filters & settings
│   │   └── QuerySuggestions.jsx # Example queries
│   ├── services/
│   │   └── api.js              # API client
│   ├── App.jsx                 # Main application
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles
├── index.html
├── package.json
└── vite.config.js
```

## Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## API Integration

The frontend connects to the FastAPI backend via proxy:

```javascript
// Configured in vite.config.js
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, '')
  }
}
```

### API Endpoints Used

- `GET /stats` - Get database statistics
- `POST /query` - Process legal research queries
- `GET /health` - Health check

## Component Overview

### Header
- Displays app title with Scale icon
- Gradient background matching brand colors

### StatsBar
- Shows 4 key metrics: Cases, Chunks, Year Range, Uptime
- Animated cards with hover effects

### QuerySuggestions
- 6 example queries with icons
- Clickable chips that trigger searches
- Only shown when chat is empty

### ChatArea
- Scrollable message history
- Input field with send button
- Loading indicator during searches
- Auto-scroll to latest message

### Message
- User and assistant message styling
- Expandable source citations
- Confidence badges
- Section type indicators
- Citation highlighting

### Sidebar
- About section with feature list
- Year range filter (dual sliders)
- Tort type multi-select
- Max sources slider
- Semantic/keyword weight adjuster
- Search history (last 5 queries)
- Clear chat button

## Styling

### Color Scheme
- **Primary Gradient**: `#667eea → #764ba2`
- **Background**: Dark gradient `#1a1a2e → #16213e`
- **Text**: Light gray `#e4e4e7`
- **Accents**: Blue, Green, Orange for badges

### Design Principles
- Glass-morphism effects with backdrop blur
- Smooth transitions (0.3s ease)
- Hover states with transform and shadow
- Consistent border-radius (0.5rem - 1rem)

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Troubleshooting

### Frontend won't start
```powershell
# Clear node_modules and reinstall
rm -r node_modules
npm install
```

### API connection errors
- Ensure backend is running on port 8000
- Check `vite.config.js` proxy settings
- Verify CORS settings in FastAPI

### Build errors
```powershell
# Clear Vite cache
npm run build -- --force
```

## Development Tips

### Hot Module Replacement
Vite supports HMR - changes reflect immediately without page reload.

### Component Development
Each component is self-contained with its own CSS file for easy maintenance.

### State Management
Using React hooks (useState, useEffect) with props drilling. Consider adding Context API or Zustand for larger scale.

## Performance

- **First Load**: ~1.5s
- **Subsequent Loads**: Instant (cached)
- **Build Size**: ~150KB (gzipped)
- **API Response**: ~2-4s (depends on query complexity)

## Future Enhancements

- [ ] Add dark/light theme toggle
- [ ] Implement chat export functionality
- [ ] Add keyboard shortcuts
- [ ] Save filter presets
- [ ] Add citation copy buttons
- [ ] Implement infinite scroll for long chats
- [ ] Add voice input
- [ ] PWA support for offline access

## License

Part of CaseLawGPT project - see main README
