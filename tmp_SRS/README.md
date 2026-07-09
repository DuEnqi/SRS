# MultiNPC Narrative Development Platform

A professional AI-powered development platform for creating, debugging, testing, and showcasing multi-NPC intelligent agent game projects.

## Features

### 🎮 Play Mode
- Visual Novel-style dialogue system
- Real-time AI-powered NPC conversations
- Player actions: Talk, Inspect, Question, Accuse, Give Evidence
- NPC-NPC automatic dialogue when scenario changes
- Time controller (days/turns)
- Simulation monitor with live event feed

### 🎭 NPC Studio
- Detailed NPC information management
- Personality traits and belief system
- Trust network visualization
- Memory timeline viewer

### 📖 Scenario Editor
- Visual timeline editor (day-by-day)
- Event configuration
- Participant management
- Information release planning
- Trust effect mapping
- One-click Play button

### 💬 Dialogue Lab
- NPC dialogue testing
- AI debug panel
- Prompt debugger with full context view

### 🧠 Memory Graph
- Memory node visualization
- Conflict simulation between NPCs
- Memory propagation tracking

### 📦 Dataset Builder
- Benchmark dataset generation
- Configurable scenario parameters
- Conflict type selection
- Expected belief configuration

### 📈 Evaluation
- Performance metrics dashboard
- Baseline comparison
- Dialogue replay analysis
- Trust stability tracking

## Tech Stack

- **Framework**: React 18 + Vite 5
- **State Management**: Zustand
- **Styling**: TailwindCSS 3
- **AI Integration**: Aliyun Qwen (通义千问)
- **Build**: Vite Plugin Singlefile (for standalone distribution)

## Quick Start

### Prerequisites

- Node.js >= 18.0.0
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-npc-agent.git
cd multi-npc-agent

# Install dependencies
npm install

# Start development server
npm run dev
```

### Build for Production

```bash
# Build as single HTML file
npm run build

# The output will be in dist/index.html
# This file can be directly opened in a browser!
```

### Configuration

To enable AI-powered NPC dialogue, set your Qwen API key in `src/services/api.js`:

```javascript
const QWEN_API_KEY = 'your-api-key-here'
```

## Project Structure

```
multi-npc-agent/
├── src/
│   ├── components/       # Shared components
│   │   ├── Header.jsx    # Top navigation bar
│   │   └── Sidebar.jsx   # Side menu
│   ├── pages/            # Application pages
│   │   ├── Play.jsx              # Core play mode
│   │   ├── Dashboard.jsx         # Overview dashboard
│   │   ├── NPCStudio.jsx         # NPC management
│   │   ├── ScenarioEditor.jsx    # Scenario timeline editor
│   │   ├── DialogueLab.jsx       # Dialogue testing
│   │   ├── MemoryGraph.jsx       # Memory visualization
│   │   ├── DatasetBuilder.jsx    # Dataset generation
│   │   └── Evaluation.jsx        # Performance metrics
│   ├── store/            # Zustand state management
│   │   └── useStore.js
│   ├── services/         # API service layer
│   │   └── api.js
│   ├── mock/             # Mock data
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## NPC Behavior Simulation

The platform implements a comprehensive NPC behavior simulation system:

1. **Memory Update**: NPCs record player interactions and events
2. **Belief Update**: Confidence levels change based on interactions
3. **Trust Update**: Trust relationships between NPCs evolve
4. **Event Propagation**: Information spreads through the NPC network
5. **Consensus Calculation**: Group opinions converge over time

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
