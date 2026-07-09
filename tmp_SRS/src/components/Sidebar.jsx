import { useStore } from '../store/useStore'

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'play', label: 'Play ⭐', icon: '🎮' },
  { id: 'npc-studio', label: 'NPC Studio', icon: '🎭' },
  { id: 'scenario-editor', label: 'Scenario Editor', icon: '📖' },
  { id: 'dialogue-lab', label: 'Dialogue Lab', icon: '💬' },
  { id: 'memory-graph', label: 'Memory Graph', icon: '🧠' },
  { id: 'dataset-builder', label: 'Dataset Builder', icon: '📦' },
  { id: 'evaluation', label: 'Evaluation', icon: '📈' },
]

export default function Sidebar({ currentPage, onNavigate }) {
  const backendConnected = useStore((s) => s.backendConnected)
  const apiLabel = import.meta.env.VITE_GRAPHMEM_API || '/api (same-origin)'

  return (
    <div className="w-64 bg-dark-surface border-r border-dark-border flex flex-col h-screen fixed left-0 top-0">
      <div className="p-6 border-b border-dark-border">
        <h1 className="text-xl font-bold text-cyber-blue glow-text">MultiNPC</h1>
        <p className="text-gray-500 text-sm mt-1">Narrative Platform</p>
      </div>
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
              currentPage === item.id
                ? 'bg-cyber-blue/10 text-cyber-blue border border-cyber-blue/30 shadow-glow-blue'
                : 'text-gray-400 hover:text-white hover:bg-dark-card'
            }`}
          >
            <span className="text-xl">{item.icon}</span>
            <span className="font-medium">{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="p-4 border-t border-dark-border">
        <div className="glass-panel rounded-lg p-3">
          <p className="text-gray-400 text-xs">Connected to:</p>
          <p className={`text-sm mt-1 ${backendConnected ? 'text-cyber-green' : 'text-cyber-yellow'}`}>
            {backendConnected ? `GraphMem-ATMS ${apiLabel}` : 'Mock (backend offline)'}
          </p>
          <div className="flex items-center gap-2 mt-2">
            <span className={`w-2 h-2 rounded-full animate-pulse ${backendConnected ? 'bg-cyber-green' : 'bg-cyber-yellow'}`} />
            <span className="text-gray-500 text-xs">{backendConnected ? 'Running' : 'Fallback'}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
