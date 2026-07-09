const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'play', label: 'Play', icon: '🎮' },
  { id: 'npc-studio', label: 'NPC Studio', icon: '🎭' },
  { id: 'scenario-editor', label: 'Scenario Editor', icon: '📖' },
  { id: 'dialogue-lab', label: 'Dialogue Lab', icon: '💬' },
  { id: 'memory-graph', label: 'Memory Graph', icon: '🧠' },
  { id: 'dataset-builder', label: 'Dataset Builder', icon: '📦' },
  { id: 'evaluation', label: 'Evaluation', icon: '📈' },
]

export default function Header({ currentPage }) {
  const pageTitle = navItems.find(item => item.id === currentPage)?.label || 'Dashboard'

  return (
    <div className="h-16 bg-dark-surface border-b border-dark-border flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-cyber-green rounded-full animate-pulse"></div>
          <h2 className="text-lg font-semibold text-white">{pageTitle}</h2>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="relative">
          <input
            type="text"
            placeholder="Search NPCs, scenarios..."
            className="bg-dark-card border border-dark-border rounded-lg px-4 py-2 text-sm text-white placeholder-gray-500 focus:border-cyber-blue focus:outline-none w-64 transition-all"
          />
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500">🔍</span>
        </div>
        <div className="flex items-center gap-3 pl-4 border-l border-dark-border">
          <div className="relative">
            <span className="text-xl">🔔</span>
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-cyber-red rounded-full text-xs flex items-center justify-center text-white">3</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-cyber-purple rounded-full flex items-center justify-center text-white font-bold">U</div>
            <span className="text-gray-300 text-sm">Developer</span>
          </div>
        </div>
      </div>
    </div>
  )
}
