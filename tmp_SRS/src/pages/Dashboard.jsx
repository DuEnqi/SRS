import { useStore } from '../store/useStore'

export default function Dashboard() {
  const {
    npcs,
    currentScenario,
    memoryNodes,
    events,
    activityFeed,
    evaluationData,
    currentDay,
    currentTurn,
  } = useStore()

  const getActivityColor = (type) => {
    switch (type) {
      case 'player': return 'text-cyber-blue'
      case 'belief': return 'text-cyber-purple'
      case 'trust': return 'text-cyber-green'
      case 'memory': return 'text-cyber-yellow'
      default: return 'text-gray-500'
    }
  }

  const stats = [
    { label: 'Active NPCs', value: npcs.length, icon: '🎭', color: 'text-cyber-purple', bgColor: 'bg-cyber-purple/20' },
    { label: 'Memory Nodes', value: memoryNodes.length, icon: '🧠', color: 'text-cyber-blue', bgColor: 'bg-cyber-blue/20' },
    { label: 'Events', value: events.length + 12, icon: '📊', color: 'text-cyber-green', bgColor: 'bg-cyber-green/20' },
    { label: 'Simulation Day', value: `${currentDay}/7`, icon: '📅', color: 'text-cyber-yellow', bgColor: 'bg-cyber-yellow/20' },
  ]

  const systemMetrics = [
    { name: 'Memory Accuracy', value: evaluationData.memoryAccuracy, trend: '+2%' },
    { name: 'Belief Consistency', value: evaluationData.beliefConsistency, trend: '+1%' },
    { name: 'Conflict Resolution', value: evaluationData.conflictResolution, trend: '-1%' },
    { name: 'Narrative Quality', value: evaluationData.narrativeQuality, trend: '0%' },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Welcome to MultiNPC</h1>
          <p className="text-gray-500 mt-1">Narrative Development Platform</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="glass-panel rounded-lg px-4 py-2">
            <span className="text-gray-500 text-sm">Current Scenario:</span>
            <span className="text-cyber-blue ml-2 font-medium">{currentScenario?.name}</span>
          </div>
          <div className="glass-panel rounded-lg px-4 py-2">
            <span className="text-gray-500 text-sm">Turn:</span>
            <span className="text-cyber-purple ml-2 font-mono">{currentDay}-{currentTurn}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <div key={index} className="glass-panel rounded-xl p-5 hover:shadow-glow-blue transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">{stat.label}</p>
                <p className={`text-3xl font-bold ${stat.color} mt-2`}>{stat.value}</p>
              </div>
              <div className={`${stat.bgColor} w-12 h-12 rounded-xl flex items-center justify-center`}>
                <span className="text-2xl">{stat.icon}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 glass-panel rounded-xl p-6">
          <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
            <span>📜</span> Recent Activity
          </h3>
          <div className="space-y-3">
            {activityFeed.slice(0, 8).map((activity, index) => (
              <div key={index} className="flex items-start gap-3 p-3 rounded-lg hover:bg-dark-card/50 transition-colors fade-in">
                <div className="w-2 h-2 rounded-full mt-2" style={{
                  backgroundColor: activity.type === 'player' ? '#00d4ff' :
                    activity.type === 'belief' ? '#a855f7' :
                    activity.type === 'trust' ? '#22c55e' :
                    activity.type === 'memory' ? '#eab308' : '#6b7280'
                }} />
                <div className="flex-1">
                  <p className={`text-gray-200 text-sm ${getActivityColor(activity.type)}`}>
                    {activity.message}
                  </p>
                  <p className="text-gray-500 text-xs mt-1">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel rounded-xl p-6">
          <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
            <span>📊</span> System Health
          </h3>
          <div className="space-y-4">
            {systemMetrics.map((metric, index) => (
              <div key={index}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-400 text-sm">{metric.name}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-white font-medium">{metric.value}%</span>
                    <span className={`text-xs ${metric.trend.startsWith('+') ? 'text-cyber-green' : metric.trend === '0%' ? 'text-gray-500' : 'text-cyber-red'}`}>
                      {metric.trend}
                    </span>
                  </div>
                </div>
                <div className="w-full bg-dark-card h-2 rounded-full overflow-hidden">
                  <div 
                    className="h-full rounded-full transition-all duration-500"
                    style={{ 
                      width: `${metric.value}%`,
                      backgroundColor: index === 0 ? '#00d4ff' : index === 1 ? '#a855f7' : index === 2 ? '#eab308' : '#22c55e'
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <div className="col-span-2 glass-panel rounded-xl p-6">
          <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
            <span>🎭</span> NPC Overview
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {npcs.slice(0, 6).map(npc => (
              <div key={npc.id} className="bg-dark-card/50 rounded-lg p-4 hover:bg-dark-card transition-colors">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-cyber-purple flex items-center justify-center text-white font-bold">
                    {npc.name.charAt(0)}
                  </div>
                  <div>
                    <p className="text-white font-medium">{npc.name}</p>
                    <p className="text-gray-500 text-xs">{npc.role}</p>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-500">Trust in Player</span>
                    <span className="text-cyber-green">{Math.round((npc.trustNetwork.find(t => t.target === 'Player')?.trust || 0) * 100)}%</span>
                  </div>
                  <div className="w-full bg-dark-surface h-1.5 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-cyber-green rounded-full"
                      style={{ width: `${(npc.trustNetwork.find(t => t.target === 'Player')?.trust || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel rounded-xl p-6">
          <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
            <span>🔮</span> Next Objective
          </h3>
          <div className="bg-gradient-to-br from-cyber-purple/20 to-cyber-blue/20 rounded-lg p-4">
            <p className="text-cyber-purple font-medium mb-2">
              {currentScenario?.timeline.find(t => t.day === currentDay + 1)?.title || 'Continue Investigation'}
            </p>
            <p className="text-gray-400 text-sm">
              {currentScenario?.timeline.find(t => t.day === currentDay + 1)?.description || 'Proceed with the narrative exploration'}
            </p>
          </div>
          <div className="mt-4 pt-4 border-t border-dark-border">
            <p className="text-gray-500 text-sm">Days remaining: <span className="text-cyber-yellow">{7 - currentDay}</span></p>
          </div>
        </div>

        <div className="glass-panel rounded-xl p-6">
          <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
            <span>💡</span> Quick Stats
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Total Beliefs</span>
              <span className="text-white font-mono">{npcs.reduce((sum, n) => sum + n.beliefs.length, 0)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Average Trust</span>
              <span className="text-cyber-green font-mono">
                {Math.round(npcs.reduce((sum, n) => sum + (n.trustNetwork.find(t => t.target === 'Player')?.trust || 0), 0) / npcs.length * 100)}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Active Events</span>
              <span className="text-cyber-yellow font-mono">{events.length + 5}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Memory Accuracy</span>
              <span className="text-cyber-blue font-mono">{evaluationData.memoryAccuracy}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
