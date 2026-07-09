import { useState } from 'react'
import { useStore } from '../store/useStore'

export default function Evaluation() {
  const [activeTab, setActiveTab] = useState('overview')
  const { evaluationData, npcs } = useStore()

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'comparison', label: 'Baseline Compare' },
    { id: 'replay', label: 'Dialogue Replay' },
  ]

  const metrics = [
    { name: 'Memory Accuracy', value: evaluationData.memoryAccuracy, color: '#00d4ff', icon: '🧠' },
    { name: 'Belief Consistency', value: evaluationData.beliefConsistency, color: '#a855f7', icon: '💭' },
    { name: 'Conflict Resolution', value: evaluationData.conflictResolution, color: '#eab308', icon: '⚔️' },
    { name: 'Narrative Quality', value: evaluationData.narrativeQuality, color: '#22c55e', icon: '📖' },
    { name: 'Trust Stability', value: evaluationData.trustStability, color: '#ec4899', icon: '🤝' },
    { name: 'Timeline Consistency', value: evaluationData.timelineConsistency, color: '#6366f1', icon: '⏱️' },
  ]

  const baselineData = {
    'No Sharing': { memory: 65, conflict: 55, belief: 70, narrative: 60, trust: 50, timeline: 62 },
    'Trust Propagation': { memory: 78, conflict: 70, belief: 82, narrative: 72, trust: 75, timeline: 75 },
    STALE: { memory: 82, conflict: 75, belief: 85, narrative: 78, trust: 78, timeline: 80 },
    'Our Method': { memory: evaluationData.memoryAccuracy, conflict: evaluationData.conflictResolution, belief: evaluationData.beliefConsistency, narrative: evaluationData.narrativeQuality, trust: evaluationData.trustStability, timeline: evaluationData.timelineConsistency },
  }

  const replays = [
    { id: 'r1', name: 'Thomas Interview - Day 3', date: '2024-01-15', duration: '120 seconds', turns: 8 },
    { id: 'r2', name: 'Duran Armor Analysis', date: '2024-01-15', duration: '90 seconds', turns: 6 },
    { id: 'r3', name: 'Mila Conversation', date: '2024-01-15', duration: '150 seconds', turns: 10 },
    { id: 'r4', name: 'Gareth Information Gathering', date: '2024-01-16', duration: '180 seconds', turns: 12 },
  ]

  const generateChartBars = (metricKey) => {
    return Object.entries(baselineData).map(([method, values]) => (
      <div key={method} className="flex items-center gap-3">
        <span className="text-gray-400 text-sm w-24">{method}</span>
        <div className="flex-1 h-6 bg-dark-surface rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${method === 'Our Method' ? 'bg-cyber-blue' : 'bg-gray-600'}`}
            style={{ width: `${values[metricKey]}%` }}
          />
        </div>
        <span className={`text-sm font-medium ${method === 'Our Method' ? 'text-cyber-blue' : 'text-gray-500'}`}>
          {values[metricKey]}%
        </span>
      </div>
    ))
  }

  return (
    <div className="h-[calc(100vh-8rem)] overflow-y-auto">
      <div className="flex gap-2 mb-6">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-cyber-blue text-white'
                : 'bg-dark-card text-gray-400 hover:bg-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && (
        <div className="space-y-6">
          <div className="grid grid-cols-6 gap-4">
            {metrics.map((metric, index) => (
              <div key={index} className="glass-panel rounded-xl p-5 text-center hover:shadow-glow-blue transition-shadow">
                <div className="text-3xl mb-3">{metric.icon}</div>
                <p className="text-gray-400 text-sm mb-2">{metric.name}</p>
                <p className="text-3xl font-bold" style={{ color: metric.color }}>{metric.value}%</p>
                <div className="mt-3 w-full bg-dark-card h-2 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${metric.value}%`, backgroundColor: metric.color }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="glass-panel rounded-xl p-6">
            <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
              <span>📊</span> NPC Performance
            </h3>
            <div className="grid grid-cols-3 gap-4">
              {npcs.slice(0, 6).map(npc => (
                <div key={npc.id} className="bg-dark-card/50 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-full bg-cyber-purple flex items-center justify-center text-white font-bold">
                      {npc.name.charAt(0)}
                    </div>
                    <div>
                      <p className="text-white font-medium">{npc.name}</p>
                      <p className="text-gray-500 text-xs">{npc.role}</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">Belief Accuracy</span>
                      <span className="text-cyber-blue">{Math.round(npc.beliefs[0]?.confidence * 100 || 0)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">Trust Stability</span>
                      <span className="text-cyber-green">{Math.round((npc.trustNetwork.find(t => t.target === 'Player')?.trust || 0) * 100)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">Memory Count</span>
                      <span className="text-cyber-purple">{npc.shortTermMemory.length + npc.longTermMemory.length}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="glass-panel rounded-xl p-6">
              <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
                <span>📈</span> Performance Trends
              </h3>
              <div className="relative h-48">
                <svg className="w-full h-full" viewBox="0 0 400 200">
                  <defs>
                    <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#00d4ff" />
                      <stop offset="100%" stopColor="#a855f7" />
                    </linearGradient>
                  </defs>
                  <path
                    d="M 20 160 Q 80 140 140 120 T 260 80 T 380 60"
                    fill="none"
                    stroke="url(#lineGradient)"
                    strokeWidth="3"
                  />
                  {[20, 80, 140, 200, 260, 320, 380].map((x, i) => (
                    <circle key={i} cx={x} cy={160 - i * 15} r="5" fill="#00d4ff" />
                  ))}
                  {['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'].map((label, i) => (
                    <text key={i} x={20 + i * 60} y={190} textAnchor="middle" className="text-gray-500 text-xs" fill="#6b7280">{label}</text>
                  ))}
                </svg>
              </div>
            </div>

            <div className="glass-panel rounded-xl p-6">
              <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
                <span>🔍</span> Key Insights
              </h3>
              <div className="space-y-3">
                <div className="flex items-start gap-3 p-3 bg-dark-card/50 rounded-lg">
                  <span className="text-cyber-green text-xl">✓</span>
                  <div>
                    <p className="text-white text-sm font-medium">High Memory Accuracy</p>
                    <p className="text-gray-500 text-xs">System demonstrates strong memory retention capabilities</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-dark-card/50 rounded-lg">
                  <span className="text-cyber-yellow text-xl">⚠️</span>
                  <div>
                    <p className="text-white text-sm font-medium">Conflict Resolution Needs Improvement</p>
                    <p className="text-gray-500 text-xs">Lower score indicates potential issues in resolving belief conflicts</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-dark-card/50 rounded-lg">
                  <span className="text-cyber-blue text-xl">⭐</span>
                  <div>
                    <p className="text-white text-sm font-medium">Strong Belief Consistency</p>
                    <p className="text-gray-500 text-xs">NPCs maintain consistent beliefs throughout the narrative</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'comparison' && (
        <div className="space-y-6">
          <div className="glass-panel rounded-xl p-6">
            <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
              <span>📊</span> Baseline Comparison
            </h3>
            <div className="grid grid-cols-2 gap-6">
              {[
                { key: 'memory', label: 'Memory Accuracy' },
                { key: 'conflict', label: 'Conflict Resolution' },
                { key: 'belief', label: 'Belief Consistency' },
                { key: 'narrative', label: 'Narrative Quality' },
              ].map(metric => (
                <div key={metric.key}>
                  <p className="text-gray-400 text-sm mb-3">{metric.label}</p>
                  <div className="space-y-2">
                    {generateChartBars(metric.key)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel rounded-xl p-6">
            <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
              <span>📈</span> Method Comparison Summary
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-dark-border">
                    <th className="text-left py-3 px-4 text-gray-400 text-sm font-medium">Metric</th>
                    <th className="text-center py-3 px-4 text-gray-400 text-sm font-medium">No Sharing</th>
                    <th className="text-center py-3 px-4 text-gray-400 text-sm font-medium">Trust Propagation</th>
                    <th className="text-center py-3 px-4 text-gray-400 text-sm font-medium">STALE</th>
                    <th className="text-center py-3 px-4 text-cyber-blue text-sm font-medium">Our Method</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { key: 'memory', label: 'Memory Accuracy' },
                    { key: 'conflict', label: 'Conflict Resolution' },
                    { key: 'belief', label: 'Belief Consistency' },
                    { key: 'narrative', label: 'Narrative Quality' },
                    { key: 'trust', label: 'Trust Stability' },
                    { key: 'timeline', label: 'Timeline Consistency' },
                  ].map(metric => (
                    <tr key={metric.key} className="border-b border-dark-border/50">
                      <td className="py-3 px-4 text-gray-300 text-sm">{metric.label}</td>
                      {Object.values(baselineData).map((values, i) => (
                        <td key={i} className={`text-center py-3 px-4 font-medium ${i === 3 ? 'text-cyber-blue' : 'text-gray-400'}`}>
                          {values[metric.key]}%
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'replay' && (
        <div className="space-y-6">
          <div className="glass-panel rounded-xl p-6">
            <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
              <span>🎬</span> Saved Dialogues
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {replays.map(replay => (
                <div key={replay.id} className="bg-dark-card/50 rounded-xl p-4 hover:bg-dark-card transition-colors cursor-pointer">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <p className="text-white font-medium">{replay.name}</p>
                      <p className="text-gray-500 text-xs">{replay.date}</p>
                    </div>
                    <div className="flex gap-2">
                      <button className="px-3 py-1 bg-cyber-blue/20 hover:bg-cyber-blue/30 text-cyber-blue rounded-lg text-sm font-medium transition-colors">
                        Replay
                      </button>
                      <button className="px-3 py-1 bg-dark-surface hover:bg-gray-700 text-gray-400 hover:text-white rounded-lg text-sm font-medium transition-colors">
                        Export
                      </button>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="text-gray-500">⏱️ {replay.duration}</span>
                    <span className="text-gray-500">🔄 {replay.turns} turns</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel rounded-xl p-6">
            <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
              <span>🔍</span> Replay Comparison
            </h3>
            <div className="flex gap-3 mb-4">
              <button className="px-4 py-2 bg-cyber-blue/20 hover:bg-cyber-blue/30 text-cyber-blue rounded-lg text-sm font-medium transition-colors">
                Select Run A
              </button>
              <button className="px-4 py-2 bg-cyber-purple/20 hover:bg-cyber-purple/30 text-cyber-purple rounded-lg text-sm font-medium transition-colors">
                Select Run B
              </button>
              <button className="px-4 py-2 bg-dark-card hover:bg-gray-700 text-gray-400 hover:text-white rounded-lg text-sm font-medium transition-colors">
                Compare
              </button>
            </div>
            <div className="text-center py-12">
              <p className="text-6xl mb-4">🎬</p>
              <p className="text-gray-500 text-lg">Select two runs to compare</p>
              <p className="text-gray-600 text-sm mt-2">Compare NPC behavior, memory updates, and belief changes between different playthroughs</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
