import { useState } from 'react'
import { useStore } from '../store/useStore'

export default function NPCStudio() {
  const [selectedNPC, setSelectedNPC] = useState('thomas')
  const [activeTab, setActiveTab] = useState('info')
  
  const { npcs } = useStore()
  const npc = npcs.find(n => n.id === selectedNPC)

  const getTrustColor = (value) => {
    if (value >= 0.7) return 'bg-cyber-green'
    if (value >= 0.4) return 'bg-cyber-yellow'
    return 'bg-cyber-red'
  }

  const getTrustTextColor = (value) => {
    if (value >= 0.7) return 'text-cyber-green'
    if (value >= 0.4) return 'text-cyber-yellow'
    return 'text-cyber-red'
  }

  const getBeliefColor = (value) => {
    if (value >= 0.7) return 'bg-cyber-blue'
    if (value >= 0.4) return 'bg-cyber-purple'
    return 'bg-cyber-yellow'
  }

  const tabs = [
    { id: 'info', label: 'Basic Info' },
    { id: 'beliefs', label: 'Beliefs' },
    { id: 'trust', label: 'Trust Network' },
    { id: 'memory', label: 'Memory' },
  ]

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
      <div className="w-72 flex-shrink-0">
        <div className="glass-panel rounded-xl p-4 h-full flex flex-col">
          <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
            <span>🎭</span> NPC List
          </h3>
          <div className="flex-1 overflow-y-auto space-y-2">
            {npcs.map(n => (
              <button
                key={n.id}
                onClick={() => setSelectedNPC(n.id)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                  selectedNPC === n.id
                    ? 'bg-cyber-blue/20 border border-cyber-blue/50 shadow-glow-blue'
                    : 'bg-dark-card/50 hover:bg-dark-card border border-transparent'
                }`}
              >
                <div className="w-10 h-10 rounded-full bg-cyber-purple flex items-center justify-center text-white font-bold">
                  {n.name.charAt(0)}
                </div>
                <div className="text-left">
                  <p className="text-white font-medium">{n.name}</p>
                  <p className="text-gray-500 text-xs">{n.role}</p>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        {npc && (
          <>
            <div className="glass-panel rounded-xl p-6 mb-4">
              <div className="flex items-center gap-6">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyber-purple to-cyber-blue flex items-center justify-center text-white text-3xl font-bold">
                  {npc.name.charAt(0)}
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">{npc.name}</h2>
                  <p className="text-cyber-purple font-medium mt-1">{npc.role}</p>
                  <p className="text-gray-400 text-sm mt-2">{npc.personality}</p>
                </div>
                <div className="ml-auto flex items-center gap-4">
                  <div className="text-center">
                    <p className="text-gray-500 text-xs">Age</p>
                    <p className="text-white font-bold">{npc.age}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-500 text-xs">Beliefs</p>
                    <p className="text-cyber-blue font-bold">{npc.beliefs.length}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-500 text-xs">Trust Connections</p>
                    <p className="text-cyber-green font-bold">{npc.trustNetwork.length}</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex gap-2 mb-4">
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

            <div className="flex-1 glass-panel rounded-xl p-6 overflow-y-auto">
              {activeTab === 'info' && (
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="bg-dark-card/50 rounded-lg p-4">
                      <h4 className="text-cyber-blue font-semibold mb-3">Goals</h4>
                      <div className="space-y-3">
                        <div>
                          <p className="text-gray-500 text-xs">Current Goal</p>
                          <p className="text-white mt-1">{npc.currentGoal}</p>
                        </div>
                        <div>
                          <p className="text-gray-500 text-xs">Hidden Motivation</p>
                          <p className="text-cyber-yellow mt-1">{npc.hiddenMotivation}</p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-dark-card/50 rounded-lg p-4">
                      <h4 className="text-cyber-blue font-semibold mb-3">Traits</h4>
                      <div className="flex flex-wrap gap-2">
                        {npc.traits.map(trait => (
                          <span
                            key={trait}
                            className="px-3 py-1 bg-cyber-purple/20 text-cyber-purple rounded-full text-sm"
                          >
                            {trait}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="bg-dark-card/50 rounded-lg p-4">
                      <h4 className="text-cyber-blue font-semibold mb-3">Emotional State</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-gray-400 text-sm">Mood</span>
                          <span className="text-cyber-green">Neutral</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400 text-sm">Stress Level</span>
                          <span className="text-cyber-yellow">Moderate</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="bg-dark-card/50 rounded-lg p-4">
                      <h4 className="text-cyber-blue font-semibold mb-3">Short-term Memory</h4>
                      <ul className="space-y-2">
                        {npc.shortTermMemory.map((memory, index) => (
                          <li key={index} className="flex items-start gap-2 text-gray-300 text-sm">
                            <span className="text-cyber-blue">•</span>
                            {memory}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="bg-dark-card/50 rounded-lg p-4">
                      <h4 className="text-cyber-blue font-semibold mb-3">Long-term Memory</h4>
                      <ul className="space-y-2">
                        {npc.longTermMemory.map((memory, index) => (
                          <li key={index} className="flex items-start gap-2 text-gray-300 text-sm">
                            <span className="text-cyber-purple">•</span>
                            {memory}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'beliefs' && (
                <div className="space-y-4">
                  {npc.beliefs.map(belief => (
                    <div key={belief.id} className="bg-dark-card/50 rounded-lg p-5">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <p className="text-white font-medium">{belief.statement}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-gray-500 text-xs">Source: {belief.source}</span>
                            <span className="text-gray-500 text-xs">|</span>
                            <span className="text-gray-500 text-xs">Evidence: {belief.evidence}</span>
                          </div>
                        </div>
                        <span className={`text-xl font-bold ${getTrustTextColor(belief.confidence)}`}>
                          {Math.round(belief.confidence * 100)}%
                        </span>
                      </div>
                      <div className="w-full bg-dark-surface h-3 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${getBeliefColor(belief.confidence)}`}
                          style={{ width: `${belief.confidence * 100}%` }}
                        />
                      </div>
                      <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
                        <span>Confidence: {Math.round(belief.confidence * 100)}%</span>
                        <span>Last Updated: {new Date(belief.timestamp).toLocaleDateString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'trust' && (
                <div className="space-y-4">
                  <div className="bg-dark-card/50 rounded-lg p-4">
                    <h4 className="text-cyber-blue font-semibold mb-4">Trust Network Visualization</h4>
                    <div className="relative h-48">
                      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 rounded-full bg-cyber-blue/30 flex items-center justify-center border-2 border-cyber-blue">
                        <span className="text-white font-bold">{npc.name}</span>
                      </div>
                      {npc.trustNetwork.slice(0, 4).map((trust, index) => {
                        const angle = (index * 90 - 90) * (Math.PI / 180)
                        const x = 50 + 35 * Math.cos(angle)
                        const y = 50 + 35 * Math.sin(angle)
                        return (
                          <div key={trust.target} className="absolute" style={{ left: `${x}%`, top: `${y}%`, transform: 'translate(-50%, -50%)' }}>
                            <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold ${trust.trust >= 0.7 ? 'bg-cyber-green' : trust.trust >= 0.4 ? 'bg-cyber-yellow' : 'bg-cyber-red'}`}>
                              {trust.target.charAt(0)}
                            </div>
                            <line
                              x1="50%" y1="50%"
                              x2={x} y2={y}
                              className={`absolute top-1/2 left-1/2 w-full h-full pointer-events-none ${trust.trust >= 0.7 ? 'stroke-cyber-green' : trust.trust >= 0.4 ? 'stroke-cyber-yellow' : 'stroke-cyber-red'}`}
                              style={{
                                strokeWidth: trust.trust * 3,
                                transformOrigin: '50% 50%',
                              }}
                            />
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  <div className="space-y-3">
                    {npc.trustNetwork.map(trust => (
                      <div key={trust.target} className="bg-dark-card/50 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${getTrustColor(trust.trust)}`}>
                              {trust.target.charAt(0)}
                            </div>
                            <span className="text-white font-medium">{trust.target}</span>
                          </div>
                          <span className={`font-bold ${getTrustTextColor(trust.trust)}`}>
                            {Math.round(trust.trust * 100)}%
                          </span>
                        </div>
                        <div className="w-full bg-dark-surface h-2 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${getTrustColor(trust.trust)}`}
                            style={{ width: `${trust.trust * 100}%` }}
                          />
                        </div>
                        <p className="text-gray-500 text-xs mt-2">Reason: {trust.reason}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === 'memory' && (
                <div className="space-y-4">
                  <div className="bg-dark-card/50 rounded-lg p-4">
                    <h4 className="text-cyber-blue font-semibold mb-4">Memory Timeline</h4>
                    <div className="relative">
                      <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-dark-border" />
                      <div className="space-y-6">
                        {[...npc.shortTermMemory, ...npc.longTermMemory].slice(0, 6).map((memory, index) => (
                          <div key={index} className="relative flex items-start gap-4">
                            <div className={`w-3 h-3 rounded-full flex-shrink-0 ${index < npc.shortTermMemory.length ? 'bg-cyber-blue' : 'bg-cyber-purple'}`} />
                            <div>
                              <p className="text-gray-300 text-sm">{memory}</p>
                              <p className={`text-xs mt-1 ${index < npc.shortTermMemory.length ? 'text-cyber-blue' : 'text-cyber-purple'}`}>
                                {index < npc.shortTermMemory.length ? 'Short-term' : 'Long-term'}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
