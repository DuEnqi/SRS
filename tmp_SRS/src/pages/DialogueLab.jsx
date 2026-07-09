import { useState } from 'react'
import { useStore } from '../store/useStore'
import { api } from '../services/api'

export default function DialogueLab() {
  const [selectedNPC, setSelectedNPC] = useState('thomas')
  const [activeTab, setActiveTab] = useState('conversation')
  const [playerInput, setPlayerInput] = useState('')
  const [debugInfo, setDebugInfo] = useState(null)
  const [generating, setGenerating] = useState(false)

  const { npcs } = useStore()
  const npc = npcs.find(n => n.id === selectedNPC)

  const tabs = [
    { id: 'conversation', label: 'Conversation' },
    { id: 'debug', label: 'AI Debug' },
    { id: 'prompt', label: 'Prompt Debugger' },
  ]

  const handleGenerate = async () => {
    if (!playerInput.trim()) return
    
    setGenerating(true)
    try {
      const context = {
        memory: npc?.shortTermMemory?.slice(0, 3) || [],
        belief: npc?.beliefs?.[0] || {},
        personality: npc?.personality || '',
      }
      
      const result = await api.generateResponse(selectedNPC, playerInput, context)
      setDebugInfo(result)
    } catch (error) {
      console.error('Generation failed:', error)
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
      <div className="w-72 flex-shrink-0">
        <div className="glass-panel rounded-xl p-4 h-full flex flex-col">
          <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
            <span>🎭</span> Select NPC
          </h3>
          <div className="flex-1 overflow-y-auto space-y-2">
            {npcs.map(n => (
              <button
                key={n.id}
                onClick={() => setSelectedNPC(n.id)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                  selectedNPC === n.id
                    ? 'bg-cyber-blue/20 border border-cyber-blue/50'
                    : 'bg-dark-card/50 hover:bg-dark-card'
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

        <div className="flex-1 glass-panel rounded-xl overflow-hidden flex flex-col">
          {activeTab === 'conversation' && (
            <>
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                <div className="flex gap-3">
                  <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold">P</div>
                  <div className="bg-blue-900/30 rounded-xl p-4 max-w-[70%]">
                    <p className="text-blue-400 font-semibold mb-1">Player</p>
                    <p className="text-gray-200">{playerInput || 'Start a conversation...'}</p>
                  </div>
                </div>
                {debugInfo && (
                  <div className="flex gap-3">
                    <div className="w-10 h-10 rounded-full bg-cyber-purple flex items-center justify-center text-white font-bold">
                      {npc?.name.charAt(0)}
                    </div>
                    <div className="bg-dark-card/80 rounded-xl p-4 max-w-[70%]">
                      <p className="text-cyber-purple font-semibold mb-1">{npc?.name}</p>
                      <p className="text-gray-200">{debugInfo.response}</p>
                    </div>
                  </div>
                )}
                {generating && (
                  <div className="flex gap-3">
                    <div className="w-10 h-10 rounded-full bg-cyber-purple flex items-center justify-center text-white font-bold">
                      {npc?.name.charAt(0)}
                    </div>
                    <div className="bg-dark-card/80 rounded-xl p-4">
                      <p className="text-cyber-purple font-semibold mb-1">{npc?.name}</p>
                      <p className="text-gray-400 flex items-center gap-2">
                        <span className="w-2 h-2 bg-cyber-yellow rounded-full animate-pulse" />
                        Generating response...
                      </p>
                    </div>
                  </div>
                )}
              </div>
              <div className="p-4 border-t border-dark-border">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={playerInput}
                    onChange={(e) => setPlayerInput(e.target.value)}
                    placeholder="Enter your message..."
                    className="flex-1 bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:border-cyber-blue focus:outline-none"
                    disabled={generating}
                  />
                  <button
                    onClick={handleGenerate}
                    disabled={generating || !playerInput.trim()}
                    className="px-6 py-3 bg-cyber-blue hover:bg-cyber-blue/80 text-white rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {generating ? 'Generating...' : 'Generate'}
                  </button>
                </div>
                <div className="flex gap-2 mt-3">
                  <button className="px-3 py-1.5 bg-dark-card text-gray-400 hover:text-white rounded-lg text-sm transition-colors">
                    Regenerate
                  </button>
                  <button className="px-3 py-1.5 bg-dark-card text-gray-400 hover:text-white rounded-lg text-sm transition-colors">
                    Compare
                  </button>
                  <button className="px-3 py-1.5 bg-dark-card text-gray-400 hover:text-white rounded-lg text-sm transition-colors">
                    Save Test
                  </button>
                </div>
              </div>
            </>
          )}

          {activeTab === 'debug' && (
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              <div className="bg-dark-card/50 rounded-xl p-4">
                <h4 className="text-cyber-purple font-semibold mb-3">Generation Pipeline</h4>
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <div className="w-8 h-8 rounded-full bg-cyber-blue flex items-center justify-center text-white font-bold">1</div>
                    <div>
                      <p className="text-white text-sm">Player Input</p>
                      <p className="text-gray-500 text-xs">{playerInput || 'Waiting for input...'}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-8 h-8 rounded-full bg-cyber-purple flex items-center justify-center text-white font-bold">2</div>
                    <div>
                      <p className="text-white text-sm">Memory Context</p>
                      <p className="text-gray-500 text-xs">{npc?.shortTermMemory?.[0] || 'Loading...'}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-8 h-8 rounded-full bg-cyber-green flex items-center justify-center text-white font-bold">3</div>
                    <div>
                      <p className="text-white text-sm">Belief Context</p>
                      <p className="text-gray-500 text-xs">{npc?.beliefs?.[0]?.statement || 'Loading...'}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-8 h-8 rounded-full bg-cyber-yellow flex items-center justify-center text-white font-bold">4</div>
                    <div>
                      <p className="text-white text-sm">Generated Response</p>
                      <p className="text-gray-500 text-xs">{debugInfo?.response || 'Not generated yet'}</p>
                    </div>
                  </div>
                  {debugInfo && (
                    <>
                      <div className="flex items-center gap-4">
                        <div className="w-8 h-8 rounded-full bg-cyber-pink flex items-center justify-center text-white font-bold">5</div>
                        <div>
                          <p className="text-white text-sm">Memory Update</p>
                          <p className="text-gray-500 text-xs">{debugInfo.memoryUpdate}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="w-8 h-8 rounded-full bg-cyber-blue flex items-center justify-center text-white font-bold">6</div>
                        <div>
                          <p className="text-white text-sm">Belief Update</p>
                          <p className="text-gray-500 text-xs">Change: {(debugInfo.beliefChange * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="w-8 h-8 rounded-full bg-cyber-green flex items-center justify-center text-white font-bold">7</div>
                        <div>
                          <p className="text-white text-sm">Trust Update</p>
                          <p className="text-gray-500 text-xs">Change: {(debugInfo.trustChange * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'prompt' && (
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              <div className="bg-dark-card/50 rounded-xl p-4">
                <h4 className="text-cyber-purple font-semibold mb-3">System Prompt</h4>
                <div className="bg-dark-surface rounded-lg p-3 font-mono text-xs text-gray-300">
                  <p>You are {npc?.name}, a {npc?.role} in the {npc?.personality} fantasy setting. Your responses should be consistent with your character's personality and background.</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-dark-card/50 rounded-xl p-4">
                  <h4 className="text-cyber-blue font-semibold mb-3">Memory Prompt</h4>
                  <div className="bg-dark-surface rounded-lg p-3 font-mono text-xs text-gray-300">
                    {npc?.shortTermMemory?.map((m, i) => (
                      <p key={i}>{i + 1}. {m}</p>
                    ))}
                  </div>
                </div>

                <div className="bg-dark-card/50 rounded-xl p-4">
                  <h4 className="text-cyber-green font-semibold mb-3">Belief Prompt</h4>
                  <div className="bg-dark-surface rounded-lg p-3 font-mono text-xs text-gray-300">
                    {npc?.beliefs?.map((b, i) => (
                      <p key={i}>{b.statement} (confidence: {Math.round(b.confidence * 100)}%)</p>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-dark-card/50 rounded-xl p-4">
                <h4 className="text-cyber-yellow font-semibold mb-3">Player Prompt</h4>
                <div className="bg-dark-surface rounded-lg p-3 font-mono text-xs text-gray-300">
                  Player says: "{playerInput}"
                </div>
              </div>

              <div className="bg-dark-card/50 rounded-xl p-4">
                <h4 className="text-cyber-pink font-semibold mb-3">Full Combined Prompt</h4>
                <div className="bg-dark-surface rounded-lg p-3 font-mono text-xs text-gray-300 max-h-64 overflow-y-auto">
                  <p>SYSTEM: You are {npc?.name}, a {npc?.role}. {npc?.personality}.</p>
                  <p className="mt-2">MEMORY: {npc?.shortTermMemory?.join('; ')}</p>
                  <p className="mt-2">BELIEFS: {npc?.beliefs?.map(b => `${b.statement}(${Math.round(b.confidence * 100)}%)`).join('; ')}</p>
                  <p className="mt-2">PLAYER: {playerInput}</p>
                  <p className="mt-2">RESPONSE:</p>
                </div>
              </div>

              <div className="flex gap-3">
                <button className="px-4 py-2 bg-dark-card text-gray-400 hover:text-white rounded-lg text-sm transition-colors">
                  Version A
                </button>
                <button className="px-4 py-2 bg-dark-card text-gray-400 hover:text-white rounded-lg text-sm transition-colors">
                  Version B
                </button>
                <button className="px-4 py-2 bg-cyber-blue/20 text-cyber-blue hover:bg-cyber-blue/30 rounded-lg text-sm transition-colors">
                  Diff Compare
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
