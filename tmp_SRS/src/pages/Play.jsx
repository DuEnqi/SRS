import { useState, useRef, useEffect } from 'react'
import { useStore } from '../store/useStore'

const ACTIONS = ['Talk', 'Inspect', 'Question', 'Accuse', 'Give Evidence', 'Continue']

export default function Play() {
  const [selectedNPC, setSelectedNPC] = useState('thomas')
  const [playerInput, setPlayerInput] = useState('')
  const [activeAction, setActiveAction] = useState(null)
  const dialogueEndRef = useRef(null)

  const {
    npcs,
    scenarios,
    currentScenario,
    currentDay,
    currentTurn,
    dialogueHistory,
    activityFeed,
    thinkingNPC,
    pendingMemoryUpdates,
    propagationQueue,
    currentConsensus,
    playerAction,
    triggerNPCDialogue,
    setCurrentScenario,
    setDay,
    advanceTurn,
    resetSimulation,
  } = useStore()

  const npc = npcs.find(n => n.id === selectedNPC)
  const maxDays = currentScenario?.totalDays || 7

  useEffect(() => {
    dialogueEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [dialogueHistory])

  const handleAction = (action) => {
    setActiveAction(action)
    playerAction(action, selectedNPC, playerInput)
    setPlayerInput('')
    setTimeout(() => setActiveAction(null), 1500)
  }

  const getActivityColor = (type) => {
    switch (type) {
      case 'player': return 'text-cyber-blue'
      case 'belief': return 'text-cyber-purple'
      case 'trust': return 'text-cyber-green'
      case 'memory': return 'text-cyber-yellow'
      default: return 'text-gray-500'
    }
  }

  const getTimeOfDay = (turn) => {
    const hours = turn
    if (hours >= 5 && hours < 12) return '🌅 Morning'
    if (hours >= 12 && hours < 17) return '☀️ Afternoon'
    if (hours >= 17 && hours < 21) return '🌆 Evening'
    return '🌙 Night'
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="glass-panel rounded-xl p-4">
          <h3 className="text-cyber-blue font-semibold mb-3 flex items-center gap-2">
            <span>📖</span> Scenario
          </h3>
          <select
            value={currentScenario?.id || ''}
            onChange={(e) => {
              const newScenario = scenarios.find(s => s.id === e.target.value)
              if (newScenario) {
                resetSimulation()
                setCurrentScenario(newScenario)
              }
            }}
            className="w-full bg-dark-card border border-dark-border rounded-lg px-3 py-2 text-white text-sm focus:border-cyber-blue focus:outline-none mb-3"
          >
            {scenarios.map(s => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </select>
          <p className="text-white font-medium">{currentScenario?.name}</p>
          <p className="text-gray-500 text-sm mt-1">{currentScenario?.location}</p>
          <p className="text-gray-400 text-sm mt-2 line-clamp-3">{currentScenario?.description}</p>
        </div>

        <div className="glass-panel rounded-xl p-4">
          <h3 className="text-cyber-blue font-semibold mb-3 flex items-center gap-2">
            <span>🎯</span> Current Objective
          </h3>
          <p className="text-gray-300 text-sm">
            {currentScenario?.timeline.find(t => t.day === currentDay)?.description || 'Explore and interact with NPCs'}
          </p>
        </div>

        <div className="glass-panel rounded-xl p-4">
          <h3 className="text-cyber-blue font-semibold mb-3 flex items-center gap-2">
            <span>🎭</span> Participants
          </h3>
          <div className="space-y-2">
            {npcs.slice(0, 4).map(n => (
              <button
                key={n.id}
                onClick={() => setSelectedNPC(n.id)}
                className={`w-full flex items-center gap-3 p-2 rounded-lg transition-all ${
                  selectedNPC === n.id
                    ? 'bg-cyber-blue/20 border border-cyber-blue/50'
                    : 'hover:bg-dark-card'
                }`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                  thinkingNPC === n.id ? 'bg-cyber-yellow animate-pulse' : 'bg-cyber-purple'
                }`}>
                  {n.name.charAt(0)}
                </div>
                <div className="text-left">
                  <p className="text-white text-sm">{n.name}</p>
                  <p className="text-gray-500 text-xs">{n.role}</p>
                </div>
                {thinkingNPC === n.id && (
                  <span className="ml-auto text-xs text-cyber-yellow">Thinking...</span>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        <div className="glass-panel rounded-xl p-4 mb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-cyber-purple font-bold">Day {currentDay}/{maxDays}</span>
              <span className="text-gray-400">|</span>
              <span className="text-gray-300">{getTimeOfDay(currentTurn)}</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setDay(Math.max(1, currentDay - 1))}
                className="px-3 py-1 bg-dark-card hover:bg-gray-700 rounded text-sm transition-colors"
              >
                ←
              </button>
              <span className="text-cyber-blue font-mono">Turn {currentTurn}</span>
              <button
                onClick={() => setDay(Math.min(maxDays, currentDay + 1))}
                className="px-3 py-1 bg-dark-card hover:bg-gray-700 rounded text-sm transition-colors"
              >
                →
              </button>
              <button
                onClick={advanceTurn}
                className="ml-4 px-4 py-1 bg-cyber-blue/20 hover:bg-cyber-blue/30 text-cyber-blue rounded text-sm transition-colors"
              >
                Skip Turn
              </button>
              <button
                onClick={triggerNPCDialogue}
                disabled={thinkingNPC !== null}
                className={`ml-2 px-4 py-1 bg-cyber-yellow/20 hover:bg-cyber-yellow/30 text-cyber-yellow rounded text-sm transition-colors ${thinkingNPC !== null ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                NPC Dialogue
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 glass-panel rounded-xl overflow-hidden flex flex-col bg-gradient-to-b from-dark-card/50 to-dark-surface">
          <div className="absolute inset-0 bg-gradient-to-br from-cyber-blue/10 via-cyber-purple/5 to-dark-surface opacity-50"></div>
          
          <div className="flex-1 overflow-y-auto p-6 relative z-10">
            <div className="space-y-6">
              {dialogueHistory.length === 0 ? (
                <div className="text-center py-12">
                  <p className="text-gray-500 text-lg">Start the conversation by selecting an action below</p>
                  <p className="text-gray-600 text-sm mt-2">Current target: {npc?.name}</p>
                </div>
              ) : (
                dialogueHistory.map(dialogue => (
                  <div key={dialogue.id} className={`fade-in space-y-3 ${dialogue.type === 'npc-dialogue' ? 'border-l-2 border-cyber-yellow pl-4' : ''}`}>
                    {dialogue.type === 'npc-dialogue' ? (
                      <div>
                        {dialogue.participants && (
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-cyber-yellow text-xs font-medium">NPC Dialogue</span>
                            <span className="text-gray-600">↔</span>
                            {dialogue.participants.map(p => {
                              const participantNPC = npcs.find(n => n.id === p)
                              return (
                                <span key={p} className="text-gray-500 text-xs">{participantNPC?.name}</span>
                              )
                            })}
                          </div>
                        )}
                        <div className="flex items-center gap-3 mb-2">
                          <div className="w-10 h-10 rounded-full bg-cyber-yellow flex items-center justify-center text-dark-bg font-bold">
                            {dialogue.speaker.charAt(0)}
                          </div>
                          <div>
                            <p className="text-cyber-yellow font-semibold">{dialogue.speaker}</p>
                          </div>
                        </div>
                        <div className="ml-13 bg-cyber-yellow/10 backdrop-blur-sm rounded-xl p-4 border border-cyber-yellow/30">
                          <p className="text-gray-200 leading-relaxed">{dialogue.text}</p>
                        </div>
                      </div>
                    ) : (
                      <>
                        {(dialogue.playerInput || dialogue.playerAction) && (
                          <div>
                            <div className="flex items-center gap-3 mb-2">
                              <div className="w-10 h-10 rounded-full bg-cyber-blue flex items-center justify-center text-white font-bold">
                                P
                              </div>
                              <div>
                                <p className="text-cyber-blue font-semibold">Player</p>
                                <p className="text-gray-500 text-xs">{dialogue.playerAction && `Action: ${dialogue.playerAction}`}</p>
                              </div>
                            </div>
                            <div className="ml-13 bg-cyber-blue/10 backdrop-blur-sm rounded-xl p-4 border border-cyber-blue/30">
                              <p className="text-gray-200 leading-relaxed">{dialogue.playerInput || `[${dialogue.playerAction}]`}</p>
                            </div>
                          </div>
                        )}
                        <div>
                          <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 rounded-full bg-cyber-purple flex items-center justify-center text-white font-bold">
                              {dialogue.speaker.charAt(0)}
                            </div>
                            <div>
                              <p className="text-cyber-purple font-semibold">{dialogue.speaker}</p>
                            </div>
                          </div>
                          <div className="ml-13 bg-dark-card/80 backdrop-blur-sm rounded-xl p-4 border border-dark-border">
                            <p className="text-gray-200 leading-relaxed">{dialogue.text}</p>
                            {dialogue.classification?.proposition_key && (
                              <p className="text-gray-500 text-xs mt-2 font-mono">
                                [{dialogue.classification.proposition_key}
                                {dialogue.classification.operation ? ` · ${dialogue.classification.operation}` : ''}
                                {dialogue.classification.confidence ? ` · ${dialogue.classification.confidence}` : ''}]
                              </p>
                            )}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                ))
              )}
              <div ref={dialogueEndRef} />
            </div>
          </div>

          <div className="p-4 border-t border-dark-border relative z-10">
            <div className="flex items-end gap-3">
              <div className="flex-1">
                <input
                  type="text"
                  value={playerInput}
                  onChange={(e) => setPlayerInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && playerInput.trim() && !thinkingNPC) {
                      handleAction('Talk')
                    }
                  }}
                  placeholder="Type your message or select an action... (Press Enter to send)"
                  className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:border-cyber-blue focus:outline-none transition-all"
                  disabled={thinkingNPC !== null}
                />
              </div>
              <div className="flex flex-wrap gap-2 max-w-xs">
                {ACTIONS.map(action => (
                  <button
                    key={action}
                    onClick={() => handleAction(action)}
                    disabled={thinkingNPC !== null}
                    className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                      activeAction === action
                        ? 'bg-cyber-blue text-white'
                        : 'bg-dark-card text-gray-300 hover:bg-gray-700 hover:text-white'
                    } ${thinkingNPC !== null ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {action}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-80 flex-shrink-0 space-y-4">
        <div className="glass-panel rounded-xl p-4">
          <h3 className="text-cyber-blue font-semibold mb-3 flex items-center gap-2">
            <span>📊</span> Simulation Monitor
          </h3>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Current Turn</span>
              <span className="text-white font-mono">Day {currentDay} - Turn {currentTurn}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Thinking NPC</span>
              <span className={thinkingNPC ? 'text-cyber-yellow' : 'text-gray-500'}>
                {thinkingNPC ? npcs.find(n => n.id === thinkingNPC)?.name : 'None'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Pending Memory Updates</span>
              <span className="text-cyber-green">{pendingMemoryUpdates.length}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Propagation Queue</span>
              <span className="text-cyber-purple">{propagationQueue.length}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Consensus</span>
              <span className="text-cyber-blue">{Math.round(currentConsensus * 100)}%</span>
            </div>
          </div>
        </div>

        <div className="glass-panel rounded-xl p-4">
          <h3 className="text-cyber-blue font-semibold mb-3 flex items-center gap-2">
            <span>📈</span> Live Event Feed
          </h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {activityFeed.map((activity, index) => (
              <div key={index} className="flex items-start gap-2 text-sm fade-in">
                <span className="text-gray-500 flex-shrink-0">{activity.time}</span>
                <span className={getActivityColor(activity.type)}>{activity.message}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel rounded-xl p-4">
          <h3 className="text-cyber-blue font-semibold mb-3 flex items-center gap-2">
            <span>🎯</span> Current Goal
          </h3>
          <p className="text-gray-300 text-sm">
            {npc?.currentGoal || 'No specific goal'}
          </p>
          <div className="mt-3 pt-3 border-t border-dark-border">
            <p className="text-gray-500 text-xs">Hidden Motivation:</p>
            <p className="text-gray-400 text-sm mt-1">{npc?.hiddenMotivation}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
