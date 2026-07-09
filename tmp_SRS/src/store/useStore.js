import { create } from 'zustand'
import { api } from '../services/api'

const initialState = {
  npcs: [],
  scenarios: [],
  currentScenario: null,
  currentDay: 1,
  currentTurn: 1,
  currentPage: 'dashboard',
  dialogueHistory: [],
  memoryNodes: [],
  memoryEdges: [],
  beliefs: [],
  trustNetwork: {},
  events: [],
  activityFeed: [],
  simulationRunning: false,
  thinkingNPC: null,
  pendingMemoryUpdates: [],
  propagationQueue: [],
  currentConsensus: 0,
  consensusMetrics: {},
  consensusHistory: [],
  stateVersion: 0,
  backendConnected: false,
  evaluationData: {
    memoryAccuracy: 87,
    beliefConsistency: 92,
    conflictResolution: 78,
    narrativeQuality: 84,
    trustStability: 89,
    timelineConsistency: 86,
  },
}

function applyBackendState(set, state, opts = {}) {
  if (!state) return
  const trustNetwork = {}
  ;(state.npcs || []).forEach((npc) => {
    trustNetwork[npc.id] = npc.trustNetwork || []
  })
  set((prev) => {
    const next = {
      ...prev,
      npcs: state.npcs || [],
      scenarios: state.scenarios || [],
      currentScenario: state.currentScenario || state.scenarios?.[0] || null,
      currentDay: state.currentDay ?? 1,
      currentTurn: state.currentTurn ?? 1,
      memoryNodes: state.memoryNodes || [],
      memoryEdges: state.memoryEdges || [],
      events: state.events || [],
      activityFeed: state.activityFeed || [],
      pendingMemoryUpdates: state.pendingMemoryUpdates || [],
      propagationQueue: state.propagationQueue || [],
      currentConsensus: state.currentConsensus ?? 0,
      consensusMetrics: state.consensusMetrics || {},
    consensusHistory: state.consensusHistory || [],
    stateVersion: state.stateVersion ?? 0,
    trustNetwork,
      backendConnected: true,
    }
    if (state.consensusMetrics && Object.keys(state.consensusMetrics).length) {
      const metrics = Object.values(state.consensusMetrics)
      const avgConv = metrics.reduce((s, m) => s + (m.convergence ?? 0), 0) / metrics.length
      const avgVar = metrics.reduce((s, m) => s + (m.variance ?? 0), 0) / metrics.length
      next.evaluationData = {
        ...prev.evaluationData,
        beliefConsistency: Math.round(avgConv * 100),
        conflictResolution: Math.round(Math.max(0, 1 - avgVar) * 100),
      }
    }
    if (!opts.preserveDialogue) {
      next.dialogueHistory = (state.dialogueHistory || []).map((d, i) => ({
        id: d.id || `srv-${i}-${d.timestamp || ''}`,
        ...d,
      }))
    }
    return next
  })
}

export const useStore = create((set, get) => ({
  ...initialState,

  applyBackendState: (state, opts) => applyBackendState(set, state, opts),

  loadData: async () => {
    try {
      await api.checkHealth()
      const state = await api.loadBackendState()
      applyBackendState(set, state)
    } catch (e) {
      console.warn('Backend load failed, using mock fallback', e)
      const [npcs, scenarios, memory] = await Promise.all([
        api.getNPCs(),
        api.getScenarios(),
        api.getMemoryGraph(),
      ])
      set({
        npcs,
        scenarios,
        currentScenario: scenarios[0],
        memoryNodes: memory.nodes || [],
        memoryEdges: memory.edges || [],
        backendConnected: false,
        activityFeed: [
          { time: '—', message: 'Mock mode (backend offline)', type: 'system' },
        ],
      })
    }
  },

  setCurrentScenario: (scenario) => set({ currentScenario: scenario }),
  setCurrentPage: (page) => set({ currentPage: page }),

  createScenario: async () => {
    const payload = {
      name: 'New Scenario',
      location: 'Unknown Location',
      description: 'A new narrative scenario waiting to be developed.',
      participants: [],
      totalDays: 7,
      timeline: Array.from({ length: 7 }, (_, i) => ({
        id: `day-${i + 1}`,
        day: i + 1,
        title: `Day ${i + 1}`,
        description: 'Default day description',
        participants: [],
        informationReleased: [],
        memoryChange: 'No memory changes planned',
        trustEffect: {},
      })),
    }
    try {
      const res = await api.createScenario(payload)
      if (res?.state) applyBackendState(set, res.state)
      else if (res?.scenario) {
        set((state) => ({
          scenarios: [...state.scenarios, res.scenario],
          currentScenario: res.scenario,
        }))
      }
    } catch {
      const newScenario = { id: `scenario-${Date.now()}`, ...payload }
      set((state) => ({
        scenarios: [...state.scenarios, newScenario],
        currentScenario: newScenario,
      }))
    }
    get().addActivity('New scenario created', 'system')
  },

  activateScenario: async (scenarioId) => {
    try {
      const res = await api.activateScenario(scenarioId)
      if (res?.state) applyBackendState(set, res.state)
    } catch {
      const scen = get().scenarios.find((s) => s.id === scenarioId)
      if (scen) set({ currentScenario: scen })
    }
    get().addActivity(`Activated scenario ${scenarioId}`, 'system')
  },

  advanceTurn: async () => {
    const res = await api.advanceTurn()
    if (res?.state) {
      applyBackendState(set, res.state)
    } else {
      const { currentTurn, currentDay } = get()
      const newTurn = currentTurn + 1
      if (newTurn > 24) {
        set({ currentTurn: 1, currentDay: currentDay + 1 })
        setTimeout(() => get().triggerNPCDialogue(), 500)
      } else {
        set({ currentTurn: newTurn })
        if (newTurn % 6 === 0) setTimeout(() => get().triggerNPCDialogue(), 500)
      }
    }
    get().addActivity('Turn advanced', 'system')
  },

  setDay: async (day) => {
    const res = await api.advanceTime(Math.max(0, day - get().currentDay))
    if (res?.state) {
      applyBackendState(set, res.state)
    } else {
      set({ currentDay: day, currentTurn: 1 })
    }
    setTimeout(() => get().triggerNPCDialogue(), 500)
    get().addActivity(`Day changed to ${day}`, 'system')
  },

  playerAction: async (actionType, npcId, text = '') => {
    const { npcs, dialogueHistory, addActivity, stateVersion } = get()
    const npc = npcs.find((n) => n.id === npcId)

    addActivity(`Player ${actionType} with ${npc?.name || 'unknown'}`, 'player')
    set({ thinkingNPC: npcId })

    const context = {
      actionType,
      stateVersion,
      dialogueHistory: dialogueHistory.slice(-8).map((d) => ({
        speaker: d.speaker,
        text: d.text,
        playerAction: d.playerAction,
        playerInput: d.playerInput,
      })),
    }

    const aiResult = await api.generateResponse(npcId, text, context)

    if (aiResult.state) {
      applyBackendState(set, aiResult.state, { preserveDialogue: true })
    }

    const effectiveInput = aiResult.effectiveUtterance || text
    const newDialogue = aiResult.dialogueEntry
      ? {
          id: aiResult.dialogueEntry.id || Date.now(),
          ...aiResult.dialogueEntry,
          classification: aiResult.classification,
        }
      : {
          id: Date.now(),
          speaker: npc?.name || 'NPC',
          text: aiResult.response,
          timestamp: new Date().toISOString(),
          playerAction: actionType,
          playerInput: effectiveInput || undefined,
        }

    set((state) => ({ dialogueHistory: [...state.dialogueHistory, newDialogue] }))

    if (aiResult.memoryUpdate) {
      get().updateMemory(aiResult.memoryUpdate)
    }

    if (!aiResult.state && aiResult.beliefChange) {
      get().updateBelief(npcId, aiResult.beliefChange)
      get().updateTrust(npcId, 'Player', aiResult.trustChange || 0)
    }

    set({ thinkingNPC: null })
    get().advanceTurn()
  },

  updateMemory: (memoryNode) => {
    set((state) => ({
      memoryNodes: [...state.memoryNodes, memoryNode],
      pendingMemoryUpdates: [...state.pendingMemoryUpdates, memoryNode],
    }))
    get().addActivity(`Memory updated: ${memoryNode.title}`, 'memory')
  },

  updateBelief: (npcId, change) => {
    set((state) => ({
      npcs: state.npcs.map((npc) => {
        if (npc.id === npcId && npc.beliefs?.length) {
          return {
            ...npc,
            beliefs: npc.beliefs.map((b) => ({
              ...b,
              confidence: Math.max(0, Math.min(1, b.confidence + change)),
            })),
          }
        }
        return npc
      }),
    }))
    get().addActivity(`Belief updated for ${npcId}`, 'belief')
  },

  updateTrust: (sourceId, targetId, change) => {
    set((state) => ({
      npcs: state.npcs.map((npc) => {
        if (npc.id === sourceId) {
          return {
            ...npc,
            trustNetwork: (npc.trustNetwork || []).map((t) => {
              if (t.target === targetId) {
                return { ...t, trust: Math.max(0, Math.min(1, t.trust + change)) }
              }
              return t
            }),
          }
        }
        return npc
      }),
    }))
    get().addActivity(`Trust updated: ${sourceId} -> ${targetId}`, 'trust')
  },

  propagateEvent: async (event) => {
    const res = await api.propagateEvent(event)
    if (res?.state) {
      applyBackendState(set, res.state)
      return
    }
    set((state) => ({
      events: [...state.events, event],
      propagationQueue: [...state.propagationQueue, event],
    }))
    setTimeout(() => {
      set((state) => ({
        propagationQueue: state.propagationQueue.filter((e) => e.id !== event.id),
      }))
    }, 2000)
  },

  triggerNPCDialogue: async () => {
    const { npcs, dialogueHistory, addActivity } = get()
    if (npcs.length < 2) return

    const shuffled = [...npcs].sort(() => Math.random() - 0.5)
    const npc1 = shuffled[0]
    const npc2 = shuffled[1]

    addActivity(`${npc1.name} and ${npc2.name} are talking...`, 'system')
    set({ thinkingNPC: npc1.id })

    const result = await api.generateNPCDialogue(npc1.id, npc2.id, {
      dialogueHistory: dialogueHistory.slice(-8),
    })

    if (result.state) {
      applyBackendState(set, result.state)
    }

    ;(result.dialogue || []).forEach((line, index) => {
      set((state) => ({
        dialogueHistory: [
          ...state.dialogueHistory,
          {
            id: Date.now() + index,
            speaker: line.speaker,
            text: line.text,
            timestamp: new Date().toISOString(),
            type: 'npc-dialogue',
            participants: [npc1.id, npc2.id],
          },
        ],
      }))
    })

    if (!result.state) {
      Object.entries(result.beliefChanges || {}).forEach(([id, ch]) => {
        get().updateBelief(id, ch)
      })
      Object.entries(result.trustChanges || {}).forEach(([pair, ch]) => {
        const [a, b] = pair.split('->')
        if (a && b) get().updateTrust(a, b, ch)
      })
    }

    set({ thinkingNPC: null })
    addActivity(`NPC dialogue completed: ${npc1.name} ↔ ${npc2.name}`, 'system')
  },

  addActivity: (message, type = 'system') => {
    const now = new Date()
    const t = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`
    set((state) => ({
      activityFeed: [{ time: t, message, type }, ...state.activityFeed].slice(0, 30),
    }))
  },

  setEvaluationData: (data) => set({ evaluationData: { ...get().evaluationData, ...data } }),

  resetSimulation: async () => {
    await get().loadData()
    set((state) => ({
      dialogueHistory: [],
      thinkingNPC: null,
      scenarios: state.scenarios,
      currentScenario: state.scenarios[0],
    }))
  },
}))
