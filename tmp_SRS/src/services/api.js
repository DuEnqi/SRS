/** Same-origin /api — Vite dev proxy → :8765, Vercel → FastAPI serverless */
const GRAPHMEM_API = import.meta.env.VITE_GRAPHMEM_API ?? ''

async function backendFetch(path, options = {}) {
  const url = `${GRAPHMEM_API}${path}`
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!res.ok) {
    const err = new Error(`Backend ${path} → ${res.status}`)
    err.status = res.status
    throw err
  }
  return res.json()
}

export const api = {
  /** Ping backend health (lightweight). */
  checkHealth: async () => backendFetch('/api/health'),

  /** Load full state from GraphMem backend (npcs, memory graph, scenario, day/turn). */
  loadBackendState: async () => backendFetch('/api/state'),

  getNPCs: async () => {
    try {
      const st = await backendFetch('/api/state')
      return st.npcs || []
    } catch {
      const mod = await import('../mock/npc.json')
      return mod.default?.agents || mod.agents || []
    }
  },

  getScenarios: async () => {
    try {
      const st = await backendFetch('/api/state')
      return st.scenarios || (st.currentScenario ? [st.currentScenario] : [])
    } catch {
      const mod = await import('../mock/scenario.json')
      return mod.default?.scenarios || mod.scenarios || []
    }
  },

  getMemoryGraph: async () => {
    try {
      const st = await backendFetch('/api/state')
      return { nodes: st.memoryNodes || [], edges: st.memoryEdges || [] }
    } catch {
      const mod = await import('../mock/memory.json')
      return mod.default || mod
    }
  },

  getDialogues: async () => {
    const mod = await import('../mock/dialogue.json')
    return mod.default?.conversations || mod.conversations || []
  },

  getReplays: async () => {
    const mod = await import('../mock/dialogue.json')
    return mod.default?.replays || mod.replays || []
  },

  generateResponse: async (npcId, playerInput, context) => {
    try {
      const idempotencyKey = context?.idempotencyKey || `play-${npcId}-${Date.now()}`
      const data = await backendFetch('/api/npc/generate', {
        method: 'POST',
        body: JSON.stringify({
          npcId,
          playerInput,
          actionType: context?.actionType || 'Talk',
          idempotencyKey,
          expectedStateVersion: context?.expectedStateVersion ?? context?.stateVersion,
          context: { ...context, dialogueHistory: context?.dialogueHistory || [] },
        }),
      })
      return {
        response: data.response || data.text,
        beliefChange: data.beliefChange ?? 0,
        trustChange: data.trustChange ?? 0,
        memoryUpdate: data.memoryUpdate,
        classification: data.classification,
        analysis: data.analysis,
        effectiveUtterance: data.effectiveUtterance,
        dialogueEntry: data.dialogueEntry,
        rolledBack: data.rolledBack,
        llmOk: data.llmOk,
        stateVersion: data.stateVersion,
        state: data.state,
      }
    } catch (e) {
      console.warn('GraphMem backend unavailable:', e.message)
      return {
        response: `[offline] Consider your words carefully.`,
        beliefChange: 0,
        trustChange: 0,
        state: null,
      }
    }
  },

  generateNPCDialogue: async (npcId1, npcId2, context) => {
    try {
      return await backendFetch('/api/npc/dialogue', {
        method: 'POST',
        body: JSON.stringify({ npcId1, npcId2, context }),
      })
    } catch (e) {
      console.warn('NPC dialogue backend failed:', e.message)
      return { dialogue: [], beliefChanges: {}, trustChanges: {} }
    }
  },

  propagateEvent: async (event) => {
    try {
      return await backendFetch('/api/event/propagate', {
        method: 'POST',
        body: JSON.stringify({
          description: event.description,
          participants: event.participants,
        }),
      })
    } catch (e) {
      return { propagated: false, state: null }
    }
  },

  advanceTime: async (days = 1) => {
    try {
      return await backendFetch('/api/time/advance', {
        method: 'POST',
        body: JSON.stringify({ days }),
      })
    } catch {
      return null
    }
  },

  advanceTurn: async () => {
    try {
      return await backendFetch('/api/time/turn', { method: 'POST', body: '{}' })
    } catch {
      return null
    }
  },

  resolveConflict: async (claimId1 = '', claimId2 = '') => {
    return backendFetch('/api/conflict/resolve', {
      method: 'POST',
      body: JSON.stringify({ claimId1, claimId2 }),
    })
  },

  listScenarios: async () => backendFetch('/api/scenarios'),

  getScenario: async (scenarioId) => backendFetch(`/api/scenarios/${scenarioId}`),

  createScenario: async (scenario) => backendFetch('/api/scenarios', {
    method: 'POST',
    body: JSON.stringify(scenario),
  }),

  updateScenario: async (scenarioId, patch) => backendFetch(`/api/scenarios/${scenarioId}`, {
    method: 'PUT',
    body: JSON.stringify(patch),
  }),

  deleteScenario: async (scenarioId) => backendFetch(`/api/scenarios/${scenarioId}`, {
    method: 'DELETE',
  }),

  activateScenario: async (scenarioId) => backendFetch(`/api/scenarios/${scenarioId}/activate`, {
    method: 'POST',
    body: '{}',
  }),

  analyzeBackground: async () => ({
    origin: 'Greyford Village',
    career: 'Investigator',
    traits: ['Curious', 'Analytical', 'Persistent'],
    keywords: ['knight', 'fake', 'investigation', 'evidence'],
  }),

  generateDataset: async (params) => ({
    id: `dataset-${Date.now()}`,
    ...params,
    generatedAt: new Date().toISOString(),
    data: [],
  }),
}

export default api
