import { useState, useEffect, useRef } from 'react'
import { useStore } from '../store/useStore'
import { api } from '../services/api'

export default function MemoryGraph() {
  const [selectedNode, setSelectedNode] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [activeTab, setActiveTab] = useState('graph')
  const [simulating, setSimulating] = useState(false)
  const [simulationStep, setSimulationStep] = useState(0)
  const [simulationSteps, setSimulationSteps] = useState([])
  const canvasRef = useRef(null)

  const { memoryNodes, memoryEdges, npcs, applyBackendState } = useStore()
  const thomas = npcs.find((n) => n.id === 'thomas')
  const duran = npcs.find((n) => n.id === 'duran')
  const thomasBelief = thomas?.beliefs?.find((b) => /trustworthy|real/i.test(b.statement)) || thomas?.beliefs?.[0]
  const duranBelief = duran?.beliefs?.find((b) => /fake/i.test(b.statement)) || duran?.beliefs?.[0]

  const filteredNodes = memoryNodes.filter(node =>
    (node.title || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
    (node.description || '').toLowerCase().includes(searchTerm.toLowerCase())
  )

  const getNodeColor = (type) => {
    switch (type) {
      case 'event': return '#00d4ff'
      case 'claim': return '#a855f7'
      case 'evidence': return '#22c55e'
      case 'belief': return '#eab308'
      case 'justification': return '#ec4899'
      default: return '#6b7280'
    }
  }

  const handleStartSimulation = async () => {
    setSimulating(true)
    setSimulationStep(0)
    let steps = []
    try {
      const result = await api.resolveConflict('knight_is_trustworthy', 'knight_is_fake')
      steps = (result.steps || []).map((s) => ({
        label: s.name,
        description: s.detail,
      }))
      if (!steps.length) {
        steps = [{ label: 'Done', description: result.result || 'resolved' }]
      }
      setSimulationSteps(steps)
      if (result.state) applyBackendState(result.state)
    } catch (e) {
      steps = [{ label: 'Error', description: e.message || 'Backend unavailable' }]
      setSimulationSteps(steps)
    }
    let step = 0
    const interval = setInterval(() => {
      step++
      setSimulationStep(step)
      if (step >= steps.length) {
        clearInterval(interval)
        setSimulating(false)
      }
    }, 1500)
  }

  useEffect(() => {
    if (activeTab !== 'graph' || !canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, height)

    const nodePositions = {}
    const nodeRadius = {
      event: 25,
      claim: 20,
      evidence: 15,
      belief: 22,
      justification: 18,
    }

    filteredNodes.forEach((node, index) => {
      const column = Math.floor(index / 4)
      const row = index % 4
      nodePositions[node.id] = {
        x: 100 + column * 200,
        y: 80 + row * 150,
      }
    })

    memoryEdges.forEach(edge => {
      const source = nodePositions[edge.source]
      const target = nodePositions[edge.target]
      if (source && target) {
        ctx.beginPath()
        ctx.strokeStyle = '#3d3d5c'
        ctx.lineWidth = 2
        ctx.moveTo(source.x, source.y)
        ctx.lineTo(target.x, target.y)
        ctx.stroke()

        const midX = (source.x + target.x) / 2
        const midY = (source.y + target.y) / 2
        ctx.fillStyle = '#6b7280'
        ctx.font = '10px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(edge.label, midX, midY - 5)
      }
    })

    filteredNodes.forEach(node => {
      const pos = nodePositions[node.id]
      if (!pos) return

      ctx.beginPath()
      ctx.fillStyle = getNodeColor(node.type)
      ctx.shadowColor = getNodeColor(node.type)
      ctx.shadowBlur = 15
      ctx.arc(pos.x, pos.y, nodeRadius[node.type] || 20, 0, Math.PI * 2)
      ctx.fill()
      ctx.shadowBlur = 0

      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 12px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(node.title.charAt(0), pos.x, pos.y)

      ctx.fillStyle = '#e2e8f0'
      ctx.font = '9px sans-serif'
      ctx.fillText(node.title, pos.x, pos.y + nodeRadius[node.type] + 15)
    })
  }, [filteredNodes, memoryEdges, activeTab])

  const displaySteps = simulationSteps.length ? simulationSteps : [
    { label: 'Receive', description: 'Waiting for Hansson incision…' },
    { label: 'Trust Check', description: 'Checking NPC trust network' },
    { label: 'Conflict Detection', description: 'knight_is_trustworthy vs knight_is_fake' },
    { label: 'Belief Revision', description: 'ATMS nogood + supersede' },
    { label: 'Consensus', description: 'Compute convergence' },
  ]

  const tabs = [
    { id: 'graph', label: 'Memory Graph' },
    { id: 'conflict', label: 'Conflict Simulator' },
  ]

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
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

        <div className="glass-panel rounded-xl p-4 mb-4">
          <div className="flex items-center gap-4">
            <div className="relative flex-1">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search nodes..."
                className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-2 text-white placeholder-gray-500 focus:border-cyber-blue focus:outline-none"
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500">🔍</span>
            </div>
            <div className="flex items-center gap-2">
              <button className="px-3 py-2 bg-dark-card hover:bg-gray-700 text-gray-400 hover:text-white rounded-lg transition-colors">
                Zoom In
              </button>
              <button className="px-3 py-2 bg-dark-card hover:bg-gray-700 text-gray-400 hover:text-white rounded-lg transition-colors">
                Zoom Out
              </button>
              <button className="px-3 py-2 bg-dark-card hover:bg-gray-700 text-gray-400 hover:text-white rounded-lg transition-colors">
                Reset View
              </button>
            </div>
          </div>
          <div className="flex items-center gap-6 mt-3">
            {[
              { type: 'event', label: 'Event', color: '#00d4ff' },
              { type: 'claim', label: 'Claim', color: '#a855f7' },
              { type: 'evidence', label: 'Evidence', color: '#22c55e' },
              { type: 'belief', label: 'Belief', color: '#eab308' },
              { type: 'justification', label: 'Justification', color: '#ec4899' },
            ].map(item => (
              <div key={item.type} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-gray-400 text-sm">{item.label}</span>
              </div>
            ))}
          </div>
        </div>

        {activeTab === 'graph' && (
          <div className="flex-1 glass-panel rounded-xl overflow-hidden">
            <canvas
              ref={canvasRef}
              width={1000}
              height={600}
              className="w-full h-full"
              onClick={(e) => {
                const rect = canvasRef.current.getBoundingClientRect()
                const scaleX = canvasRef.current.width / rect.width
                const scaleY = canvasRef.current.height / rect.height
                const clickX = (e.clientX - rect.left) * scaleX
                const clickY = (e.clientY - rect.top) * scaleY

                for (const node of filteredNodes) {
                  const pos = {
                    x: 100 + Math.floor(filteredNodes.indexOf(node) / 4) * 200,
                    y: 80 + (filteredNodes.indexOf(node) % 4) * 150,
                  }
                  const dist = Math.sqrt((clickX - pos.x) ** 2 + (clickY - pos.y) ** 2)
                  if (dist < 30) {
                    setSelectedNode(node)
                    return
                  }
                }
                setSelectedNode(null)
              }}
            />
          </div>
        )}

        {activeTab === 'conflict' && (
          <div className="flex-1 glass-panel rounded-xl p-6 overflow-y-auto">
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-dark-card/50 rounded-xl p-4">
                <h4 className="text-cyber-green font-semibold mb-3">NPC A</h4>
                <div className="text-center py-6">
                  <div className="w-16 h-16 rounded-full bg-cyber-green/30 flex items-center justify-center mx-auto mb-3">
                    <span className="text-2xl">🛡️</span>
                  </div>
                  <p className="text-white font-bold text-lg">Guard Thomas</p>
                  <p className="text-gray-500 text-sm">Role: Village Guard</p>
                </div>
                <div className="mt-4">
                  <p className="text-gray-400 text-xs mb-2">Belief</p>
                  <p className="text-cyber-green text-sm">{thomasBelief?.statement || '"Knight is trustworthy"'}</p>
                  <div className="w-full bg-dark-surface h-2 rounded-full mt-2">
                    <div className="h-full bg-cyber-green rounded-full" style={{ width: `${Math.round((thomasBelief?.confidence || 0.72) * 100)}%` }} />
                  </div>
                  <p className="text-right text-cyber-green text-xs mt-1">{Math.round((thomasBelief?.confidence || 0.72) * 100)}%</p>
                </div>
              </div>

              <div className="flex flex-col items-center justify-center">
                <div className="text-4xl mb-4">⚔️</div>
                <p className="text-cyber-red font-bold mb-2">Conflict</p>
                <p className="text-gray-400 text-sm text-center">Two NPCs hold opposing beliefs about the knight's authenticity</p>
                <button
                  onClick={handleStartSimulation}
                  disabled={simulating}
                  className="mt-6 px-6 py-3 bg-cyber-blue hover:bg-cyber-blue/80 text-white rounded-xl font-medium transition-all disabled:opacity-50"
                >
                  {simulating ? 'Simulating...' : 'Start Simulation'}
                </button>
                {simulating && (
                  <div className="mt-4 flex gap-2">
                    <button className="px-4 py-2 bg-dark-card hover:bg-gray-700 text-gray-400 rounded-lg text-sm">
                      Pause
                    </button>
                    <button className="px-4 py-2 bg-dark-card hover:bg-gray-700 text-gray-400 rounded-lg text-sm">
                      Step
                    </button>
                  </div>
                )}
              </div>

              <div className="bg-dark-card/50 rounded-xl p-4">
                <h4 className="text-cyber-red font-semibold mb-3">NPC B</h4>
                <div className="text-center py-6">
                  <div className="w-16 h-16 rounded-full bg-cyber-red/30 flex items-center justify-center mx-auto mb-3">
                    <span className="text-2xl">⚒️</span>
                  </div>
                  <p className="text-white font-bold text-lg">Blacksmith Duran</p>
                  <p className="text-gray-500 text-sm">Role: Village Blacksmith</p>
                </div>
                <div className="mt-4">
                  <p className="text-gray-400 text-xs mb-2">Belief</p>
                  <p className="text-cyber-red text-sm">{duranBelief?.statement || '"Knight is fake"'}</p>
                  <div className="w-full bg-dark-surface h-2 rounded-full mt-2">
                    <div className="h-full bg-cyber-red rounded-full" style={{ width: `${Math.round((duranBelief?.confidence || 0.88) * 100)}%` }} />
                  </div>
                  <p className="text-right text-cyber-red text-xs mt-1">{Math.round((duranBelief?.confidence || 0.88) * 100)}%</p>
                </div>
              </div>
            </div>

            <div className="mt-6 bg-dark-card/50 rounded-xl p-4">
              <h4 className="text-cyber-blue font-semibold mb-4">Simulation Progress</h4>
              <div className="space-y-3">
                {displaySteps.map((step, index) => (
                  <div
                    key={index}
                    className={`flex items-center gap-4 p-3 rounded-lg transition-all ${
                      index < simulationStep
                        ? 'bg-cyber-green/20 border border-cyber-green/50'
                        : index === simulationStep && simulating
                          ? 'bg-cyber-yellow/20 border border-cyber-yellow/50 animate-pulse'
                          : 'bg-dark-surface'
                    }`}
                  >
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                      index < simulationStep ? 'bg-cyber-green' : index === simulationStep ? 'bg-cyber-yellow' : 'bg-gray-600'
                    }`}>
                      {index < simulationStep ? '✓' : index + 1}
                    </div>
                    <div className="flex-1">
                      <p className={`font-medium ${
                        index < simulationStep ? 'text-cyber-green' : index === simulationStep ? 'text-cyber-yellow' : 'text-gray-400'
                      }`}>{step.label}</p>
                      <p className="text-gray-500 text-xs">{step.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {selectedNode && (
        <div className="w-80 flex-shrink-0">
          <div className="glass-panel rounded-xl p-4 h-full">
            <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
              <span>🔍</span> Inspector
            </h3>
            <div className="space-y-3">
              <div>
                <p className="text-gray-500 text-xs">Node Type</p>
                <p className="text-white font-medium capitalize">{selectedNode.type}</p>
              </div>
              <div>
                <p className="text-gray-500 text-xs">Title</p>
                <p className="text-white">{selectedNode.title}</p>
              </div>
              <div>
                <p className="text-gray-500 text-xs">Description</p>
                <p className="text-gray-300 text-sm">{selectedNode.description}</p>
              </div>
              {selectedNode.confidence !== undefined && (
                <div>
                  <p className="text-gray-500 text-xs">Confidence</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-dark-surface h-2 rounded-full">
                      <div
                        className="h-full bg-cyber-blue rounded-full"
                        style={{ width: `${selectedNode.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-cyber-blue font-medium text-sm">{Math.round(selectedNode.confidence * 100)}%</span>
                  </div>
                </div>
              )}
              {selectedNode.source && (
                <div>
                  <p className="text-gray-500 text-xs">Source</p>
                  <p className="text-cyber-purple">{selectedNode.source}</p>
                </div>
              )}
              {selectedNode.evidence && (
                <div>
                  <p className="text-gray-500 text-xs">Evidence</p>
                  <p className="text-cyber-green text-sm">{selectedNode.evidence}</p>
                </div>
              )}
              {selectedNode.relatedNPCs && (
                <div>
                  <p className="text-gray-500 text-xs">Related NPCs</p>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {selectedNode.relatedNPCs.map(npc => (
                      <span key={npc} className="px-2 py-1 bg-cyber-blue/20 text-cyber-blue text-xs rounded">
                        {npc}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {selectedNode.timestamp && (
                <div>
                  <p className="text-gray-500 text-xs">Created Time</p>
                  <p className="text-gray-400 text-sm">{new Date(selectedNode.timestamp).toLocaleString()}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
