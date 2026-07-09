import { useState } from 'react'
import { useStore } from '../store/useStore'
import { api } from '../services/api'

export default function DatasetBuilder() {
  const [formData, setFormData] = useState({
    scenario: 'Fake Knight Incident',
    npcCount: 4,
    conflictType: 'Belief Conflict',
    initialMemory: '',
    playerActions: '',
    expectedBeliefs: '',
  })
  const [generatedDataset, setGeneratedDataset] = useState(null)
  const [generating, setGenerating] = useState(false)

  const { scenarios } = useStore()

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      const dataset = await api.generateDataset(formData)
      setGeneratedDataset(dataset)
    } catch (error) {
      console.error('Dataset generation failed:', error)
    } finally {
      setGenerating(false)
    }
  }

  const handleExport = () => {
    if (!generatedDataset) return
    const dataStr = JSON.stringify(generatedDataset, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${generatedDataset.id}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
      <div className="flex-1">
        <div className="glass-panel rounded-xl p-6">
          <h2 className="text-xl font-bold text-white mb-6">Generate Benchmark Dataset</h2>
          
          <div className="space-y-6">
            <div className="grid grid-cols-3 gap-6">
              <div>
                <label className="text-gray-400 text-sm mb-2 block">Scenario</label>
                <select
                  name="scenario"
                  value={formData.scenario}
                  onChange={handleChange}
                  className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-white focus:border-cyber-blue focus:outline-none"
                >
                  {scenarios.map(s => (
                    <option key={s.id} value={s.name}>{s.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-2 block">NPC Count</label>
                <input
                  type="number"
                  name="npcCount"
                  value={formData.npcCount}
                  onChange={handleChange}
                  min="2"
                  max="10"
                  className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-white focus:border-cyber-blue focus:outline-none"
                />
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-2 block">Conflict Type</label>
                <select
                  name="conflictType"
                  value={formData.conflictType}
                  onChange={handleChange}
                  className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-white focus:border-cyber-blue focus:outline-none"
                >
                  <option value="Belief Conflict">Belief Conflict</option>
                  <option value="Trust Conflict">Trust Conflict</option>
                  <option value="Memory Conflict">Memory Conflict</option>
                  <option value="Goal Conflict">Goal Conflict</option>
                </select>
              </div>
            </div>

            <div>
              <label className="text-gray-400 text-sm mb-2 block">Initial Memory</label>
              <textarea
                name="initialMemory"
                value={formData.initialMemory}
                onChange={handleChange}
                placeholder="Enter initial memory for NPCs...&#10;Example: Knight arrived at the village.&#10;Duran inspected the armor."
                rows={4}
                className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:border-cyber-blue focus:outline-none resize-none"
              />
            </div>

            <div>
              <label className="text-gray-400 text-sm mb-2 block">Player Actions</label>
              <textarea
                name="playerActions"
                value={formData.playerActions}
                onChange={handleChange}
                placeholder="Enter expected player actions...&#10;Example: Talk to Thomas, Give Evidence to Duran, Question Mila"
                rows={4}
                className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:border-cyber-blue focus:outline-none resize-none"
              />
            </div>

            <div>
              <label className="text-gray-400 text-sm mb-2 block">Expected Beliefs</label>
              <textarea
                name="expectedBeliefs"
                value={formData.expectedBeliefs}
                onChange={handleChange}
                placeholder="Enter expected belief outcomes...&#10;Example: Thomas: Knight is fake (0.8)&#10;Duran: Knight is fake (0.95)"
                rows={4}
                className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:border-cyber-blue focus:outline-none resize-none"
              />
            </div>

            <button
              onClick={handleGenerate}
              disabled={generating}
              className="w-full py-4 bg-cyber-blue hover:bg-cyber-blue/80 text-white rounded-xl font-medium transition-all disabled:opacity-50"
            >
              {generating ? 'Generating Dataset...' : 'Generate JSON Dataset'}
            </button>
          </div>
        </div>

        {generatedDataset && (
          <div className="mt-4 glass-panel rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-cyber-blue font-semibold">Generated Dataset</h3>
              <button
                onClick={handleExport}
                className="px-4 py-2 bg-cyber-green/20 hover:bg-cyber-green/30 text-cyber-green rounded-lg text-sm font-medium transition-colors"
              >
                Export JSON
              </button>
            </div>
            <div className="bg-dark-surface rounded-lg p-4 font-mono text-xs text-gray-300 max-h-64 overflow-y-auto">
              <pre>{JSON.stringify(generatedDataset, null, 2)}</pre>
            </div>
          </div>
        )}
      </div>

      <div className="w-80 flex-shrink-0">
        <div className="glass-panel rounded-xl p-4 h-full">
          <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
            <span>📦</span> Dataset Info
          </h3>
          <div className="space-y-4">
            <div className="bg-dark-card/50 rounded-lg p-3">
              <p className="text-gray-500 text-xs">Dataset ID</p>
              <p className="text-cyber-purple font-mono text-sm">{generatedDataset?.id || 'Not generated'}</p>
            </div>

            <div className="bg-dark-card/50 rounded-lg p-3">
              <p className="text-gray-500 text-xs">Generated At</p>
              <p className="text-gray-300 text-sm">{generatedDataset?.generatedAt ? new Date(generatedDataset.generatedAt).toLocaleString() : 'Not generated'}</p>
            </div>

            <div className="bg-dark-card/50 rounded-lg p-3">
              <p className="text-gray-500 text-xs">Scenario</p>
              <p className="text-white">{formData.scenario}</p>
            </div>

            <div className="bg-dark-card/50 rounded-lg p-3">
              <p className="text-gray-500 text-xs">NPC Count</p>
              <p className="text-cyber-blue font-bold">{formData.npcCount}</p>
            </div>

            <div className="bg-dark-card/50 rounded-lg p-3">
              <p className="text-gray-500 text-xs">Conflict Type</p>
              <p className="text-cyber-yellow">{formData.conflictType}</p>
            </div>

            <div className="pt-4 border-t border-dark-border">
              <h4 className="text-gray-400 text-sm mb-2">Previous Runs</h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-2 bg-dark-surface rounded">
                  <span className="text-gray-400 text-sm">Run001</span>
                  <span className="text-gray-500 text-xs">15:30</span>
                </div>
                <div className="flex items-center justify-between p-2 bg-dark-surface rounded">
                  <span className="text-gray-400 text-sm">Run002</span>
                  <span className="text-gray-500 text-xs">15:35</span>
                </div>
                <div className="flex items-center justify-between p-2 bg-dark-surface rounded">
                  <span className="text-gray-400 text-sm">Run003</span>
                  <span className="text-gray-500 text-xs">15:42</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
