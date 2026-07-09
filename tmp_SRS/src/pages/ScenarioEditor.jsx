import { useState } from 'react'
import { useStore } from '../store/useStore'

export default function ScenarioEditor() {
  const [selectedScenario, setSelectedScenario] = useState(0)
  const [selectedDay, setSelectedDay] = useState(null)

  const { scenarios, createScenario, setCurrentScenario, setCurrentPage } = useStore()
  const scenario = scenarios[selectedScenario]

  const handleSaveJSON = () => {
    const dataStr = JSON.stringify(scenario, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${scenario.name.replace(/\s+/g, '_')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
      <div className="w-72 flex-shrink-0">
        <div className="glass-panel rounded-xl p-4 h-full flex flex-col">
          <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
            <span>📖</span> Scenarios
          </h3>
          <div className="flex-1 overflow-y-auto space-y-2">
            {scenarios.map((s, index) => (
              <button
                key={s.id}
                onClick={() => { setSelectedScenario(index); setSelectedDay(null) }}
                className={`w-full text-left p-3 rounded-lg transition-all ${
                  selectedScenario === index
                    ? 'bg-cyber-blue/20 border border-cyber-blue/50'
                    : 'bg-dark-card/50 hover:bg-dark-card'
                }`}
              >
                <p className="text-white font-medium">{s.name}</p>
                <p className="text-gray-500 text-xs mt-1">{s.location}</p>
                <p className="text-gray-500 text-xs mt-1">{s.totalDays} days</p>
              </button>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-dark-border">
            <button 
              onClick={() => { createScenario(); setSelectedScenario(scenarios.length); }}
              className="w-full py-2 bg-cyber-blue/20 hover:bg-cyber-blue/30 text-cyber-blue rounded-lg text-sm font-medium transition-colors"
            >
              Create New Scenario
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        {scenario && (
          <>
            <div className="glass-panel rounded-xl p-6 mb-4">
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-white">{scenario.name}</h2>
                  <div className="flex items-center gap-4 mt-2">
                    <span className="text-gray-400 text-sm">📍 {scenario.location}</span>
                    <span className="text-gray-400 text-sm">🎭 {scenario.participants.length} participants</span>
                    <span className="text-gray-400 text-sm">📅 {scenario.totalDays} days</span>
                  </div>
                  <p className="text-gray-400 mt-3">{scenario.description}</p>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => { setCurrentScenario(scenario); setCurrentPage('play'); }}
                    className="px-4 py-2 bg-cyber-blue/20 hover:bg-cyber-blue/30 text-cyber-blue rounded-lg text-sm font-medium transition-colors"
                  >
                    🎮 Play
                  </button>
                  <button
                    onClick={handleSaveJSON}
                    className="px-4 py-2 bg-cyber-green/20 hover:bg-cyber-green/30 text-cyber-green rounded-lg text-sm font-medium transition-colors"
                  >
                    Save JSON
                  </button>
                </div>
              </div>
            </div>

            <div className="flex-1 glass-panel rounded-xl overflow-hidden flex">
              <div className="w-1/3 p-6 border-r border-dark-border overflow-y-auto">
                <h3 className="text-cyber-blue font-semibold mb-4 flex items-center gap-2">
                  <span>📅</span> Timeline
                </h3>
                <div className="relative">
                  <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-dark-border" />
                  <div className="space-y-4">
                    {scenario.timeline.map((day) => (
                      <button
                        key={day.id}
                        onClick={() => setSelectedDay(day)}
                        className={`relative flex items-start gap-4 p-3 rounded-lg transition-all w-full text-left ${
                          selectedDay?.id === day.id
                            ? 'bg-cyber-blue/20 border border-cyber-blue/50'
                            : 'bg-dark-card/50 hover:bg-dark-card'
                        }`}
                      >
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0 ${
                          selectedDay?.id === day.id ? 'bg-cyber-blue' : 'bg-cyber-purple'
                        }`}>
                          {day.day}
                        </div>
                        <div>
                          <p className="text-white font-medium">{day.title}</p>
                          <p className="text-gray-500 text-xs mt-1 line-clamp-2">{day.description}</p>
                          <div className="flex flex-wrap gap-1 mt-2">
                            {day.participants.map(p => (
                              <span key={p} className="px-2 py-0.5 bg-dark-surface text-gray-400 text-xs rounded">
                                {p}
                              </span>
                            ))}
                          </div>
                        </div>
                      </button>
                    ))}
                    <div className="relative flex items-start gap-4 p-3">
                      <div className="w-8 h-8 rounded-full bg-cyber-green flex items-center justify-center text-white font-bold flex-shrink-0">
                        E
                      </div>
                      <div>
                        <p className="text-white font-medium">Ending</p>
                        <p className="text-gray-500 text-xs mt-1">Scenario conclusion</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex-1 p-6 overflow-y-auto">
                {selectedDay ? (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-cyber-blue font-semibold mb-3">Day {selectedDay.day}: {selectedDay.title}</h3>
                      <p className="text-gray-300">{selectedDay.description}</p>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-dark-card/50 rounded-xl p-4">
                        <h4 className="text-cyber-purple font-semibold mb-3">Participants</h4>
                        <div className="flex flex-wrap gap-2">
                          {selectedDay.participants.map(p => (
                            <span key={p} className="px-3 py-1 bg-cyber-purple/20 text-cyber-purple rounded-full text-sm">
                              {p}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="bg-dark-card/50 rounded-xl p-4">
                        <h4 className="text-cyber-green font-semibold mb-3">Information Released</h4>
                        <ul className="space-y-2">
                          {selectedDay.informationReleased.map((info, index) => (
                            <li key={index} className="flex items-start gap-2 text-gray-300 text-sm">
                              <span className="text-cyber-green">•</span>
                              {info}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div className="bg-dark-card/50 rounded-xl p-4">
                      <h4 className="text-cyber-blue font-semibold mb-3">Memory Changes</h4>
                      <p className="text-gray-300 text-sm">{selectedDay.memoryChange}</p>
                    </div>

                    <div className="bg-dark-card/50 rounded-xl p-4">
                      <h4 className="text-cyber-yellow font-semibold mb-3">Trust Effects</h4>
                      <div className="space-y-2">
                        {Object.entries(selectedDay.trustEffect).map(([npc, effect]) => (
                          <div key={npc} className="flex items-center justify-between">
                            <span className="text-gray-300 text-sm">{npc}</span>
                            <span className={`font-medium ${effect > 0 ? 'text-cyber-green' : 'text-cyber-red'}`}>
                              {effect > 0 ? '+' : ''}{(effect * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="flex gap-3">
                      <button className="px-4 py-2 bg-cyber-blue/20 hover:bg-cyber-blue/30 text-cyber-blue rounded-lg text-sm font-medium transition-colors">
                        Edit Day
                      </button>
                      <button className="px-4 py-2 bg-cyber-purple/20 hover:bg-cyber-purple/30 text-cyber-purple rounded-lg text-sm font-medium transition-colors">
                        Add Event
                      </button>
                      <button className="px-4 py-2 bg-dark-card hover:bg-gray-700 text-gray-400 hover:text-white rounded-lg text-sm transition-colors">
                        Delete Day
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <p className="text-6xl mb-4">📋</p>
                      <p className="text-gray-500 text-lg">Select a day from the timeline</p>
                      <p className="text-gray-600 text-sm mt-2">Click on a day node to view and edit details</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
