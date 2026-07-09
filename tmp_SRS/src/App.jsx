import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import Dashboard from './pages/Dashboard'
import Play from './pages/Play'
import NPCStudio from './pages/NPCStudio'
import ScenarioEditor from './pages/ScenarioEditor'
import DialogueLab from './pages/DialogueLab'
import MemoryGraph from './pages/MemoryGraph'
import DatasetBuilder from './pages/DatasetBuilder'
import Evaluation from './pages/Evaluation'
import { useStore } from './store/useStore'

export default function App() {
  const [loading, setLoading] = useState(true)
  const loadData = useStore(state => state.loadData)
  const currentPage = useStore(state => state.currentPage)
  const setCurrentPage = useStore(state => state.setCurrentPage)

  useEffect(() => {
    const initialize = async () => {
      try {
        await loadData()
      } catch (error) {
        console.error('Failed to load data:', error)
      } finally {
        setLoading(false)
      }
    }
    initialize()
  }, [loadData])

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-cyber-blue border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-cyber-blue text-lg">Initializing MultiNPC Platform...</p>
          <p className="text-gray-500 text-sm mt-2">Connecting GraphMem-ATMS backend…</p>
        </div>
      </div>
    )
  }

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />
      case 'play':
        return <Play />
      case 'npc-studio':
        return <NPCStudio />
      case 'scenario-editor':
        return <ScenarioEditor />
      case 'dialogue-lab':
        return <DialogueLab />
      case 'memory-graph':
        return <MemoryGraph />
      case 'dataset-builder':
        return <DatasetBuilder />
      case 'evaluation':
        return <Evaluation />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="min-h-screen bg-dark-bg">
      <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />
      <div className="ml-64 flex flex-col min-h-screen">
        <Header currentPage={currentPage} />
        <main className="flex-1 overflow-y-auto p-6">
          {renderPage()}
        </main>
      </div>
    </div>
  )
}
