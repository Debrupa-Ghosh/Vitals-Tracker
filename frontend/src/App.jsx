import { useState, useEffect } from 'react'
import { submitVitals, getUIConfig } from './api/vitalsApi.js'
import Navbar from './components/Navbar.jsx'
import VitalsForm from './components/VitalsForm.jsx'
import Dashboard from './components/Dashboard.jsx'
import ReportPreview from './components/ReportPreview.jsx'
import ChatAssistant from './components/ChatAssistant.jsx'
import Recommendations from './components/Recommendations.jsx'

export default function App() {
  const [activeTab, setActiveTab] = useState('input')
  const [uiConfig, setUiConfig] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Results from the agent pipeline
  const [results, setResults] = useState(null)

  // Load UI config from Agent 5 on mount
  useEffect(() => {
    getUIConfig()
      .then(setUiConfig)
      .catch(() => {
        // Use fallback if backend not ready
        setUiConfig({ theme: {}, formSchema: { steps: [] }, dashboardLayout: {} })
      })
  }, [])

  const handleSubmit = async (formData) => {
    setLoading(true)
    setError(null)
    try {
      const data = await submitVitals(formData)
      if (!data.valid) {
        setError(data.errors?.join(', ') || 'Validation failed')
        setLoading(false)
        return
      }
      setResults(data)
      setActiveTab('dashboard')
    } catch (err) {
      setError(err.message || 'Failed to submit vitals. Is the backend running?')
    }
    setLoading(false)
  }

  const tabs = [
    { id: 'input', label: 'Input', icon: '📝' },
    { id: 'dashboard', label: 'Dashboard', icon: '📊', disabled: !results },
    { id: 'report', label: 'Report', icon: '📄', disabled: !results },
    { id: 'assistant', label: 'Assistant', icon: '💬', disabled: !results },
  ]

  return (
    <div className="app">
      <Navbar tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="app-main">
        {activeTab === 'input' && (
          <VitalsForm
            schema={uiConfig?.formSchema}
            onSubmit={handleSubmit}
            loading={loading}
            error={error}
          />
        )}

        {activeTab === 'dashboard' && results && (
          <Dashboard
            analysis={results.analysis}
            trends={results.trends}
            doctors={results.doctors}
            layout={uiConfig?.dashboardLayout}
          />
        )}

        {activeTab === 'report' && results && (
          <ReportPreview
            analysis={results.analysis}
            trends={results.trends}
          />
        )}

        {activeTab === 'assistant' && results && (
          <div className="assistant-page">
            <div className="assistant-grid">
              <ChatAssistant
                userId={results.analysis.user_id}
              />
              <div className="assistant-side">
                <Recommendations items={results.recommendations} />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
