import React, { useState } from 'react'
import { investigate } from './api/client'

export default function App() {
  const [file, setFile] = useState(null)
  const [accountId, setAccountId] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async () => {
    if (!file || !accountId) return
    setLoading(true); setError(null)
    try {
      const data = await investigate(file, accountId)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      <h1 className="text-2xl font-bold mb-6">AML Investigation System</h1>
      <div className="flex gap-4 mb-6">
        <input type="file" accept=".csv" onChange={e => setFile(e.target.files[0])} className="text-sm" />
        <input className="border border-gray-600 bg-gray-800 px-3 py-1 rounded" placeholder="Account ID" value={accountId} onChange={e => setAccountId(e.target.value)} />
        <button onClick={handleSubmit} disabled={loading} className="bg-blue-600 hover:bg-blue-700 px-4 py-1 rounded disabled:opacity-50">
          {loading ? 'Investigating...' : 'Investigate'}
        </button>
      </div>
      {error && <p className="text-red-400">{error}</p>}
      {result && (
        <div className="bg-gray-800 rounded p-4 mt-4">
          <p><strong>Risk Score:</strong> {result.risk_score}</p>
          <p><strong>Risk Tier:</strong> {result.risk_tier}</p>
          <p><strong>Patterns:</strong> {result.detected_patterns?.join(', ') || 'None'}</p>
          {result.sar_narrative && <pre className="mt-4 text-sm whitespace-pre-wrap">{result.sar_narrative}</pre>}
          {result.exit_summary && <p className="mt-4 text-green-400">{result.exit_summary}</p>}
        </div>
      )}
    </div>
  )
}