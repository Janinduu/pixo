import { useState, useEffect } from 'react'
import './index.css'

const API = '/api'

function Nav({ page, setPage }) {
  const tabs = [
    ['models', 'Models'],
    ['run', 'Run'],
    ['jobs', 'Jobs'],
    ['hardware', 'Hardware'],
  ]
  return (
    <nav className="bg-slate-800 border-b border-slate-700 px-6 py-3 flex items-center gap-6">
      <h1 className="text-xl font-bold text-cyan-400 mr-4 cursor-pointer" onClick={() => setPage('models')}>pixo</h1>
      {tabs.map(([key, label]) => (
        <button key={key} onClick={() => setPage(key)}
          className={`px-3 py-1 rounded text-sm ${page === key ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'}`}>
          {label}
        </button>
      ))}
    </nav>
  )
}

function ModelCard({ model, onRun }) {
  const taskColors = {
    'detection': 'bg-green-600', 'segmentation': 'bg-purple-600',
    'depth-estimation': 'bg-blue-600', 'vision-language': 'bg-orange-600',
    'video-tracking-segmentation': 'bg-pink-600',
  }
  return (
    <div className="bg-slate-800 rounded-lg p-5 border border-slate-700 hover:border-cyan-500 transition">
      <div className="flex justify-between items-start mb-3">
        <h3 className="text-lg font-semibold text-white">{model.name}</h3>
        {model.downloaded && <span className="text-xs bg-green-700 text-green-200 px-2 py-0.5 rounded">downloaded</span>}
      </div>
      <span className={`text-xs px-2 py-0.5 rounded text-white ${taskColors[model.task] || 'bg-slate-600'}`}>{model.task}</span>
      <p className="text-sm text-slate-400 mt-3">{model.description}</p>
      <div className="flex justify-between items-center mt-4">
        <span className="text-xs text-slate-500">{model.default_size_mb}MB</span>
        <button onClick={() => onRun(model.name)}
          className="px-3 py-1 bg-cyan-600 hover:bg-cyan-500 text-white text-sm rounded">
          Run
        </button>
      </div>
    </div>
  )
}

function ModelsPage({ setPage, setRunModel }) {
  const [models, setModels] = useState([])
  useEffect(() => { fetch(`${API}/models`).then(r => r.json()).then(setModels) }, [])
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Available Models</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {models.map(m => <ModelCard key={m.name} model={m} onRun={(name) => { setRunModel(name); setPage('run') }} />)}
      </div>
    </div>
  )
}

function RunPage({ initialModel }) {
  const [model, setModel] = useState(initialModel || 'yolov8')
  const [file, setFile] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [models, setModels] = useState([])

  useEffect(() => { fetch(`${API}/models`).then(r => r.json()).then(setModels) }, [])
  useEffect(() => { if (initialModel) setModel(initialModel) }, [initialModel])

  const handleRun = async () => {
    if (!file) return setError('Please select a file')
    setRunning(true); setResult(null); setError(null)

    try {
      // Upload file
      const formData = new FormData()
      formData.append('file', file)
      const uploadRes = await fetch(`${API}/upload`, { method: 'POST', body: formData })
      const uploadData = await uploadRes.json()

      // Start run
      const runRes = await fetch(`${API}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, input_path: uploadData.path, prompt: prompt || undefined }),
      })
      const runData = await runRes.json()

      // Poll for completion
      const jobId = runData.job_id
      const poll = setInterval(async () => {
        const statusRes = await fetch(`${API}/jobs/${jobId}`)
        const statusData = await statusRes.json()
        if (statusData.status === 'complete') {
          clearInterval(poll)
          setResult(statusData.result || statusData)
          setRunning(false)
        } else if (statusData.status === 'error') {
          clearInterval(poll)
          setError(statusData.error || 'Job failed')
          setRunning(false)
        }
      }, 2000)
    } catch (e) {
      setError(e.message)
      setRunning(false)
    }
  }

  const needsPrompt = ['grounding_dino'].includes(model)

  return (
    <div className="max-w-xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">Run Model</h2>
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 space-y-4">
        <div>
          <label className="block text-sm text-slate-400 mb-1">Model</label>
          <select value={model} onChange={e => setModel(e.target.value)}
            className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white">
            {models.map(m => <option key={m.name} value={m.name}>{m.name} - {m.task}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-sm text-slate-400 mb-1">Input File</label>
          <input type="file" accept="image/*,video/*" onChange={e => setFile(e.target.files[0])}
            className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white file:bg-cyan-600 file:text-white file:border-0 file:rounded file:px-3 file:py-1 file:mr-3" />
        </div>

        {needsPrompt && (
          <div>
            <label className="block text-sm text-slate-400 mb-1">Text Prompt</label>
            <input type="text" value={prompt} onChange={e => setPrompt(e.target.value)}
              placeholder="person, car, dog" className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white" />
          </div>
        )}

        <button onClick={handleRun} disabled={running}
          className={`w-full py-2 rounded font-semibold ${running ? 'bg-slate-600 text-slate-400' : 'bg-cyan-600 hover:bg-cyan-500 text-white'}`}>
          {running ? 'Running...' : 'Run'}
        </button>

        {error && <div className="bg-red-900/50 border border-red-700 rounded p-3 text-red-300 text-sm">{error}</div>}
        {result && (
          <div className="bg-green-900/30 border border-green-700 rounded p-4 space-y-2">
            <p className="text-green-400 font-semibold">Done!</p>
            {result.objects !== undefined && <p className="text-sm text-slate-300">Objects: {result.objects}</p>}
            {result.classes && <p className="text-sm text-slate-300">Classes: {[...new Set(result.classes)].join(', ')}</p>}
            {result.time_seconds && <p className="text-sm text-slate-300">Time: {result.time_seconds}s</p>}
            {result.output_dir && <p className="text-xs text-slate-500">Output: {result.output_dir}</p>}
          </div>
        )}
      </div>
    </div>
  )
}

function JobsPage() {
  const [jobs, setJobs] = useState([])
  useEffect(() => {
    const load = () => fetch(`${API}/jobs`).then(r => r.json()).then(setJobs)
    load()
    const interval = setInterval(load, 5000)
    return () => clearInterval(interval)
  }, [])

  const statusColor = { complete: 'text-green-400', running: 'text-yellow-400', error: 'text-red-400', paused: 'text-orange-400' }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Jobs</h2>
      {jobs.length === 0 ? (
        <p className="text-slate-500">No jobs yet. Run a model to see results here.</p>
      ) : (
        <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-700 text-slate-300">
              <tr>
                <th className="px-4 py-3">ID</th>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Progress</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j, i) => (
                <tr key={i} className="border-t border-slate-700">
                  <td className="px-4 py-3 font-mono text-xs">{j.job_id}</td>
                  <td className="px-4 py-3">{j.model}</td>
                  <td className={`px-4 py-3 ${statusColor[j.status] || 'text-slate-400'}`}>{j.status}</td>
                  <td className="px-4 py-3">{j.progress ? `${j.progress}%` : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function HardwarePage() {
  const [hw, setHw] = useState(null)
  const [cloud, setCloud] = useState(null)
  useEffect(() => {
    fetch(`${API}/hardware`).then(r => r.json()).then(setHw)
    fetch(`${API}/cloud-status`).then(r => r.json()).then(setCloud)
  }, [])

  if (!hw) return <p className="text-slate-500">Loading...</p>

  const items = [
    ['CPU', `${hw.cpu_name} (${hw.cpu_cores} cores)`],
    ['RAM', `${hw.ram_total_gb} GB total, ${hw.ram_available_gb} GB available`],
    ['GPU', hw.has_gpu ? `${hw.gpu_name} (${hw.gpu_vram_gb} GB VRAM)` : 'Not detected'],
    ['Disk', `${hw.disk_free_gb} GB free`],
    ['OS', hw.os_name],
  ]

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Hardware</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-cyan-400">System</h3>
          {items.map(([k, v]) => (
            <div key={k} className="flex justify-between py-2 border-b border-slate-700 last:border-0">
              <span className="text-slate-400">{k}</span>
              <span className="text-white text-sm">{v}</span>
            </div>
          ))}
          <p className="text-xs text-slate-500 mt-3">{hw.recommendation}</p>
        </div>

        <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-cyan-400">Cloud Backends</h3>
          {cloud && (
            <>
              <div className="flex justify-between py-2 border-b border-slate-700">
                <span className="text-slate-400">Kaggle</span>
                <span className={cloud.kaggle.configured ? 'text-green-400' : 'text-slate-500'}>
                  {cloud.kaggle.configured ? `Connected (${cloud.kaggle.username})` : 'Not configured'}
                </span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-slate-400">Colab</span>
                <span className={cloud.colab.configured ? 'text-green-400' : 'text-slate-500'}>
                  {cloud.colab.configured ? 'Connected' : 'Not configured'}
                </span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [page, setPage] = useState('models')
  const [runModel, setRunModel] = useState(null)

  return (
    <div className="min-h-screen">
      <Nav page={page} setPage={setPage} />
      <main className="max-w-6xl mx-auto px-6 py-8">
        {page === 'models' && <ModelsPage setPage={setPage} setRunModel={setRunModel} />}
        {page === 'run' && <RunPage initialModel={runModel} />}
        {page === 'jobs' && <JobsPage />}
        {page === 'hardware' && <HardwarePage />}
      </main>
    </div>
  )
}

export default App