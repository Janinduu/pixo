import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Layers, Play, Briefcase, Cpu, Upload, CheckCircle2, XCircle,
  Loader2, HardDrive, MonitorSmartphone, CircuitBoard,
  Cloud, FileImage, Eye, Zap, Box,
  BarChart3, Timer, Crosshair, GitBranch
} from 'lucide-react'
import { Button } from './components/ui/button'
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card'
import { Badge } from './components/ui/badge'
import { Progress } from './components/ui/progress'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from './components/ui/table'
import { Input, Select } from './components/ui/input'
import { cn } from './lib/utils'
import './index.css'

const API = '/api'

// ─── Logo ───────────────────────────────────────────────────────────

function PixoLogo({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 76 76">
      {/* Row 1 */}
      <rect x="0"  y="0"  width="22" height="22" rx="5" fill="#7F77DD"/>
      <rect x="27" y="0"  width="22" height="22" rx="5" fill="#7F77DD"/>
      <rect x="54" y="0"  width="22" height="22" rx="5" fill="#7F77DD"/>
      {/* Row 2 */}
      <rect x="0"  y="27" width="22" height="22" rx="5" fill="#7F77DD"/>
      <rect x="27" y="27" width="22" height="22" rx="5" fill="#C8C4F0" opacity="0.6"/>
      <rect x="54" y="27" width="22" height="22" rx="5" fill="#7F77DD"/>
      {/* Row 3 */}
      <rect x="0"  y="54" width="22" height="22" rx="5" fill="#534AB7"/>
      <rect x="27" y="54" width="22" height="22" rx="5" fill="#534AB7"/>
      <rect x="54" y="54" width="22" height="22" rx="5" fill="#534AB7"/>
    </svg>
  )
}

// ─── Navigation ─────────────────────────────────────────────────────

function Nav({ page, setPage }) {
  const tabs = [
    { key: 'models', label: 'Models', icon: Layers },
    { key: 'run', label: 'Run', icon: Play },
    { key: 'jobs', label: 'Jobs', icon: Briefcase },
    { key: 'hardware', label: 'Hardware', icon: Cpu },
  ]
  return (
    <nav className="sticky top-0 z-50 border-b border-zinc-200 bg-white/80 backdrop-blur-xl">
      <div className="px-8 h-14 flex items-center justify-between">
        <button onClick={() => setPage('models')}
          className="flex items-center gap-3 hover:opacity-80 transition-opacity cursor-pointer">
          <PixoLogo size={36} />
          <div className="flex flex-col items-start">
            <span className="text-lg font-bold leading-none" style={{ color: '#1a1a2e', letterSpacing: '-0.5px' }}>pixo</span>
            <span className="text-[11px] leading-none mt-1" style={{ color: '#7F77DD' }}>vision, simplified.</span>
          </div>
        </button>
        <div className="flex items-center gap-0.5">
          {tabs.map(({ key, label, icon: Icon }) => (
            <button key={key} onClick={() => setPage(key)}
              className={cn(
                "px-3 py-1.5 rounded-md text-[13px] font-medium flex items-center gap-1.5 transition-all cursor-pointer",
                page === key
                  ? "bg-zinc-100 text-zinc-900"
                  : "text-zinc-400 hover:text-zinc-600 hover:bg-zinc-50"
              )}>
              <Icon size={14} strokeWidth={1.5} />
              {label}
            </button>
          ))}
        </div>
      </div>
    </nav>
  )
}

// ─── Status Indicator ───────────────────────────────────────────────

function StatusIndicator({ status }) {
  const config = {
    complete: { color: 'bg-emerald-500', label: 'Complete' },
    running: { color: 'bg-amber-400 animate-pulse', label: 'Running' },
    error: { color: 'bg-red-500', label: 'Error' },
    paused: { color: 'bg-zinc-400', label: 'Paused' },
  }
  const { color, label } = config[status] || { color: 'bg-zinc-300', label: status }
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cn("w-1.5 h-1.5 rounded-full", color)} />
      <span className="text-zinc-600 capitalize text-[13px]">{label}</span>
    </span>
  )
}

// ─── Empty State ────────────────────────────────────────────────────

function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="w-12 h-12 rounded-xl bg-zinc-50 border border-zinc-200 flex items-center justify-center mb-4">
        <Icon size={20} strokeWidth={1.5} className="text-zinc-400" />
      </div>
      <p className="text-[14px] font-medium text-zinc-600 mb-1">{title}</p>
      <p className="text-[13px] text-zinc-400 max-w-xs">{description}</p>
      {action}
    </div>
  )
}

// ─── Models Page ────────────────────────────────────────────────────

const taskIcons = {
  'detection': Crosshair,
  'segmentation': GitBranch,
  'depth-estimation': Layers,
  'vision-language': Eye,
  'video-tracking-segmentation': Play,
}

const taskLabels = {
  'detection': 'Detection',
  'segmentation': 'Segmentation',
  'depth-estimation': 'Depth',
  'vision-language': 'Vision + Language',
  'video-tracking-segmentation': 'Video Tracking',
}

function FamilyCard({ family, onRun }) {
  const TaskIcon = taskIcons[family.task] || Box
  const hasMultipleVersions = family.versions.length > 1
  const [selectedVersion, setSelectedVersion] = useState(0)
  const [selectedVariant, setSelectedVariant] = useState('default')
  const currentModel = family.versions[selectedVersion]
  const allVariants = ['default', ...(currentModel?.variants || [])]
  const anyDownloaded = family.versions.some(v => v.downloaded)

  return (
    <Card className="group hover:border-zinc-300 hover:shadow-sm transition-all duration-200">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-zinc-50 border border-zinc-200 flex items-center justify-center">
              <TaskIcon size={14} strokeWidth={1.5} className="text-zinc-500" />
            </div>
            <div>
              <CardTitle className="text-[14px]">{family.display_name}</CardTitle>
              <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
                {taskLabels[family.task] || family.task}
              </span>
            </div>
          </div>
          {anyDownloaded && (
            <Badge variant="success" className="gap-1">
              <CheckCircle2 size={10} />
              Ready
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-[13px] text-zinc-500 leading-relaxed line-clamp-2 mb-3">
          {family.description}
        </p>

        {/* Version selector (only if multiple versions like YOLO) */}
        {hasMultipleVersions && (
          <div className="mb-3">
            <span className="text-[11px] text-zinc-400 uppercase tracking-wider block mb-1.5">Version</span>
            <div className="flex gap-1">
              {family.versions.map((v, i) => (
                <button key={v.name} onClick={() => { setSelectedVersion(i); setSelectedVariant('default') }}
                  className={cn(
                    "px-2.5 py-1 rounded text-[12px] font-medium transition-all cursor-pointer",
                    selectedVersion === i
                      ? "bg-zinc-900 text-white"
                      : "bg-zinc-100 text-zinc-500 hover:bg-zinc-200"
                  )}>
                  {v.name.replace('yolov', 'v').replace('yolo', '')}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Variant selector */}
        {allVariants.length > 1 && (
          <div className="mb-3">
            <span className="text-[11px] text-zinc-400 uppercase tracking-wider block mb-1.5">Variant</span>
            <div className="flex flex-wrap gap-1">
              {allVariants.map(v => (
                <button key={v} onClick={() => setSelectedVariant(v)}
                  className={cn(
                    "px-2 py-0.5 rounded text-[11px] font-medium transition-all cursor-pointer",
                    selectedVariant === v
                      ? "bg-zinc-800 text-white"
                      : "bg-zinc-50 border border-zinc-200 text-zinc-500 hover:bg-zinc-100"
                  )}>
                  {v}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center justify-end mt-2">
          <Button variant="secondary" size="sm" onClick={() => {
            const modelWithVariant = selectedVariant !== 'default'
              ? `${currentModel.name}:${selectedVariant}`
              : currentModel.name
            onRun(modelWithVariant)
          }}>
            <Play size={12} />
            Run
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function ModelsPage({ setPage, setRunModel }) {
  const [families, setFamilies] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/models/families`).then(r => r.json()).then(data => { setFamilies(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <LoadingState />

  const totalModels = families.reduce((acc, f) => acc + f.versions.length, 0)

  return (
    <div>
      <PageHeader
        title="Models"
        description={`${totalModels} models in ${families.length} families`}
      />
      {families.length === 0 ? (
        <EmptyState
          icon={Layers}
          title="No models found"
          description="Pull a model to get started."
          action={
            <code className="mt-3 text-[12px] bg-zinc-50 border border-zinc-200 px-3 py-1.5 rounded-lg text-zinc-600">
              pixo pull yolov8
            </code>
          }
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {families.map(f => (
            <FamilyCard key={f.family} family={f} onRun={(name) => { setRunModel(name); setPage('run') }} />
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Run Page ───────────────────────────────────────────────────────

function RunPage({ initialModel }) {
  const [model, setModel] = useState(initialModel || 'yolov8')
  const [file, setFile] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [models, setModels] = useState([])
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  useEffect(() => { fetch(`${API}/models`).then(r => r.json()).then(setModels) }, [])
  useEffect(() => { if (initialModel) setModel(initialModel) }, [initialModel])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) setFile(dropped)
  }, [])

  const handleRun = async () => {
    if (!file) return setError('Select a file first')
    setRunning(true); setResult(null); setError(null); setProgress(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      const uploadRes = await fetch(`${API}/upload`, { method: 'POST', body: formData })
      if (!uploadRes.ok) throw new Error('Upload failed')
      const uploadData = await uploadRes.json()

      const runRes = await fetch(`${API}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, input_path: uploadData.path, prompt: prompt || undefined }),
      })
      if (!runRes.ok) throw new Error('Failed to start job')
      const runData = await runRes.json()

      const jobId = runData.job_id
      const poll = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API}/jobs/${jobId}`)
          const statusData = await statusRes.json()
          if (statusData.progress) setProgress(statusData.progress)
          if (statusData.status === 'complete') {
            clearInterval(poll)
            setResult(statusData.result || statusData)
            setRunning(false)
          } else if (statusData.status === 'error') {
            clearInterval(poll)
            setError(statusData.error || 'Job failed')
            setRunning(false)
          }
        } catch {
          clearInterval(poll)
          setError('Lost connection to server')
          setRunning(false)
        }
      }, 2000)
    } catch (e) {
      setError(e.message)
      setRunning(false)
    }
  }

  const needsPrompt = ['grounding_dino', 'florence2'].includes(model)
  const selectedModel = models.find(m => m.name === model)

  return (
    <div className="max-w-2xl mx-auto">
      <PageHeader title="Run" description="Select a model, upload a file, run inference." />

      <Card>
        <CardContent className="pt-5 space-y-5">
          {/* Model selector */}
          <div>
            <Label>Model</Label>
            <Select value={model} onChange={e => setModel(e.target.value)}>
              {models.map(m => (
                <option key={m.name} value={m.name}>
                  {m.name} — {taskLabels[m.task] || m.task}
                </option>
              ))}
            </Select>
            {selectedModel && (
              <p className="mt-1.5 text-[12px] text-zinc-400">
                {taskLabels[selectedModel.task] || selectedModel.task} &middot; {selectedModel.default_size_mb} MB
                {selectedModel.downloaded && <span className="ml-2 text-emerald-500">Ready</span>}
              </p>
            )}
          </div>

          {/* File drop zone */}
          <div>
            <Label>Input</Label>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "relative border border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200",
                dragOver
                  ? "border-zinc-400 bg-zinc-50"
                  : file
                    ? "border-zinc-300 bg-zinc-50/50"
                    : "border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50/50"
              )}>
              <input ref={fileInputRef} type="file" accept="image/*,video/*"
                onChange={e => setFile(e.target.files[0])} className="hidden" />
              {file ? (
                <div className="flex items-center justify-center gap-2.5">
                  <FileImage size={18} strokeWidth={1.5} className="text-zinc-500" />
                  <span className="text-[14px] text-zinc-700 font-medium">{file.name}</span>
                  <Badge variant="outline">{(file.size / 1024 / 1024).toFixed(1)} MB</Badge>
                </div>
              ) : (
                <>
                  <div className="w-10 h-10 rounded-lg bg-zinc-100 border border-zinc-200 flex items-center justify-center mx-auto mb-3">
                    <Upload size={20} strokeWidth={1.5} className="text-zinc-400" />
                  </div>
                  <p className="text-[13px] text-zinc-500">
                    Drop your file here or <span className="text-zinc-900 font-medium">browse</span>
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Images and videos supported</p>
                </>
              )}
            </div>
          </div>

          {/* Text prompt */}
          {needsPrompt && (
            <div>
              <Label>Prompt</Label>
              <Input
                type="text" value={prompt} onChange={e => setPrompt(e.target.value)}
                placeholder="e.g. person, car, dog"
              />
            </div>
          )}

          {/* Run button */}
          <Button
            onClick={handleRun}
            disabled={running}
            variant={running ? "secondary" : "default"}
            className="w-full h-10"
          >
            {running ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Running{progress ? ` (${progress}%)` : '...'}
              </>
            ) : (
              <>
                <Zap size={14} />
                Run inference
              </>
            )}
          </Button>

          {/* Progress */}
          {running && progress != null && (
            <Progress value={progress} />
          )}

          {/* Error */}
          {error && (
            <div className="flex items-start gap-3 bg-red-50 border border-red-200 rounded-xl p-4">
              <XCircle size={16} className="text-red-500 mt-0.5 shrink-0" />
              <p className="text-[13px] text-red-600">{error}</p>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="bg-zinc-50 border border-zinc-200 rounded-xl p-5 space-y-3">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle2 size={16} className="text-emerald-500" />
                <span className="text-[14px] font-medium text-zinc-900">Complete</span>
              </div>

              <div className="grid grid-cols-2 gap-2">
                {result.objects !== undefined && (
                  <StatCard icon={BarChart3} label="Objects" value={result.objects} />
                )}
                {result.time_seconds && (
                  <StatCard icon={Timer} label="Time" value={`${result.time_seconds}s`} />
                )}
              </div>

              {result.classes && (
                <div className="pt-2">
                  <span className="text-[11px] text-zinc-400 uppercase tracking-wider block mb-2">Classes</span>
                  <div className="flex flex-wrap gap-1.5">
                    {[...new Set(result.classes)].map(c => (
                      <Badge key={c} variant="outline">{c}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {result.output_dir && (
                <p className="text-[12px] text-zinc-400 pt-3 border-t border-zinc-200">
                  Output: {result.output_dir}
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function StatCard({ icon: Icon, label, value }) {
  return (
    <div className="bg-white rounded-lg p-3 border border-zinc-200">
      <div className="flex items-center gap-1.5 mb-1">
        <Icon size={12} strokeWidth={1.5} className="text-zinc-400" />
        <span className="text-[11px] text-zinc-400 uppercase tracking-wider">{label}</span>
      </div>
      <span className="text-lg font-semibold text-zinc-900">{value}</span>
    </div>
  )
}

// ─── Jobs Page ──────────────────────────────────────────────────────

function JobsPage() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = () => fetch(`${API}/jobs`).then(r => r.json()).then(data => { setJobs(data); setLoading(false) }).catch(() => setLoading(false))
    load()
    const interval = setInterval(load, 5000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <LoadingState />

  return (
    <div>
      <PageHeader
        title="Jobs"
        description={
          <span>
            {jobs.length === 0 ? 'No jobs yet' : `${jobs.length} job${jobs.length !== 1 ? 's' : ''}`}
            <span className="ml-2 text-zinc-300">&middot; auto-refreshes</span>
          </span>
        }
      />
      {jobs.length === 0 ? (
        <EmptyState icon={Briefcase} title="No jobs yet" description="Run a model to see your jobs here." />
      ) : (
        <Card className="overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Job</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Progress</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((j, i) => (
                <TableRow key={i}>
                  <TableCell className="font-mono text-[12px] text-zinc-400">
                    {j.job_id?.substring(0, 8) || '-'}
                  </TableCell>
                  <TableCell className="font-medium text-zinc-700">{j.model}</TableCell>
                  <TableCell>
                    <StatusIndicator status={j.status} />
                  </TableCell>
                  <TableCell className="w-44">
                    {j.progress ? (
                      <div className="flex items-center gap-2.5">
                        <Progress value={j.progress} className="flex-1" />
                        <span className="text-[12px] text-zinc-400 w-8 text-right tabular-nums">{j.progress}%</span>
                      </div>
                    ) : (
                      <span className="text-zinc-400 text-[13px]">
                        {j.status === 'complete' ? '100%' : '-'}
                      </span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}
    </div>
  )
}

// ─── Hardware Page ──────────────────────────────────────────────────

function HardwareMetric({ icon: Icon, label, value, usage }) {
  return (
    <div className="flex items-start gap-3 py-3.5">
      <div className="w-8 h-8 rounded-lg bg-zinc-50 border border-zinc-200 flex items-center justify-center shrink-0">
        <Icon size={14} strokeWidth={1.5} className="text-zinc-500" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between mb-0.5">
          <span className="text-[13px] font-medium text-zinc-700">{label}</span>
          {usage !== undefined && (
            <span className="text-[11px] text-zinc-400 tabular-nums">{Math.round(usage)}%</span>
          )}
        </div>
        <p className="text-[12px] text-zinc-500 truncate">{value}</p>
        {usage !== undefined && <Progress value={usage} className="mt-2" />}
      </div>
    </div>
  )
}

function HardwarePage() {
  const [hw, setHw] = useState(null)
  const [cloud, setCloud] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = () =>
      Promise.all([
        fetch(`${API}/hardware`).then(r => r.json()),
        fetch(`${API}/cloud-status`).then(r => r.json()),
      ]).then(([hwData, cloudData]) => {
        setHw(hwData); setCloud(cloudData); setLoading(false)
      }).catch(() => setLoading(false))
    load()
    const interval = setInterval(load, 5000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <LoadingState />
  if (!hw) return <EmptyState icon={Cpu} title="Could not load hardware info" description="Make sure the pixo server is running." />

  const ramUsedPct = hw.ram_total_gb ? ((hw.ram_total_gb - hw.ram_available_gb) / hw.ram_total_gb * 100) : 0

  return (
    <div>
      <PageHeader title="Hardware" description="System resources and cloud backend status." />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-[11px] uppercase tracking-wider text-zinc-400 font-medium">System</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="divide-y divide-zinc-100">
              <HardwareMetric icon={Cpu} label="CPU" value={`${hw.cpu_name} (${hw.cpu_cores} cores)`} />
              <HardwareMetric icon={CircuitBoard} label="RAM"
                value={`${hw.ram_available_gb} GB free of ${hw.ram_total_gb} GB`}
                usage={ramUsedPct} />
              <HardwareMetric icon={MonitorSmartphone} label="GPU"
                value={hw.has_gpu ? `${hw.gpu_name} (${hw.gpu_vram_gb} GB VRAM)` : 'Not detected'} />
              <HardwareMetric icon={HardDrive} label="Disk"
                value={`${hw.disk_free_gb} GB free`} />
            </div>
            {hw.recommendation && (
              <p className="text-[11px] text-zinc-400 mt-3 pt-3 border-t border-zinc-100 leading-relaxed">
                {hw.recommendation}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-[11px] uppercase tracking-wider text-zinc-400 font-medium">Cloud Backends</CardTitle>
          </CardHeader>
          <CardContent>
            {cloud ? (
              <div className="divide-y divide-zinc-100">
                <CloudRow name="Kaggle" configured={cloud.kaggle?.configured} detail={cloud.kaggle?.username} />
                <CloudRow name="Colab" configured={cloud.colab?.configured} detail={cloud.colab?.configured ? 'Connected' : null} />
              </div>
            ) : (
              <p className="text-[13px] text-zinc-400">Loading...</p>
            )}
            <p className="text-[11px] text-zinc-400 mt-4 pt-3 border-t border-zinc-100 leading-relaxed">
              Run <code className="text-[11px] bg-zinc-50 border border-zinc-200 px-1.5 py-0.5 rounded text-zinc-600">pixo setup-cloud</code> to connect backends.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function CloudRow({ name, configured, detail }) {
  return (
    <div className="flex items-center justify-between py-3.5">
      <div className="flex items-center gap-2.5">
        <Cloud size={14} strokeWidth={1.5} className="text-zinc-400" />
        <span className="text-[13px] text-zinc-700">{name}</span>
      </div>
      {configured ? (
        <Badge variant="success" className="gap-1">
          <CheckCircle2 size={10} />
          {detail || 'Connected'}
        </Badge>
      ) : (
        <span className="text-[12px] text-zinc-400">Not configured</span>
      )}
    </div>
  )
}

// ─── Shared ─────────────────────────────────────────────────────────

function PageHeader({ title, description }) {
  return (
    <div className="mb-8">
      <h2 className="text-xl font-semibold text-zinc-900 mb-1">{title}</h2>
      <div className="text-[13px] text-zinc-400">{description}</div>
    </div>
  )
}

function Label({ children }) {
  return <label className="block text-[13px] font-medium text-zinc-600 mb-2">{children}</label>
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center py-24">
      <Loader2 size={20} strokeWidth={1.5} className="animate-spin text-zinc-300" />
    </div>
  )
}

// ─── App ────────────────────────────────────────────────────────────

function App() {
  const getPageFromHash = () => {
    const hash = window.location.hash.replace('#', '') || 'models'
    return hash
  }

  const [page, setPageState] = useState(getPageFromHash)
  const [runModel, setRunModel] = useState(null)

  const setPage = useCallback((newPage) => {
    window.location.hash = newPage
    setPageState(newPage)
  }, [])

  // Listen for browser back/forward
  useEffect(() => {
    const handleHashChange = () => setPageState(getPageFromHash())
    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  return (
    <div className="min-h-screen bg-white">
      <Nav page={page} setPage={setPage} />
      <main className="px-8 py-8">
        {page === 'models' && <ModelsPage setPage={setPage} setRunModel={setRunModel} />}
        {page === 'run' && <RunPage initialModel={runModel} />}
        {page === 'jobs' && <JobsPage />}
        {page === 'hardware' && <HardwarePage />}
      </main>
    </div>
  )
}

export default App
