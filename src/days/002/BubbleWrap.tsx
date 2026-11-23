import { DialRoot, useDialKitController, type DialConfig } from 'dialkit'
import { useReducedMotion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { BubbleWrapApp, webgpuAvailable, type DebugInfo, type Params, type Stats } from './app.ts'
import { CORRECTIONS, DIAG_VIEWS } from './materials.ts'
import { SHAPES, TILINGS, type Shape, type Tiling } from './generators/lattice.ts'

const MODES = ['press', 'view'] as const
type Mode = (typeof MODES)[number]

const dialConfig = {
  mode: { type: 'select', options: [...MODES], default: 'press' },
  shape: { type: 'select', options: [...SHAPES], default: 'circle' },
  sheet: {
    tiling: { type: 'select', options: [...TILINGS], default: 'square' },
    'bubble size': [1.1, 0.6, 2.2, 0.05],
    spacing: [0.28, 0.1, 1, 0.01],
    // Bubbles along the shorter side; the longer side fills the screen.
    'bubbles across': [8, 3, 16, 1],
    // Extra flat seal film between the outermost bubbles and the sheet edge (cm).
    'edge margin': [0.4, 0, 2, 0.05],
    'pinhole chance': [0.06, 0, 0.5, 0.01],
  },
  physics: {
    'fill pressure': [0.12, 0, 0.5, 0.01],
    'burst threshold': [0.6, 0.2, 3, 0.05],
    'film stiffness': [0.75, 0, 1, 0.01],
    substeps: [8, 2, 24, 1],
  },
  press: {
    tool: { type: 'select', options: ['finger', 'roller'], default: 'finger' },
    radius: [0.75, 0.2, 2, 0.05],
    strength: [0.9, 0, 1, 0.01],
  },
  look: {
    'film thickness': [420, 100, 1000, 10],
    iridescence: [0.35, 0, 1, 0.01],
    haze: [0.08, 0, 0.6, 0.01],
    'wrinkle strength': [0.25, 0, 1.5, 0.01],
    'wrinkle scale': [1, 0.3, 3, 0.05],
    'crease sharpness': [0.4, 0, 1, 0.01],
    'render detail': { type: 'select', options: ['1x', '2x', '4x'], default: '2x' },
    texture: { type: 'select', options: ['none', 'grid', 'dots'], default: 'none' },
    'texture scale': [1, 0.4, 3, 0.05],
    'texture depth': [0.8, 0, 2, 0.01],
    'texture sharpness': [0.5, 0, 1, 0.01],
    tint: { type: 'select', options: ['clear', 'pink', 'blue', 'custom'], default: 'clear' },
    'tint color': { type: 'color', default: '#ff9ccc' },
    density: [1, 0, 4, 0.05],
  },
  crumple: {
    amount: [0.45, 0, 2, 0.01],
    scale: [1, 0.3, 3, 0.05],
    'crease sharpness': [0.7, 0, 1, 0.01],
    'film vs dome': [0.65, 0, 1, 0.01],
    seed: [0, 0, 100, 1],
  },
  studio: {
    'key strip': [3, 0, 8, 0.05],
    'rim intensity': [2.5, 0, 8, 0.05],
    'light rotation': [0, -180, 180, 1],
    background: { type: 'select', options: ['black', 'dark gray'], default: 'black' },
  },
  camera: {
    exposure: [1, 0.2, 3, 0.01],
    elevation: [56, 12, 85, 1],
    'dof amount': [0.35, 0, 1, 0.01],
    'focus follows pointer': true,
    bloom: [0.3, 0, 2, 0.01],
    grain: [0.04, 0, 0.3, 0.005],
    vignette: [0.3, 0, 1, 0.01],
  },
  sound: {
    'pop volume': [0.7, 0, 1, 0.01],
    'pop pitch': [1, 0.5, 2, 0.01],
  },
  debug: {
    _collapsed: true,
    simulation: { type: 'select', options: ['running', 'paused', 'rest pose'], default: 'running' },
    'debug view': {
      type: 'select',
      options: ['shaded', 'normals', 'wireframe', 'map', ...DIAG_VIEWS],
      default: 'shaded',
    },
    'correction type': { type: 'select', options: [...CORRECTIONS], default: 'total' },
    'map range': { type: 'select', options: ['auto', 'fixed'], default: 'auto' },
    'show hud': false,
    'sim speed': [1, 0.1, 1, 0.05],
    'freeze on first pop': false,
    'step frame': { type: 'action' },
    'step substep': { type: 'action' },
  },
  reset: { type: 'action' },
  'reset view': { type: 'action' },
} satisfies DialConfig

type Status = 'loading' | 'running' | 'unsupported'

export function BubbleWrap() {
  const reducedMotion = useReducedMotion() ?? false
  const actionRef = useRef<(path: string) => void>(() => {})
  const dial = useDialKitController('day 002 · bubble wrap', dialConfig, {
    onAction: (path) => actionRef.current(path),
  })
  const v = dial.values
  // Holding Space is a temporary View mode on desktop.
  const [held, setHeld] = useState(false)
  const mode: Mode = held ? 'view' : (v.mode as Mode)
  const params: Params = {
    mode,
    shape: v.shape as Shape,
    tiling: v.sheet.tiling as Tiling,
    size: v.sheet['bubble size'],
    spacing: v.sheet.spacing,
    across: v.sheet['bubbles across'],
    margin: v.sheet['edge margin'],
    pinholeChance: v.sheet['pinhole chance'],
    fill: v.physics['fill pressure'],
    threshold: v.physics['burst threshold'],
    stiffness: v.physics['film stiffness'],
    substeps: v.physics.substeps,
    tool: v.press.tool as 'finger' | 'roller',
    pressRadius: v.press.radius,
    pressStrength: v.press.strength,
    thickness: v.look['film thickness'],
    iridescence: v.look.iridescence,
    haze: v.look.haze,
    wrinkle: v.look['wrinkle strength'],
    wrinkleScale: v.look['wrinkle scale'],
    sharpness: v.look['crease sharpness'],
    crumple: v.crumple.amount,
    crumpleScale: v.crumple.scale,
    crumpleSharpness: v.crumple['crease sharpness'],
    crumpleBalance: v.crumple['film vs dome'],
    crumpleSeed: v.crumple.seed,
    keyStrip: v.studio['key strip'],
    rimLight: v.studio['rim intensity'],
    lightRotation: v.studio['light rotation'],
    background: v.studio.background as Params['background'],
    exposure: v.camera.exposure,
    elevation: v.camera.elevation,
    dof: v.camera['dof amount'],
    focusFollows: v.camera['focus follows pointer'],
    bloom: v.camera.bloom,
    grain: v.camera.grain,
    vignette: v.camera.vignette,
    volume: v.sound['pop volume'],
    pitch: v.sound['pop pitch'],
    reducedMotion,
    simulation: v.debug.simulation as Params['simulation'],
    view: v.debug['debug view'] as Params['view'],
    corrKind: v.debug['correction type'] as Params['corrKind'],
    mapRange: v.debug['map range'] as Params['mapRange'],
    hud: v.debug['show hud'],
    simSpeed: v.debug['sim speed'],
    freezeOnPop: v.debug['freeze on first pop'],
    detail: parseInt(v.look['render detail']) as Params['detail'],
    texture: v.look.texture as Params['texture'],
    textureScale: v.look['texture scale'],
    textureDepth: v.look['texture depth'],
    textureSharpness: v.look['texture sharpness'],
    tint: v.look.tint as Params['tint'],
    tintColor: v.look['tint color'],
    density: v.look.density,
  }
  const paramsRef = useRef(params)

  const container = useRef<HTMLDivElement>(null)
  const appRef = useRef<BubbleWrapApp | null>(null)
  const [status, setStatus] = useState<Status>('loading')
  const [stats, setStats] = useState<Stats>({ popped: 0, total: 0, viewMoved: false })
  const [debug, setDebug] = useState<DebugInfo | null>(null)
  const freezeRef = useRef<() => void>(() => {})

  useEffect(() => {
    let app: BubbleWrapApp | null = null
    let cancelled = false
    void (async () => {
      if (!(await webgpuAvailable())) {
        if (!cancelled) setStatus('unsupported')
        return
      }
      if (cancelled || !container.current) return
      // Skip no-op updates: a sheet rebuild reports 0/N on every dial tick, and re-rendering for
      // an unchanged count re-runs the params effect inside DialKit's update.
      const onStats = (s: Stats) =>
        setStats((prev) =>
          prev.popped === s.popped && prev.total === s.total && prev.viewMoved === s.viewMoved ? prev : s,
        )
      app = new BubbleWrapApp(container.current, paramsRef.current, onStats, {
        onDebug: setDebug,
        onFreeze: () => freezeRef.current(),
      })
      try {
        await app.init()
      } catch {
        app.dispose()
        app = null
        if (!cancelled) setStatus('unsupported')
        return
      }
      if (cancelled) return app.dispose()
      appRef.current = app
      // `?debug` exposes the app for inspection and the headless checks.
      if (import.meta.env.DEV || new URLSearchParams(location.search).has('debug')) {
        ;(window as unknown as { bubblewrap: BubbleWrapApp }).bubblewrap = app
      }
      setStatus('running')
    })()
    return () => {
      cancelled = true
      app?.dispose()
      appRef.current = null
    }
  }, [])

  useEffect(() => {
    paramsRef.current = params
    appRef.current?.setParams(params)
  })

  useEffect(() => {
    actionRef.current = (path) => {
      if (path === 'reset') appRef.current?.reset()
      if (path === 'reset view') appRef.current?.resetView()
      if (path === 'debug.step frame') appRef.current?.stepOnce('frame')
      if (path === 'debug.step substep') appRef.current?.stepOnce('substep')
    }
    freezeRef.current = () => {
      dial.setValue('debug.simulation', 'paused')
    }
    const typing = (e: KeyboardEvent) =>
      e.target instanceof HTMLInputElement ||
      e.target instanceof HTMLTextAreaElement ||
      e.target instanceof HTMLSelectElement
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'r') appRef.current?.reset()
      if (e.key === 'v' && !e.repeat) dial.setValue('mode', v.mode === 'view' ? 'press' : 'view')
      if (e.code === 'Space') {
        // Also keeps a focused button from being clicked by the held key.
        e.preventDefault()
        setHeld(true)
      }
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') setHeld(false)
    }
    const onBlur = () => setHeld(false)
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onBlur)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
    }
  })

  const pill =
    'pointer-events-auto whitespace-nowrap rounded-full border border-white/15 bg-white/5 px-3 py-2 font-mono text-[11px] text-white/70 backdrop-blur transition-colors hover:border-white/30 hover:text-white focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:outline-none sm:py-1.5'

  return (
    <>
      <main className="fixed inset-0 overflow-hidden bg-black select-none">
        <div
          ref={container}
          className="absolute inset-0"
          style={{ cursor: status === 'running' ? 'none' : 'default' }}
        />

        {status === 'loading' && (
          <p className="absolute inset-0 grid place-items-center font-mono text-xs text-white/40">
            warming up the GPU…
          </p>
        )}
        {status === 'unsupported' && (
          <div className="absolute inset-0 grid place-items-center p-6">
            <div className="max-w-md rounded-2xl border border-white/10 bg-white/[0.03] p-6 text-center">
              <img
                src={`${import.meta.env.BASE_URL}002/og.png`}
                alt="Bubble wrap with iridescent domes, some popped"
                className="mb-5 w-full rounded-lg opacity-80"
              />
              <h1 className="text-lg font-medium text-white/90">This toy needs WebGPU</h1>
              <p className="mt-2 text-sm text-white/55">
                The bubble wrap is simulated with compute shaders on your GPU. Try a recent Chrome, Edge or Safari
                (Safari 26+), or enable WebGPU in your browser&apos;s flags.
              </p>
            </div>
          </div>
        )}

        {status === 'running' && debug && (params.hud || params.view !== 'shaded') && (
          <DebugOverlay debug={debug} showLegend={!!debug.legend && params.view === debug.legend.view} />
        )}

        <div
          data-chrome
          className="pointer-events-none absolute top-4 left-4 max-w-[65vw] font-mono text-[11px] tracking-wide text-white/55 lg:top-auto lg:bottom-5 lg:left-5"
        >
          <a
            href={import.meta.env.BASE_URL}
            className="pointer-events-auto text-white/35 transition-colors hover:text-white/80"
          >
            everydays
          </a>
          <span className="text-white/25"> / </span>
          day 002 / bubble wrap
        </div>

        {status === 'running' && (
          <div
            data-chrome
            className="pointer-events-none absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 sm:bottom-5 sm:gap-3"
          >
            <span className="font-mono text-[11px] whitespace-nowrap text-white/30">
              <span className="hidden md:inline">
                {mode === 'view' ? 'drag to pan · scroll to zoom' : 'press & drag to pop'} ·{' '}
              </span>
              {stats.popped}/{stats.total}
            </span>
            <div
              role="radiogroup"
              aria-label="Mode"
              className="pointer-events-auto flex rounded-full border border-white/15 bg-white/5 p-0.5 backdrop-blur"
            >
              {MODES.map((m) => (
                <button
                  key={m}
                  type="button"
                  role="radio"
                  aria-checked={mode === m}
                  onClick={() => dial.setValue('mode', m)}
                  className={`rounded-full px-3 py-1.5 font-mono text-[11px] transition-colors focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:outline-none sm:py-1 ${
                    mode === m ? 'bg-white/15 text-white' : 'text-white/50 hover:text-white/80'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
            {stats.viewMoved && (
              <button type="button" onClick={() => appRef.current?.resetView()} className={pill}>
                reset view
              </button>
            )}
            <button type="button" onClick={() => appRef.current?.reset()} className={pill}>
              new sheet
            </button>
          </div>
        )}
      </main>
      <DialRoot
        position="top-right"
        theme="dark"
        defaultOpen={typeof window === 'undefined' || window.innerWidth >= 640}
        productionEnabled
      />
    </>
  )
}

/** HUD (top left, under the title) and the colour legend of the active debug map. */
function DebugOverlay({ debug, showLegend }: { debug: DebugInfo; showLegend: boolean }) {
  const { history } = debug
  const peak = Math.max(...history, 1e-12)
  const points = history
    .map((e, i) => `${(i / Math.max(history.length - 1, 1)) * 120},${28 - (e / peak) * 26}`)
    .join(' ')
  const legend = debug.legend
  const fmt = (x: number) => (Math.abs(x) >= 100 || (Math.abs(x) < 0.01 && x !== 0) ? x.toExponential(1) : x.toFixed(3))
  const ramp = legend?.signed
    ? 'linear-gradient(90deg, rgb(31,89,255), rgb(235,235,235), rgb(255,56,31))'
    : 'linear-gradient(90deg, rgb(13,8,41), rgb(140,26,115), rgb(250,115,26), rgb(255,242,153))'
  return (
    <div
      data-chrome
      className="pointer-events-none absolute top-12 left-4 w-[200px] rounded-lg border border-white/10 bg-black/60 p-2.5 font-mono text-[10px] leading-4 text-white/75 backdrop-blur lg:top-4"
    >
      <div className="flex justify-between">
        <span className="text-white/45">max |v|</span>
        <span>{debug.maxSpeed.toFixed(2)} cm/s</span>
      </div>
      <div className="flex justify-between">
        <span className="text-white/45">max strain</span>
        <span>{(debug.maxStrain * 100).toFixed(2)} %</span>
      </div>
      <div className="flex justify-between">
        <span className="text-white/45">kinetic energy</span>
        <span>{debug.energy.toExponential(2)}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-white/45">pops / s</span>
        <span>{debug.popsPerSecond}</span>
      </div>
      <svg viewBox="0 0 120 30" className="mt-1 h-[30px] w-full" aria-label="kinetic energy, last few seconds">
        <polyline points={points} fill="none" stroke="rgb(250,170,60)" strokeWidth="1.2" />
      </svg>
      <div className="text-white/35">energy, last ~6 s (peak {peak.toExponential(1)})</div>
      {debug.frozen && <div className="mt-1 text-[rgb(250,170,60)]">frozen on first pop</div>}
      {showLegend && legend && (
        <div className="mt-2 border-t border-white/10 pt-2">
          <div className="text-white/90">{legend.view}</div>
          <div className="mt-1 h-2 rounded-sm" style={{ background: ramp }} />
          <div className="mt-0.5 flex justify-between">
            <span>{fmt(legend.lo)}</span>
            {legend.signed && <span>0</span>}
            <span>{fmt(legend.hi)}</span>
          </div>
          <div className="text-white/45">{legend.units}</div>
          {legend.extra && <div className="text-white/45">{legend.extra}</div>}
        </div>
      )}
    </div>
  )
}
