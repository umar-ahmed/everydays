import { DialRoot, useDialKitController, type DialConfig } from 'dialkit'
import { useReducedMotion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { BubbleWrapApp, webgpuAvailable, type Params, type Stats } from './app.ts'
import { SHAPES, TILINGS, type Shape, type Tiling } from './generators/lattice.ts'

const dialConfig = {
  shape: { type: 'select', options: [...SHAPES], default: 'circle' },
  sheet: {
    tiling: { type: 'select', options: [...TILINGS], default: 'square' },
    'bubble size': [1.1, 0.6, 2.2, 0.05],
    spacing: [0.28, 0.1, 1, 0.01],
    'grid count': [11, 4, 20, 1],
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
    'wrinkle strength': [0.45, 0, 1.5, 0.01],
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
  studio: {
    'key strip': [3, 0, 8, 0.05],
    'rim intensity': [2.5, 0, 8, 0.05],
    'light rotation': [0, -180, 180, 1],
    background: { type: 'select', options: ['black', 'dark gray'], default: 'black' },
  },
  camera: {
    exposure: [1, 0.2, 3, 0.01],
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
    'debug view': { type: 'select', options: ['shaded', 'normals', 'wireframe', 'map'], default: 'shaded' },
  },
  reset: { type: 'action' },
} satisfies DialConfig

type Status = 'loading' | 'running' | 'unsupported'

export function BubbleWrap() {
  const reducedMotion = useReducedMotion() ?? false
  const actionRef = useRef<(path: string) => void>(() => {})
  const dial = useDialKitController('day 002 · bubble wrap', dialConfig, {
    onAction: (path) => actionRef.current(path),
  })
  const v = dial.values
  const params: Params = {
    shape: v.shape as Shape,
    tiling: v.sheet.tiling as Tiling,
    size: v.sheet['bubble size'],
    spacing: v.sheet.spacing,
    cols: v.sheet['grid count'],
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
    keyStrip: v.studio['key strip'],
    rimLight: v.studio['rim intensity'],
    lightRotation: v.studio['light rotation'],
    background: v.studio.background as Params['background'],
    exposure: v.camera.exposure,
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
  const [stats, setStats] = useState<Stats>({ popped: 0, total: 0 })

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
        setStats((prev) => (prev.popped === s.popped && prev.total === s.total ? prev : s))
      app = new BubbleWrapApp(container.current, paramsRef.current, onStats)
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
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'r' && !(e.target instanceof HTMLInputElement)) appRef.current?.reset()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <>
      <main className="fixed inset-0 overflow-hidden bg-black select-none">
        <div ref={container} className="absolute inset-0" style={{ cursor: status === 'running' ? 'none' : 'default' }} />

        {status === 'loading' && (
          <p className="absolute inset-0 grid place-items-center font-mono text-xs text-white/40">warming up the GPU…</p>
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

        <div data-chrome className="pointer-events-none absolute bottom-4 left-4 max-w-[65vw] font-mono text-[11px] tracking-wide text-white/55 sm:bottom-5 sm:left-5">
          <a href={import.meta.env.BASE_URL} className="pointer-events-auto text-white/35 transition-colors hover:text-white/80">
            everydays
          </a>
          <span className="text-white/25"> / </span>
          day 002 / bubble wrap
        </div>

        {status === 'running' && (
          <div
            data-chrome
            className="pointer-events-none absolute right-4 bottom-4 flex items-center gap-3 sm:right-auto sm:bottom-5 sm:left-1/2 sm:-translate-x-1/2"
          >
            <span className="hidden font-mono text-[11px] text-white/30 sm:inline">
              press &amp; drag to pop · {stats.popped}/{stats.total}
            </span>
            <button
              type="button"
              onClick={() => appRef.current?.reset()}
              className="pointer-events-auto rounded-full border border-white/15 bg-white/5 px-3 py-1.5 font-mono text-[11px] text-white/70 backdrop-blur transition-colors hover:border-white/30 hover:text-white focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:outline-none"
            >
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
