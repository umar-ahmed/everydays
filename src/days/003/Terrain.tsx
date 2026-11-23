import { DialRoot, useDialKitController, type DialConfig } from 'dialkit'
import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { TerrainApp, compactDevice, webgpuAvailable, type Params, type Stats } from './app.ts'
import { PALETTES, PALETTE_NAMES, type PaletteName } from './palettes.ts'
import { RESOLUTIONS, defaultResolution, maxResolution } from './sim.ts'

const compact = compactDevice()

const dialConfig = {
  erode: true,
  terrain: {
    seed: [1, 1, 999, 1],
    octaves: [7, 1, 10, 1],
    roughness: [0.5, 0.25, 0.75, 0.01],
    'ridge mix': [0.55, 0, 1, 0.01],
    'height scale': [2.4, 0.5, 4, 0.05],
    // Cells per side; the same mountains, finer or coarser. Phones stop at 512.
    resolution: {
      type: 'select',
      options: RESOLUTIONS.filter((r) => r <= maxResolution(compact)).map(String),
      default: String(defaultResolution(compact)),
    },
  },
  look: {
    palette: { type: 'select', options: [...PALETTE_NAMES], default: 'alpine' },
  },
  snow: {
    // Fraction of the peak height; picking a palette resets it to that palette's line.
    'snow line': [PALETTES.alpine.snowLine, 0, 1.5, 0.01],
    softness: [0.06, 0, 0.3, 0.005],
    'slope limit': [55, 10, 75, 1],
    'aspect bias': [0.35, 0, 1, 0.01],
    sparkle: [0.5, 0, 1, 0.01],
  },
  erosion: {
    'rain amount': [0.012, 0, 0.05, 0.001],
    capacity: [1, 0.1, 4, 0.05],
    'erosion rate': [0.05, 0, 0.3, 0.005],
    'deposition rate': [0.05, 0, 0.3, 0.005],
    evaporation: [0.015, 0, 0.1, 0.001],
    'thermal weathering': [0.2, 0, 1, 0.01],
    'iterations per frame': [12, 1, 60, 1],
  },
  view: {
    'debug view': { type: 'select', options: ['shaded', 'water', 'erosion'], default: 'shaded' },
  },
  reset: { type: 'action' },
  'new seed': { type: 'action' },
} satisfies DialConfig

type Status = 'loading' | 'running' | 'unsupported'

export function Terrain() {
  const actionRef = useRef<(path: string) => void>(() => {})
  const dial = useDialKitController('day 003 · terrain', dialConfig, {
    onAction: (path) => actionRef.current(path),
  })
  const v = dial.values
  const params: Params = {
    seed: v.terrain.seed,
    octaves: v.terrain.octaves,
    roughness: v.terrain.roughness,
    ridge: v.terrain['ridge mix'],
    heightScale: v.terrain['height scale'],
    rain: v.erosion['rain amount'],
    capacity: v.erosion.capacity,
    erosion: v.erosion['erosion rate'],
    deposition: v.erosion['deposition rate'],
    evaporation: v.erosion.evaporation,
    thermal: v.erosion['thermal weathering'],
    iterations: v.erosion['iterations per frame'],
    eroding: v.erode,
    view: v.view['debug view'] as Params['view'],
    resolution: Number(v.terrain.resolution),
    palette: v.look.palette as PaletteName,
    snowLine: v.snow['snow line'],
    snowSoftness: v.snow.softness,
    snowSlope: v.snow['slope limit'],
    snowAspect: v.snow['aspect bias'],
    sparkle: v.snow.sparkle,
  }
  // A new palette brings its own snow line (Alpine snow-capped, Desert bare, Arctic white).
  const palette = useRef(params.palette)
  useEffect(() => {
    if (palette.current === params.palette) return
    palette.current = params.palette
    dial.setValue('snow.snow line', PALETTES[params.palette].snowLine)
  }, [params.palette, dial])
  const paramsRef = useRef(params)
  const container = useRef<HTMLDivElement>(null)
  const appRef = useRef<TerrainApp | null>(null)
  const [status, setStatus] = useState<Status>('loading')

  useEffect(() => {
    let app: TerrainApp | null = null
    let cancelled = false
    void (async () => {
      if (!(await webgpuAvailable())) {
        if (!cancelled) setStatus('unsupported')
        return
      }
      if (cancelled || !container.current) return
      app = new TerrainApp(container.current, paramsRef.current, stepStore.set)
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
      // `?debug` exposes the app for inspection and the OG script.
      if (import.meta.env.DEV || new URLSearchParams(location.search).has('debug')) {
        ;(window as unknown as { terrain: TerrainApp }).terrain = app
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
      if (path === 'new seed') dial.setValue('terrain.seed', 1 + Math.floor(Math.random() * 998))
    }
  })

  const pill =
    'pointer-events-auto whitespace-nowrap rounded-full border border-white/20 bg-black/25 px-3 py-2 font-mono text-[11px] text-white/80 backdrop-blur transition-colors hover:border-white/40 hover:text-white focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:outline-none sm:py-1.5'

  return (
    <>
      <main className="fixed inset-0 overflow-hidden bg-[#5b86c9] select-none">
        <div ref={container} className="absolute inset-0" />
        {status === 'loading' && (
          <p className="absolute inset-0 grid place-items-center font-mono text-xs text-white/70">raising mountains…</p>
        )}
        {status === 'unsupported' && (
          <div className="absolute inset-0 grid place-items-center bg-black p-6">
            <div className="max-w-md rounded-2xl border border-white/10 bg-white/[0.03] p-6 text-center">
              <img
                src={`${import.meta.env.BASE_URL}003/og.png`}
                alt="An eroded mountain range with river channels and sediment fans"
                className="mb-5 w-full rounded-lg opacity-80"
              />
              <h1 className="text-lg font-medium text-white/90">This toy needs WebGPU</h1>
              <p className="mt-2 text-sm text-white/55">
                The terrain is generated and eroded with compute shaders on your GPU. Try a recent Chrome, Edge or
                Safari (Safari 26+).
              </p>
            </div>
          </div>
        )}
        <div
          data-chrome
          className="pointer-events-none absolute top-4 left-4 max-w-[65vw] font-mono text-[11px] tracking-wide text-white/80 lg:top-auto lg:bottom-5 lg:left-5"
        >
          <a
            href={import.meta.env.BASE_URL}
            className="pointer-events-auto text-white/60 transition-colors hover:text-white"
          >
            everydays
          </a>
          <span className="text-white/40"> / </span>
          day 003 / terrain
        </div>
        {status === 'running' && (
          <div
            data-chrome
            className="pointer-events-none absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 sm:bottom-5 sm:gap-3"
          >
            <span className="font-mono text-[11px] whitespace-nowrap text-white/75">
              <span className="hidden md:inline">drag to orbit · </span>
              <StepCounter /> steps
            </span>
            <button type="button" onClick={() => dial.setValue('erode', !v.erode)} className={pill}>
              {v.erode ? 'pause' : 'erode'}
            </button>
            <button type="button" onClick={() => appRef.current?.reset()} className={pill}>
              reset
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

/**
 * Step counts arrive several times a second; a small store lets only the counter re-render for
 * them, not the whole toy and its dial panel.
 */
const stepStore = (() => {
  let steps = 0
  const listeners = new Set<() => void>()
  return {
    set: (s: Stats) => {
      steps = s.steps
      for (const l of listeners) l()
    },
    get: () => steps,
    subscribe: (l: () => void) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
  }
})()

function StepCounter() {
  const steps = useSyncExternalStore(stepStore.subscribe, stepStore.get)
  return <>{steps.toLocaleString()}</>
}
