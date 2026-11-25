import { DialRoot, useDialKitController, type DialConfig } from 'dialkit'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState, useSyncExternalStore, type PointerEvent as ReactPointerEvent } from 'react'
import { CpuTracer } from './cpu.ts'
import { GpuTracer, MAX_BOUNCES, type Tracer } from './gpu.ts'
import {
  LENS_OFFSET,
  MAX_SHAPES,
  TINTS,
  TINT_ORDER,
  defaultScene,
  makeShape,
  polygon,
  refit,
  viewFor,
  type Material,
  type Optics,
  type Scene,
  type Shape,
  type ShapeKind,
} from './optics.ts'

const compact = typeof screen !== 'undefined' && Math.min(screen.width, screen.height) < 600
/** Paths the GPU buffers hold per frame; the samples dial picks how many of them run. */
const MAX_PATHS = compact ? 32 * 1024 : 64 * 1024
const initialAspect = typeof window === 'undefined' ? 16 / 9 : window.innerWidth / Math.max(1, window.innerHeight)
const initial = defaultScene(initialAspect)

const dialConfig = {
  flashlight: {
    power: [1, 0.05, 4, 0.01],
    // Full spread of the beam, degrees. A narrow beam shows the cleanest rainbow.
    cone: [2, 0, 120, 0.5],
    rotation: [initial.rotation, -180, 180, 1],
    // Width of the lens the light leaves from, world units (the short side of the view is 2).
    lens: [0.04, 0, 0.4, 0.005],
  },
  glass: {
    // Cauchy A: the index in the deep red.
    'base IOR': [1.5, 1.1, 2.4, 0.01],
    // Cauchy B (µm²): how much harder violet bends than red. Crown glass ~0.004, dense flint ~0.014.
    dispersion: [0.02, 0, 0.08, 0.001],
  },
  light: {
    bounces: [8, 1, MAX_BOUNCES, 1],
    exposure: [0.85, 0.1, 8, 0.05],
    // Thousands of light paths traced per frame.
    'samples per frame': [compact ? 8 : 24, 1, MAX_PATHS / 1024, 1],
    // Scattering in the air: beams fade and glow as they go.
    haze: [0.05, 0, 1.5, 0.01],
    // The window's edges are the room's walls: grey Lambertian reflectors bouncing back this
    // fraction of the light, whatever its colour.
    walls: [0.7, 0, 0.95, 0.01],
    // Share of light matte shapes bounce back, before their tint's spectral curve.
    'diffuse albedo': [0.8, 0, 0.95, 0.01],
  },
  'reset scene': { type: 'action' },
} satisfies DialConfig

type Target = { type: 'shape'; id: number } | { type: 'flash' }
type Grab = { target: Target; mode: 'move' | 'rotate'; dx: number; dy: number }
type Gesture = {
  target: Target
  angle: number
  dist: number
  rot: number
  r: number
  ox: number
  oy: number
}

const wrapDeg = (d: number) => ((((d + 180) % 360) + 360) % 360) - 180
const ROTATE_ARM = 0.14
const FLASH_ARM = 0.32

export function Light() {
  const actionRef = useRef<(path: string) => void>(() => {})
  const dial = useDialKitController('day 004 · spectral pathtracer', dialConfig, {
    onAction: (path) => actionRef.current(path),
  })
  const v = dial.values
  const optics: Optics = {
    power: v.flashlight.power,
    cone: v.flashlight.cone,
    rotation: v.flashlight.rotation,
    aperture: v.flashlight.lens,
    bounces: v.light.bounces,
    ior: v.glass['base IOR'],
    dispersion: v.glass.dispersion,
    haze: v.light.haze,
    walls: v.light.walls,
    albedo: v.light['diffuse albedo'],
  }
  const samples = v.light['samples per frame'] * 1024
  const exposure = v.light.exposure

  const [scene, setScene] = useState<Scene>(initial.scene)
  const [selected, setSelected] = useState<number | null>(null)
  const [newMaterial, setNewMaterial] = useState<Material>('glass')
  const [mode, setMode] = useState<'loading' | 'gpu' | 'cpu'>('loading')
  const [size, setSize] = useState({ width: 1, height: 1 })
  const container = useRef<HTMLDivElement>(null)
  const svg = useRef<SVGSVGElement>(null)
  const tracerRef = useRef<Tracer | null>(null)
  const latest = useRef({ scene, optics, samples, exposure })
  useEffect(() => {
    latest.current = { scene, optics, samples, exposure }
  })

  const view = viewFor(size.width / Math.max(1, size.height))

  useEffect(() => {
    const el = container.current!
    const ro = new ResizeObserver(() => setSize({ width: el.clientWidth, height: el.clientHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // The walls are the window's edges: when it changes shape, bring the scene along.
  const lastView = useRef<typeof view | null>(null)
  useEffect(() => {
    if (size.width <= 1 || size.height <= 1) return
    const to = viewFor(size.width / size.height)
    const from = lastView.current ?? viewFor(initialAspect)
    lastView.current = to
    if (from.hw === to.hw && from.hh === to.hh) return
    const { scene, optics } = latest.current
    const next = refit(scene, optics.rotation, from, to)
    setScene(next.scene)
    if (next.rotation !== optics.rotation) dial.setValue('flashlight.rotation', next.rotation)
  }, [size.width, size.height, dial])

  useEffect(() => {
    let tracer: Tracer | null = null
    let cancelled = false
    const start = (t: Tracer, m: 'gpu' | 'cpu') => {
      tracer = t
      tracerRef.current = t
      const { scene, optics, samples, exposure } = latest.current
      t.setScene(scene, optics, samples)
      t.setDisplay({ exposure })
      setMode(m)
      // `?debug` exposes the tracer for inspection and the OG script.
      if (import.meta.env.DEV || new URLSearchParams(location.search).has('debug')) {
        ;(window as unknown as { light: Tracer }).light = t
      }
    }
    void (async () => {
      const el = container.current!
      const adapter =
        'gpu' in navigator && navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null
      if (cancelled) return
      if (adapter) {
        const gpu = new GpuTracer(el, MAX_PATHS, pathStore.set)
        try {
          await gpu.init()
          if (cancelled) return gpu.dispose()
          return start(gpu, 'gpu')
        } catch {
          gpu.dispose()
        }
      }
      if (!cancelled) start(new CpuTracer(el, pathStore.set), 'cpu')
    })()
    return () => {
      cancelled = true
      tracer?.dispose()
      tracerRef.current = null
    }
  }, [])

  const opticsKey = JSON.stringify(optics)
  useEffect(() => {
    const { scene, optics, samples } = latest.current
    tracerRef.current?.setScene(scene, optics, samples)
  }, [scene, opticsKey, samples, mode])
  useEffect(() => {
    tracerRef.current?.setDisplay({ exposure })
  }, [exposure, mode])

  const updateShape = (id: number, patch: Partial<Shape>) =>
    setScene((s) => ({ ...s, shapes: s.shapes.map((sh) => (sh.id === id ? { ...sh, ...patch } : sh)) }))
  const removeShape = (id: number) => {
    setScene((s) => ({ ...s, shapes: s.shapes.filter((sh) => sh.id !== id) }))
    setSelected(null)
  }
  const addShape = (kind: ShapeKind) => {
    if (scene.shapes.length >= MAX_SHAPES) return
    const jitter = () => (Math.random() - 0.5) * 0.5
    const shape = makeShape({
      kind,
      material: newMaterial,
      x: view.hw * 0.25 + jitter(),
      y: jitter(),
      r: 0.22,
      angle: Math.random() * Math.PI,
    })
    setScene((s) => ({ ...s, shapes: [...s.shapes, shape] }))
    setSelected(shape.id)
  }

  useEffect(() => {
    actionRef.current = (path) => {
      if (path !== 'reset scene') return
      const fresh = defaultScene(size.width / Math.max(1, size.height))
      setScene(fresh.scene)
      setSelected(null)
      dial.setValue('flashlight.rotation', fresh.rotation)
    }
  })

  // Pointer interaction: one pointer drags or turns, two pointers on a grabbed object turn it
  // and (for shapes) resize it.
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const grab = useRef<Grab | null>(null)
  const gesture = useRef<Gesture | null>(null)

  const toWorld = (clientX: number, clientY: number) => {
    const rect = svg.current!.getBoundingClientRect()
    return {
      x: ((clientX - rect.left) / rect.width) * 2 * view.hw - view.hw,
      y: ((clientY - rect.top) / rect.height) * 2 * view.hh - view.hh,
    }
  }
  const targetOf = (el: EventTarget | null): { target: Target; mode: Grab['mode'] } | null => {
    const node = (el as Element | null)?.closest?.('[data-target]')
    const tag = node?.getAttribute('data-target')
    if (!tag) return null
    const [what, id, m] = tag.split(':')
    const mode = m === 'rotate' ? 'rotate' : 'move'
    return what === 'flash' ? { target: { type: 'flash' }, mode } : { target: { type: 'shape', id: Number(id) }, mode }
  }
  const positionOf = (t: Target) => {
    const s = latest.current.scene
    if (t.type === 'flash') return s.flashlight
    return s.shapes.find((sh) => sh.id === t.id) ?? null
  }
  const moveTarget = (t: Target, x: number, y: number) => {
    const cx = Math.max(-view.hw, Math.min(view.hw, x))
    const cy = Math.max(-view.hh, Math.min(view.hh, y))
    if (t.type === 'flash') setScene((s) => ({ ...s, flashlight: { x: cx, y: cy } }))
    else updateShape(t.id, { x: cx, y: cy })
  }
  const rotateTarget = (t: Target, radians: number) => {
    if (t.type === 'flash') dial.setValue('flashlight.rotation', Math.round(wrapDeg((radians * 180) / Math.PI)))
    else updateShape(t.id, { angle: radians })
  }
  const angleOf = (t: Target) => {
    if (t.type === 'flash') return (latest.current.optics.rotation * Math.PI) / 180
    return latest.current.scene.shapes.find((sh) => sh.id === t.id)?.angle ?? 0
  }

  const startGesture = () => {
    const g = grab.current
    if (!g || pointers.current.size !== 2) return
    const [a, b] = [...pointers.current.values()]
    const pos = positionOf(g.target)
    if (!pos) return
    const shape = g.target.type === 'shape' ? (pos as Shape) : null
    gesture.current = {
      target: g.target,
      angle: Math.atan2(b.y - a.y, b.x - a.x),
      dist: Math.hypot(b.x - a.x, b.y - a.y),
      rot: angleOf(g.target),
      r: shape?.r ?? 0,
      ox: pos.x - (a.x + b.x) / 2,
      oy: pos.y - (a.y + b.y) / 2,
    }
  }

  const onPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    const p = toWorld(e.clientX, e.clientY)
    pointers.current.set(e.pointerId, p)
    svg.current!.setPointerCapture(e.pointerId)
    if (pointers.current.size === 2) return startGesture()
    if (pointers.current.size > 2) return
    const hit = targetOf(e.target)
    if (!hit) {
      grab.current = null
      setSelected(null)
      return
    }
    const pos = positionOf(hit.target)
    if (!pos) return
    grab.current = { ...hit, dx: pos.x - p.x, dy: pos.y - p.y }
    if (hit.target.type === 'shape') setSelected(hit.target.id)
    else setSelected(null)
  }

  const onPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!pointers.current.has(e.pointerId)) return
    const p = toWorld(e.clientX, e.clientY)
    pointers.current.set(e.pointerId, p)
    const ges = gesture.current
    if (ges && pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()]
      const angle = Math.atan2(b.y - a.y, b.x - a.x)
      const dist = Math.hypot(b.x - a.x, b.y - a.y)
      rotateTarget(ges.target, ges.rot + angle - ges.angle)
      if (ges.target.type === 'shape') {
        updateShape(ges.target.id, { r: Math.max(0.06, Math.min(0.8, (ges.r * dist) / Math.max(ges.dist, 1e-3))) })
      }
      moveTarget(ges.target, (a.x + b.x) / 2 + ges.ox, (a.y + b.y) / 2 + ges.oy)
      return
    }
    const g = grab.current
    if (!g || pointers.current.size !== 1) return
    if (g.mode === 'move') return moveTarget(g.target, p.x + g.dx, p.y + g.dy)
    const pos = positionOf(g.target)
    if (!pos) return
    const a = Math.atan2(p.y - pos.y, p.x - pos.x)
    // A shape's handle sits "above" it (at -90° when unrotated); the flashlight's straight ahead.
    rotateTarget(g.target, g.target.type === 'flash' ? a : a + Math.PI / 2)
  }

  const onPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    pointers.current.delete(e.pointerId)
    if (pointers.current.size < 2) gesture.current = null
    if (pointers.current.size === 0) grab.current = null
    else if (pointers.current.size === 1 && grab.current) {
      // Lifting one finger of a gesture: keep dragging with the other from where it is.
      const [p] = [...pointers.current.values()]
      const pos = positionOf(grab.current.target)
      if (pos) grab.current = { ...grab.current, mode: 'move', dx: pos.x - p.x, dy: pos.y - p.y }
    }
  }

  // Scroll over a shape or the flashlight turns it; pinch-zoom on a trackpad (ctrl + wheel) resizes shapes.
  const wheelRef = useRef<(e: WheelEvent) => void>(() => {})
  useEffect(() => {
    wheelRef.current = (e) => {
      const hit = targetOf(e.target)
      if (!hit) return
      e.preventDefault()
      const t = hit.target
      if (t.type === 'shape' && e.ctrlKey) {
        const s = latest.current.scene.shapes.find((sh) => sh.id === t.id)
        if (s) updateShape(t.id, { r: Math.max(0.06, Math.min(0.8, s.r * Math.exp(-e.deltaY * 0.01))) })
        return
      }
      rotateTarget(t, angleOf(t) + e.deltaY * 0.0025)
    }
  })
  useEffect(() => {
    const el = svg.current!
    const onWheel = (e: WheelEvent) => wheelRef.current(e)
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (selected === null || (e.target as HTMLElement).closest?.('input,textarea,[contenteditable]')) return
      if (e.key === 'Delete' || e.key === 'Backspace') removeShape(selected)
      if (e.key === 'Escape') setSelected(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const selectedShape = scene.shapes.find((s) => s.id === selected) ?? null
  const rot = (optics.rotation * Math.PI) / 180
  const { x: fx, y: fy } = scene.flashlight

  const pill =
    'pointer-events-auto whitespace-nowrap rounded-full border border-white/15 bg-black/40 px-3 py-2 font-mono text-[11px] text-white/80 backdrop-blur transition-colors hover:border-white/40 hover:text-white focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:outline-none disabled:opacity-40 sm:py-1.5'

  return (
    <>
      <main className="fixed inset-0 overflow-hidden bg-[#020203] select-none">
        <div ref={container} className="absolute inset-0" />
        <svg
          ref={svg}
          className="absolute inset-0 h-full w-full touch-none"
          viewBox={`${-view.hw} ${-view.hh} ${2 * view.hw} ${2 * view.hh}`}
          preserveAspectRatio="none"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          aria-label="Optical bench: drag the flashlight and shapes"
        >
          <rect
            x={-view.hw}
            y={-view.hh}
            width={2 * view.hw}
            height={2 * view.hh}
            fill="none"
            stroke="rgba(255,255,255,0.16)"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
            pointerEvents="none"
          />
          {scene.shapes.map((s) => (
            <ShapeView key={s.id} shape={s} selected={s.id === selected} />
          ))}
          <g data-target="flash" className="cursor-grab">
            <g transform={`translate(${fx} ${fy}) rotate(${optics.rotation})`}>
              <rect x={-0.13} y={-0.075} width={0.26} height={0.15} fill="transparent" />
              <rect
                x={-0.11}
                y={-0.032}
                width={0.15}
                height={0.064}
                rx={0.014}
                fill="#14161d"
                stroke="rgba(255,255,255,0.55)"
                strokeWidth={1.25}
                vectorEffect="non-scaling-stroke"
              />
              <path
                d={`M0.04 -0.034 L${LENS_OFFSET} -0.05 L${LENS_OFFSET} 0.05 L0.04 0.034 Z`}
                fill="#1c1f28"
                stroke="rgba(255,255,255,0.55)"
                strokeWidth={1.25}
                vectorEffect="non-scaling-stroke"
              />
              <line
                x1={LENS_OFFSET}
                y1={-0.046}
                x2={LENS_OFFSET}
                y2={0.046}
                stroke="#fffbe8"
                strokeWidth={2.5}
                vectorEffect="non-scaling-stroke"
              />
              <rect x={-0.085} y={-0.008} width={0.05} height={0.016} rx={0.006} fill="rgba(255,255,255,0.25)" />
            </g>
          </g>
          <g data-target="flash::rotate" className="cursor-alias">
            <line
              x1={fx + Math.cos(rot) * (LENS_OFFSET + 0.03)}
              y1={fy + Math.sin(rot) * (LENS_OFFSET + 0.03)}
              x2={fx + Math.cos(rot) * FLASH_ARM}
              y2={fy + Math.sin(rot) * FLASH_ARM}
              stroke="rgba(255,255,255,0.25)"
              strokeDasharray="3 4"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
            <circle
              cx={fx + Math.cos(rot) * FLASH_ARM}
              cy={fy + Math.sin(rot) * FLASH_ARM}
              r={0.07}
              fill="transparent"
            />
            <circle
              cx={fx + Math.cos(rot) * FLASH_ARM}
              cy={fy + Math.sin(rot) * FLASH_ARM}
              r={0.022}
              fill="rgba(10,10,14,0.7)"
              stroke="rgba(255,255,255,0.7)"
              strokeWidth={1.25}
              vectorEffect="non-scaling-stroke"
            />
          </g>
        </svg>

        {mode === 'loading' && (
          <p className="pointer-events-none absolute inset-0 grid place-items-center font-mono text-xs text-white/60">
            warming up the light…
          </p>
        )}

        <div
          data-chrome
          className="pointer-events-none absolute top-4 left-4 max-w-[65vw] font-mono text-[11px] tracking-wide text-white/75 lg:top-auto lg:bottom-5 lg:left-5"
        >
          <a
            href={import.meta.env.BASE_URL}
            className="pointer-events-auto text-white/55 transition-colors hover:text-white"
          >
            everydays
          </a>
          <span className="text-white/35"> / </span>
          day 004 / spectral pathtracer
          <div className="mt-1 text-white/45">
            <PathCounter /> paths{mode === 'cpu' && ' · CPU preview (no WebGPU)'}
          </div>
        </div>

        <div
          data-chrome
          className="pointer-events-none absolute bottom-4 left-1/2 flex w-max max-w-[96vw] -translate-x-1/2 flex-col items-center gap-2 sm:bottom-5"
        >
          <AnimatePresence>
            {selectedShape && (
              <motion.div
                key="selected"
                initial={{ opacity: 0, y: 8, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.96 }}
                transition={{ type: 'spring', stiffness: 500, damping: 34 }}
                className="flex items-center gap-2"
              >
                <span className="hidden font-mono text-[11px] text-white/55 sm:inline">{selectedShape.kind}</span>
                <MaterialPicker
                  id="selected-material"
                  value={selectedShape.material}
                  onChange={(material) => updateShape(selectedShape.id, { material })}
                />
                {selectedShape.material === 'diffuse' && (
                  <button
                    type="button"
                    className={`${pill} flex items-center gap-1.5`}
                    aria-label={`tint: ${selectedShape.tint ?? 'white'} (click for the next)`}
                    onClick={() => {
                      const tint = selectedShape.tint ?? 'white'
                      const next = TINT_ORDER[(TINT_ORDER.indexOf(tint) + 1) % TINT_ORDER.length]
                      updateShape(selectedShape.id, { tint: next })
                    }}
                  >
                    <span
                      className="h-2.5 w-2.5 rounded-full ring-1 ring-white/40"
                      style={{ background: TINTS[selectedShape.tint ?? 'white'].swatch }}
                    />
                    {selectedShape.tint ?? 'white'}
                  </button>
                )}
                <button type="button" className={pill} onClick={() => removeShape(selectedShape.id)}>
                  remove
                </button>
              </motion.div>
            )}
          </AnimatePresence>
          <div className="flex items-center gap-1.5 sm:gap-2">
            <MaterialPicker id="new-material" value={newMaterial} onChange={setNewMaterial} />
            {(['triangle', 'circle', 'hexagon'] as const).map((k) => (
              <button
                key={k}
                type="button"
                className={`${pill} flex items-center gap-1.5`}
                disabled={scene.shapes.length >= MAX_SHAPES}
                onClick={() => addShape(k)}
                aria-label={`add ${newMaterial} ${k}`}
              >
                <KindIcon kind={k} />
                <span className="hidden sm:inline">{k}</span>
              </button>
            ))}
          </div>
          <p className="hidden font-mono text-[10px] text-white/40 md:block">
            drag to move · drag the handle or scroll to rotate · ctrl+scroll resizes · two fingers turn and resize
          </p>
        </div>
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

const MATERIALS: Material[] = ['glass', 'mirror', 'diffuse']

function MaterialPicker({ id, value, onChange }: { id: string; value: Material; onChange: (m: Material) => void }) {
  return (
    <div className="pointer-events-auto flex rounded-full border border-white/15 bg-black/40 p-0.5 font-mono text-[11px] backdrop-blur">
      {MATERIALS.map((m) => (
        <button
          key={m}
          type="button"
          aria-pressed={value === m}
          onClick={() => onChange(m)}
          className={`relative rounded-full px-2 py-1.5 transition-colors sm:px-2.5 sm:py-1 ${value === m ? 'text-black' : 'text-white/70 hover:text-white'}`}
        >
          {value === m && (
            <motion.span
              layoutId={id}
              className="absolute inset-0 rounded-full bg-white/85"
              transition={{ type: 'spring', stiffness: 500, damping: 36 }}
            />
          )}
          <span className="relative">{m}</span>
        </button>
      ))}
    </div>
  )
}

const SHAPE_STYLE: Record<Material, { fill: string; stroke: string; strokeWidth: number }> = {
  glass: { fill: 'rgba(170,205,255,0.045)', stroke: 'rgba(205,228,255,0.5)', strokeWidth: 1.25 },
  mirror: { fill: 'rgba(150,160,180,0.22)', stroke: 'rgba(240,244,252,0.95)', strokeWidth: 2.25 },
  diffuse: { fill: '', stroke: 'rgba(0,0,0,0.35)', strokeWidth: 1 },
}

function ShapeView({ shape: s, selected }: { shape: Shape; selected: boolean }) {
  const base = SHAPE_STYLE[s.material]
  const style = {
    ...base,
    // Matte shapes are solid: filled with their tint, so they read as opaque objects.
    fill: s.material === 'diffuse' ? TINTS[s.tint ?? 'white'].swatch : base.fill,
    vectorEffect: 'non-scaling-stroke' as const,
  }
  const outline =
    s.kind === 'circle' ? (
      <circle cx={s.x} cy={s.y} r={s.r} {...style} />
    ) : (
      <polygon points={polygon(s).join(' ')} {...style} strokeLinejoin="round" />
    )
  const hx = s.x + Math.sin(s.angle) * (s.r + ROTATE_ARM)
  const hy = s.y - Math.cos(s.angle) * (s.r + ROTATE_ARM)
  return (
    <g>
      <g data-target={`shape:${s.id}`} className="cursor-grab">
        {outline}
        {selected && (
          <circle
            cx={s.x}
            cy={s.y}
            r={s.r + 0.03}
            fill="none"
            stroke="rgba(255,255,255,0.35)"
            strokeDasharray="4 5"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        )}
      </g>
      {selected && (
        <g data-target={`shape:${s.id}:rotate`} className="cursor-alias">
          <line
            x1={s.x + Math.sin(s.angle) * (s.r + 0.03)}
            y1={s.y - Math.cos(s.angle) * (s.r + 0.03)}
            x2={hx}
            y2={hy}
            stroke="rgba(255,255,255,0.4)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
          <circle cx={hx} cy={hy} r={0.07} fill="transparent" />
          <circle
            cx={hx}
            cy={hy}
            r={0.024}
            fill="rgba(10,10,14,0.7)"
            stroke="white"
            strokeWidth={1.25}
            vectorEffect="non-scaling-stroke"
          />
        </g>
      )}
    </g>
  )
}

function KindIcon({ kind }: { kind: ShapeKind }) {
  return (
    <svg viewBox="-1 -1 2 2" className="h-3 w-3" aria-hidden>
      {kind === 'circle' ? (
        <circle r={0.8} fill="none" stroke="currentColor" strokeWidth={0.22} />
      ) : (
        <polygon
          points={polygon({ id: 0, kind, material: 'glass', x: 0, y: 0.08, r: 0.9, angle: 0 }).join(' ')}
          fill="none"
          stroke="currentColor"
          strokeWidth={0.22}
          strokeLinejoin="round"
        />
      )}
    </svg>
  )
}

/** Path counts arrive every frame; a small store re-renders only the counter. */
const pathStore = (() => {
  let paths = 0
  let shown = 0
  const listeners = new Set<() => void>()
  return {
    set: (p: number) => {
      const now = performance.now()
      if (now - shown < 150 && p !== 0) return
      shown = now
      paths = p
      for (const l of listeners) l()
    },
    get: () => paths,
    subscribe: (l: () => void) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
  }
})()

function PathCounter() {
  const paths = useSyncExternalStore(pathStore.subscribe, pathStore.get)
  const text =
    paths >= 1e9
      ? `${(paths / 1e9).toFixed(2)}B`
      : paths >= 1e6
        ? `${(paths / 1e6).toFixed(1)}M`
        : `${Math.round(paths / 1e3)}k`
  return <>{text}</>
}
