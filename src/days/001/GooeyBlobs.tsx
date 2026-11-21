import { DialRoot, useDialKitController, type DialConfig } from 'dialkit'
import {
  AnimatePresence,
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
  type MotionValue,
} from 'motion/react'
import { useCallback, useEffect, useRef, useState, type PointerEvent } from 'react'
import { step, type Body } from './physics.ts'
import { useTilt } from './useTilt.ts'

const MAX_DROPS = 8
const BLOB_PX = 100
// After the last pointer input, tilt takes over the head of the trail.
const POINTER_IDLE_MS = 1200
// Gravity dial 1 at full tilt, in px/s².
const GRAVITY_PX = 2600

const dialConfig = {
  blobs: [8, 3, 12, 1],
  stiffness: [260, 30, 900, 5],
  damping: [22, 4, 60, 0.5],
  trail: [0.1, 0, 0.3, 0.01],
  size: [110, 40, 200, 1],
  blur: [16, 2, 32, 0.5],
  threshold: [0.2, 0, 1, 0.01],
  hue: [322, 0, 360, 1],
  gloss: [0.55, 0, 1, 0.01],
  tilt: true,
  gravity: [1, 0, 3, 0.05],
  friction: [0.9, 0, 4, 0.05],
  bounce: [0.65, 0, 1, 0.01],
  calm: false,
  'reset scene': { type: 'action' },
  'reset dials': { type: 'action' },
} satisfies DialConfig

type Drop = { id: number; x: number; y: number; scale: number }
type SpringOpts = { stiffness: number; damping: number; mass: number }

function springFor(index: number, stiffness: number, damping: number, trail: number, calm: boolean): SpringOpts {
  const s = stiffness * Math.pow(1 - trail, index) * (calm ? 0.35 : 1)
  // Calm mode is critically damped so nothing overshoots or wobbles.
  const d = calm ? Math.max(damping, 2 * Math.sqrt(s)) : damping
  return { stiffness: s, damping: d, mass: 1 }
}

type ChainProps = {
  index: number
  count: number
  sourceX: MotionValue<number>
  sourceY: MotionValue<number>
  stiffness: number
  damping: number
  trail: number
  size: number
  calm: boolean
  color: string
}

// Each blob springs toward the previous one, so laziness compounds down the trail.
function BlobChain(props: ChainProps) {
  const { index, count, sourceX, sourceY, stiffness, damping, trail, size, calm, color } = props
  const opts = springFor(index, stiffness, damping, trail, calm)
  const x = useSpring(sourceX, opts)
  const y = useSpring(sourceY, opts)
  const scale = (size / BLOB_PX) * (1 - (index / Math.max(count - 1, 1)) * 0.45)

  return (
    <>
      <motion.div
        className="pointer-events-none absolute top-0 left-0 rounded-full will-change-transform"
        style={{
          x,
          y,
          scale,
          width: BLOB_PX,
          height: BLOB_PX,
          marginLeft: -BLOB_PX / 2,
          marginTop: -BLOB_PX / 2,
          background: color,
        }}
      />
      {index + 1 < count && <BlobChain {...props} index={index + 1} sourceX={x} sourceY={y} />}
    </>
  )
}

type DropProps = {
  drop: Drop
  size: number
  color: string
  bodies: Map<number, Body>
  popIn: object
}

// Position comes from the physics loop when tilt is live; otherwise the drop rests where it landed.
function DropBlob({ drop, size, color, bodies, popIn }: DropProps) {
  const d = size * drop.scale
  const x = useMotionValue(drop.x)
  const y = useMotionValue(drop.y)

  useEffect(() => {
    bodies.set(drop.id, { x: drop.x, y: drop.y, vx: 0, vy: 0, r: 0, mx: x, my: y })
    return () => {
      bodies.delete(drop.id)
    }
  }, [bodies, drop, x, y])

  useEffect(() => {
    const body = bodies.get(drop.id)
    if (body) body.r = d / 2
  }, [bodies, drop.id, d])

  return (
    <motion.div className="pointer-events-none absolute top-0 left-0" style={{ x, y }}>
      <motion.div
        {...popIn}
        className="rounded-full"
        style={{ width: d, height: d, marginLeft: -d / 2, marginTop: -d / 2, background: color }}
      />
    </motion.div>
  )
}

export function GooeyBlobs() {
  const prefersReduced = useReducedMotion() ?? false
  const actionRef = useRef<(path: string) => void>(() => {})
  const dial = useDialKitController('day 001 · goo', dialConfig, {
    onAction: (path) => actionRef.current(path),
  })
  const v = dial.values
  const calm = prefersReduced || v.calm

  const pointerX = useMotionValue(typeof window === 'undefined' ? 0 : window.innerWidth / 2)
  const pointerY = useMotionValue(typeof window === 'undefined' ? 0 : window.innerHeight / 2)
  const [drops, setDrops] = useState<Drop[]>([])
  const nextId = useRef(0)
  const downAt = useRef<{ x: number; y: number } | null>(null)

  const resetScene = useCallback(() => {
    setDrops([])
    pointerX.set(window.innerWidth / 2)
    pointerY.set(window.innerHeight / 2)
  }, [pointerX, pointerY])

  useEffect(() => {
    actionRef.current = (path) => {
      if (path === 'reset scene') resetScene()
      if (path === 'reset dials') dial.resetValues()
    }
  })

  const lastPointerAt = useRef(-Infinity)
  const pointerHeld = useRef(false)

  useEffect(() => {
    const move = (e: globalThis.PointerEvent) => {
      pointerX.set(e.clientX)
      pointerY.set(e.clientY)
      lastPointerAt.current = performance.now()
    }
    const down = (e: globalThis.PointerEvent) => {
      pointerHeld.current = true
      move(e)
    }
    const up = () => {
      pointerHeld.current = false
      lastPointerAt.current = performance.now()
    }
    window.addEventListener('pointermove', move, { passive: true })
    window.addEventListener('pointerdown', down, { passive: true })
    window.addEventListener('pointerup', up, { passive: true })
    window.addEventListener('pointercancel', up, { passive: true })
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerdown', down)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointercancel', up)
    }
  }, [pointerX, pointerY])

  const gravityRef = useRef({ x: 0, y: 0 })
  const onGravity = useCallback((gx: number, gy: number) => {
    // Light low-pass to take the jitter out of raw sensor readings.
    const g = gravityRef.current
    g.x = g.x * 0.5 + gx * 0.5
    g.y = g.y * 0.5 + gy * 0.5
  }, [])
  const tilt = useTilt(v.tilt, onGravity)

  const [bodies] = useState(() => new Map<number, Body>())
  // Reduced motion: weaker gravity, more friction, softer bounces.
  const world = {
    gravity: v.gravity * GRAVITY_PX * (calm ? 0.4 : 1),
    friction: v.friction + (calm ? 2.5 : 0),
    bounce: v.bounce * (calm ? 0.5 : 1),
  }
  const worldRef = useRef(world)
  const sizeRef = useRef(v.size)
  useEffect(() => {
    worldRef.current = world
    sizeRef.current = v.size
  })

  const physics = v.tilt && tilt.status === 'active'
  useEffect(() => {
    if (!physics) return
    const head: Body = { x: pointerX.get(), y: pointerY.get(), vx: 0, vy: 0, r: 0, mx: pointerX, my: pointerY }
    let last = performance.now()
    let raf = requestAnimationFrame(function frame(now) {
      // Real elapsed time in fixed substeps, so slow frames don't turn into slow motion.
      // The cap only stops a jump after the tab was hidden.
      const elapsed = Math.min((now - last) / 1000, 0.1)
      last = now
      const steps = Math.max(1, Math.ceil(elapsed * 120))
      const list = [...bodies.values()]
      const idle = !pointerHeld.current && now - lastPointerAt.current > POINTER_IDLE_MS
      if (idle) {
        head.r = sizeRef.current / 2
        list.push(head)
      } else {
        // The pointer owns the head; pick up from wherever it leaves it, at rest.
        head.x = pointerX.get()
        head.y = pointerY.get()
        head.vx = head.vy = 0
      }
      const g = gravityRef.current
      const w = { ...worldRef.current, gx: g.x, gy: g.y, width: window.innerWidth, height: window.innerHeight }
      for (let i = 0; i < steps; i++) step(list, w, elapsed / steps)
      raf = requestAnimationFrame(frame)
    })
    return () => cancelAnimationFrame(raf)
  }, [physics, bodies, pointerX, pointerY])

  const onPointerDown = (e: PointerEvent) => {
    downAt.current = { x: e.clientX, y: e.clientY }
  }

  const onPointerUp = (e: PointerEvent) => {
    const start = downAt.current
    downAt.current = null
    if (!start || Math.hypot(e.clientX - start.x, e.clientY - start.y) > 8) return
    const drop = { id: nextId.current++, x: e.clientX, y: e.clientY, scale: 0.65 + Math.random() * 0.35 }
    setDrops((prev) => [...prev, drop].slice(-MAX_DROPS))
  }

  const color = `hsl(${v.hue} 96% 62%)`
  const contrast = 48 * (1 - v.threshold) + 2.5 * v.threshold
  const offset = contrast * 0.42
  const popIn = calm
    ? { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 }, transition: { duration: 0.4 } }
    : {
        initial: { scale: 0 },
        animate: { scale: 1 },
        exit: { scale: 0 },
        transition: { type: 'spring' as const, stiffness: 420, damping: 18 },
      }

  return (
    <>
      <main
        className="fixed inset-0 overflow-hidden select-none"
        style={{
          touchAction: 'none',
          cursor: 'crosshair',
          background: `radial-gradient(120% 90% at 50% 40%, hsl(${v.hue} 40% 10%) 0%, #0b0a12 55%, #06050a 100%)`,
        }}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
      >
        <svg aria-hidden className="absolute h-0 w-0">
          <defs>
            <filter id="goo" x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
              <feGaussianBlur in="SourceGraphic" stdDeviation={v.blur} result="blur" />
              <feColorMatrix
                in="blur"
                type="matrix"
                values={`1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 ${contrast} -${offset}`}
                result="goo"
              />
              <feGaussianBlur in="goo" stdDeviation="5" result="soft" />
              <feSpecularLighting
                in="soft"
                surfaceScale="5"
                specularConstant={v.gloss * 1.4}
                specularExponent="24"
                lightingColor="#ffffff"
                result="spec"
              >
                <feDistantLight azimuth="235" elevation="52" />
              </feSpecularLighting>
              <feComposite in="spec" in2="goo" operator="in" result="specIn" />
              <feComposite in="goo" in2="specIn" operator="arithmetic" k1="0" k2="1" k3="0.7" k4="0" />
            </filter>
          </defs>
        </svg>

        <div className="absolute inset-0" style={{ filter: 'url(#goo)' }}>
          <AnimatePresence>
            {drops.map((d) => (
              <DropBlob
                key={d.id}
                drop={d}
                size={v.size}
                color={color}
                bodies={bodies}
                popIn={popIn}
              />
            ))}
          </AnimatePresence>
          <BlobChain
            index={0}
            count={v.blobs}
            sourceX={pointerX}
            sourceY={pointerY}
            stiffness={v.stiffness}
            damping={v.damping}
            trail={v.trail}
            size={v.size}
            calm={calm}
            color={color}
          />
        </div>

        <div className="pointer-events-none absolute bottom-4 left-4 max-w-[65vw] font-mono text-[11px] tracking-wide text-white/55 sm:bottom-5 sm:left-5">
          <a href={import.meta.env.BASE_URL} className="pointer-events-auto text-white/35 transition-colors hover:text-white/80">
            everydays
          </a>
          <span className="text-white/25"> / </span>
          day 001 / gooey cursor blobs
          {calm && <span className="block text-white/35 sm:ml-2 sm:inline">· calm mode</span>}
        </div>

        {v.tilt && tilt.status === 'prompt' && (
          <button
            type="button"
            data-chrome
            onPointerDown={(e) => e.stopPropagation()}
            onPointerUp={(e) => e.stopPropagation()}
            onClick={tilt.request}
            className="absolute top-4 left-4 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 font-mono text-[11px] text-white/70 backdrop-blur transition-colors hover:border-white/30 hover:text-white focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:outline-none sm:top-5 sm:left-5"
          >
            enable tilt
          </button>
        )}

        {/* Centered on wider screens, where the open DialKit panel reaches the bottom-right corner. */}
        <div data-chrome className="pointer-events-none absolute right-4 bottom-4 flex items-center gap-3 sm:right-auto sm:bottom-5 sm:left-1/2 sm:-translate-x-1/2">
          <span className="hidden font-mono text-[11px] text-white/30 sm:inline">
            click to drop · {drops.length}/{MAX_DROPS}
            {v.tilt && tilt.status === 'active' && ' · tilt'}
          </span>
          <button
            type="button"
            onPointerDown={(e) => e.stopPropagation()}
            onPointerUp={(e) => e.stopPropagation()}
            onClick={resetScene}
            className="pointer-events-auto rounded-full border border-white/15 bg-white/5 px-3 py-1.5 font-mono text-[11px] text-white/70 backdrop-blur transition-colors hover:border-white/30 hover:text-white focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:outline-none"
          >
            reset
          </button>
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
