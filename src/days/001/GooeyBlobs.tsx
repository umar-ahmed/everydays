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

const MAX_DROPS = 8
const BLOB_PX = 100

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

  useEffect(() => {
    const move = (e: globalThis.PointerEvent) => {
      pointerX.set(e.clientX)
      pointerY.set(e.clientY)
    }
    window.addEventListener('pointermove', move, { passive: true })
    window.addEventListener('pointerdown', move, { passive: true })
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerdown', move)
    }
  }, [pointerX, pointerY])

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
              <motion.div
                key={d.id}
                className="pointer-events-none absolute top-0 left-0"
                style={{ x: d.x, y: d.y }}
              >
                <motion.div
                  {...popIn}
                  className="rounded-full"
                  style={{
                    width: v.size * d.scale,
                    height: v.size * d.scale,
                    marginLeft: (-v.size * d.scale) / 2,
                    marginTop: (-v.size * d.scale) / 2,
                    background: color,
                  }}
                />
              </motion.div>
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
          <a href="/" className="pointer-events-auto text-white/35 transition-colors hover:text-white/80">
            everydays
          </a>
          <span className="text-white/25"> / </span>
          day 001 / gooey cursor blobs
          {calm && <span className="block text-white/35 sm:ml-2 sm:inline">· calm mode</span>}
        </div>

        <div data-chrome className="pointer-events-none absolute right-4 bottom-4 flex items-center gap-3 sm:right-5 sm:bottom-5">
          <span className="hidden font-mono text-[11px] text-white/30 sm:inline">
            click to drop · {drops.length}/{MAX_DROPS}
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
