import { DialRoot, useDialKitController, type DialConfig } from 'dialkit'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Engine } from './audio.ts'
import { midiName, midiToHz, NOTE_NAMES, SCALES, stringMidi, VOICES, type NoteName, type ScaleName } from './ks.ts'

const dialConfig = {
  strings: [8, 4, 16, 1],
  // Seconds for the fundamental to fall 60 dB.
  decay: [3.5, 0.3, 12, 0.1],
  // The loop filter and the colour of the noise burst.
  brightness: [0.55, 0, 1, 0.01],
  // Fraction of the string from the bridge end (bottom or right).
  'pluck position': [0.18, 0.04, 0.5, 0.01],
  'pluck where crossed': false,
  scale: { type: 'select', options: Object.keys(SCALES), default: 'pentatonic' },
  root: { type: 'select', options: [...NOTE_NAMES], default: 'D' },
  reverb: [0.3, 0, 1, 0.01],
  strum: { type: 'action' },
} satisfies DialConfig

type Settings = {
  count: number
  decay: number
  brightness: number
  position: number
  followCross: boolean
  scale: ScaleName
  root: NoteName
}

type StringState = {
  /** When the last pluck lands, in performance.now() seconds. */
  t0: number
  velocity: number
  /** Where along the string (0 at the nut end) the pluck's peak sits. */
  peak: number
  sign: number
  /** A finger holding the string: pushed `perp` px sideways at `at` along it. */
  held: { perp: number; at: number } | null
}

type Pointer = { x: number; y: number; t: number; hold: number | null; anchorAlong: number }

type Geometry = {
  vertical: boolean
  count: number
  spacing: number
  /** Long-axis coordinate of each string. */
  pos: number[]
  /** Where strings start (nut) and end (bridge) along the short axis. */
  b0: number
  b1: number
}

declare global {
  interface Window {
    pluckLab?: {
      strum: () => void
      audio: () => { state: AudioContextState; ready: boolean; levels: number[] } | null
      plucks: number
    }
  }
}

/** Strings spread along the long side of the view, pitch rising left to right or bottom to top. */
function geometry(w: number, h: number, count: number): Geometry {
  const vertical = w >= h
  const [a0, a1] = vertical ? [w * 0.1, w * 0.9] : [h * 0.17, h * 0.83]
  const spacing = (a1 - a0) / count
  const pos = Array.from({ length: count }, (_, i) => (vertical ? a0 + spacing * (i + 0.5) : a1 - spacing * (i + 0.5)))
  const [b0, b1] = vertical ? [h * 0.12, h * 0.84] : [w * 0.1, w * 0.92]
  return { vertical, count, spacing, pos, b0, b1 }
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const now = () => performance.now() / 1000
const MODES = 8

export function PluckLab() {
  const actionRef = useRef<(path: string) => void>(() => {})
  const dial = useDialKitController('day 006 · pluck lab', dialConfig, {
    onAction: (path) => actionRef.current(path),
  })
  const v = dial.values
  const settings: Settings = {
    count: v.strings,
    decay: v.decay,
    brightness: v.brightness,
    position: v['pluck position'],
    followCross: v['pluck where crossed'],
    scale: v.scale as ScaleName,
    root: v.root as NoteName,
  }
  const live = useRef(settings)
  useEffect(() => {
    live.current = settings
  })
  const reduced = useReducedMotion() ?? false
  const reducedRef = useRef(reduced)
  useEffect(() => {
    reducedRef.current = reduced
  }, [reduced])

  const engine = useRef<Engine | null>(null)
  const [sound, setSound] = useState<'off' | 'starting' | 'on'>('off')
  const startSound = useCallback(() => {
    if (engine.current) {
      if (engine.current.ctx.state !== 'running') void engine.current.ctx.resume()
      return
    }
    engine.current = Engine.start()
    engine.current.setReverb(v.reverb)
    setSound('starting')
    const ctx = engine.current.ctx
    const check = () => setSound(ctx.state === 'running' ? 'on' : 'starting')
    ctx.addEventListener('statechange', check)
    check()
  }, [v.reverb])
  useEffect(() => {
    engine.current?.setReverb(v.reverb)
  }, [v.reverb])

  const strings = useRef<StringState[]>(
    Array.from({ length: VOICES }, () => ({ t0: -1e9, velocity: 0, peak: 0.8, sign: 1, held: null })),
  )
  const canvas = useRef<HTMLCanvasElement>(null)
  const geo = useRef<Geometry>(geometry(window.innerWidth, window.innerHeight, settings.count))

  const pluck = useCallback((i: number, velocity: number, along: number, sign: number, delayMs = 0) => {
    const s = live.current
    const g = geo.current
    if (i >= s.count) return
    const fromEnd = clamp(Math.min(along, 1 - along), 0.04, 0.5)
    const beta = s.followCross ? fromEnd : s.position
    const st = strings.current[i]
    if (window.pluckLab) window.pluckLab.plucks++
    st.t0 = now() + delayMs / 1000
    st.velocity = velocity
    st.peak = s.followCross ? clamp(along, 0.04, 0.96) : 1 - s.position
    st.sign = sign
    const pan = g.count > 1 ? (i / (g.count - 1)) * 1.3 - 0.65 : 0
    engine.current?.pluck(
      i,
      {
        freq: midiToHz(stringMidi(i, s.scale, s.root)),
        velocity,
        position: beta,
        brightness: s.brightness,
        decay: s.decay,
      },
      pan,
      delayMs,
    )
  }, [])

  const strum = useCallback(() => {
    const n = live.current.count
    for (let i = 0; i < n; i++) pluck(i, 0.55 + 0.25 * Math.sin(i * 1.7) ** 2, 0.5, 1, i * 32)
  }, [pluck])
  useEffect(() => {
    actionRef.current = (path) => {
      if (path === 'strum') strum()
    }
    window.pluckLab = {
      strum,
      plucks: 0,
      audio: () => {
        const e = engine.current
        return e && { state: e.ctx.state, ready: e.ready, levels: [...e.levels] }
      },
    }
  }, [strum])

  // Drawing: each string is a polyline displaced by a decaying sum of standing-wave modes.
  useEffect(() => {
    const el = canvas.current!
    const ctx = el.getContext('2d')!
    let frame = 0
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      el.width = Math.round(window.innerWidth * dpr)
      el.height = Math.round(window.innerHeight * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    const envelope = (i: number, st: StringState, t: number) => {
      const age = t - st.t0
      if (age < 0) return 0
      const e = engine.current
      // The worklet's level lags a message round trip, so the first moments come from the model.
      if (e?.ready && age > 0.08) return Math.min(1.2, e.levels[i])
      return st.velocity * Math.exp((-6.9 * age) / Math.max(0.2, live.current.decay * 0.7))
    }

    const draw = () => {
      frame = requestAnimationFrame(draw)
      const s = live.current
      const w = window.innerWidth
      const h = window.innerHeight
      const g = (geo.current = geometry(w, h, s.count))
      const t = now()
      const still = reducedRef.current
      ctx.clearRect(0, 0, w, h)
      const amp = Math.min(g.spacing * 0.42, 64)
      const seg = 64
      const point = (i: number, u: number, d: number): [number, number] => {
        const b = g.b0 + (g.b1 - g.b0) * u
        const a = g.pos[i] + d
        return g.vertical ? [a, b] : [b, a]
      }

      // Nut and bridge.
      ctx.lineCap = 'round'
      ctx.strokeStyle = 'rgba(236,226,206,0.16)'
      ctx.lineWidth = 3
      for (const u of [0, 1]) {
        ctx.beginPath()
        ctx.moveTo(...point(0, u, (g.vertical ? -1 : 1) * g.spacing * 0.5))
        ctx.lineTo(...point(g.count - 1, u, (g.vertical ? 1 : -1) * g.spacing * 0.5))
        ctx.stroke()
      }

      ctx.font = '500 10px ui-monospace, SFMono-Regular, Menlo, monospace'
      ctx.textAlign = g.vertical ? 'center' : 'right'
      ctx.textBaseline = g.vertical ? 'bottom' : 'middle'
      for (let i = 0; i < g.count; i++) {
        const st = strings.current[i]
        const env = envelope(i, st, t)
        const age = Math.max(0, t - st.t0)
        const ratio = g.count > 1 ? i / (g.count - 1) : 0
        // Lower strings swing more slowly; the true pitch would alias at 60 fps.
        const fv = 3.2 + 5.5 * ratio
        const pk = st.peak
        const weights = Array.from({ length: MODES }, (_, k) => {
          const n = k + 1
          const decay = Math.exp(-k * 1.1 * age)
          return (
            ((2 * Math.sin(n * Math.PI * pk)) / (n * n * Math.PI * Math.PI * pk * (1 - pk))) *
            Math.cos(2 * Math.PI * n * fv * age) *
            decay
          )
        })
        const held = st.held
        const disp = (u: number) => {
          let d = 0
          if (!still && env > 1e-3) {
            let sum = 0
            for (let k = 0; k < MODES; k++) sum += weights[k] * Math.sin((k + 1) * Math.PI * u)
            // A gentle power curve keeps quiet tails visible; it still rises and falls with the audio.
            d = st.sign * amp * env ** 0.6 * sum
          }
          if (held) d += held.perp * (u < held.at ? u / held.at : (1 - u) / (1 - held.at))
          return d
        }
        const hue = 32 + ratio * 170
        const glow = clamp(env ** 0.6 * 1.2, 0, 1)
        const width = 1 + 2.4 * (1 - ratio)
        ctx.beginPath()
        for (let k = 0; k <= seg; k++) {
          const [x, y] = point(i, k / seg, disp(k / seg))
          if (k === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        if (glow > 0.01) {
          ctx.strokeStyle = `hsla(${hue}, 90%, 62%, ${0.28 * glow})`
          ctx.lineWidth = width + 7 * glow
          ctx.stroke()
        }
        ctx.strokeStyle = glow > 0.01 ? `hsl(${hue}, ${55 * glow}%, ${78 + 10 * glow}%)` : 'rgba(232,222,200,0.62)'
        ctx.globalAlpha = 0.65 + 0.35 * glow
        ctx.lineWidth = width
        ctx.stroke()
        ctx.globalAlpha = 1

        ctx.fillStyle = 'rgba(236,226,206,0.5)'
        for (const u of [0, 1]) {
          const [x, y] = point(i, u, 0)
          ctx.beginPath()
          ctx.arc(x, y, 2.4, 0, Math.PI * 2)
          ctx.fill()
        }
        const [lx, ly] = point(i, 0, 0)
        ctx.fillStyle = `rgba(236,226,206,${0.32 + 0.5 * glow})`
        ctx.fillText(midiName(stringMidi(i, s.scale, s.root)), g.vertical ? lx : lx - 9, g.vertical ? ly - 9 : ly)
      }
    }
    frame = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', resize)
    }
  }, [])

  // Pointers: crossing a string while dragging plucks it; pressing on one holds it for bends.
  const pointers = useRef(new Map<number, Pointer>())
  const long = (x: number, y: number) => (geo.current.vertical ? x : y)
  const short = (x: number, y: number) => (geo.current.vertical ? y : x)
  const alongOf = (b: number) => (b - geo.current.b0) / (geo.current.b1 - geo.current.b0)

  const hit = (x: number, y: number) => {
    const g = geo.current
    const u = alongOf(short(x, y))
    if (u < 0 || u > 1) return null
    const a = long(x, y)
    const reach = clamp(g.spacing * 0.32, 8, 20)
    let best: number | null = null
    for (let i = 0; i < g.count; i++) {
      if (Math.abs(a - g.pos[i]) < reach && (best === null || Math.abs(a - g.pos[i]) < Math.abs(a - g.pos[best]))) {
        best = i
      }
    }
    return best === null ? null : { i: best, u }
  }

  const bendHeld = (p: Pointer, x: number, y: number) => {
    const g = geo.current
    const i = p.hold!
    const perp = clamp(long(x, y) - g.pos[i], -g.spacing * 0.45, g.spacing * 0.45)
    const along = short(x, y) - p.anchorAlong
    const semis = clamp(Math.hypot(perp, along * 0.35) / 110, 0, 1) * 2
    strings.current[i].held = { perp, at: clamp(alongOf(short(x, y)), 0.05, 0.95) }
    engine.current?.bend(i, 2 ** (semis / 12))
  }

  const release = (id: number) => {
    const p = pointers.current.get(id)
    pointers.current.delete(id)
    if (p?.hold == null) return
    const st = strings.current[p.hold]
    const held = st.held
    st.held = null
    engine.current?.bend(p.hold, 1)
    // Letting go of a pushed string snaps it back: a second, softer pluck.
    if (held && Math.abs(held.perp) > 10) {
      pluck(
        p.hold,
        clamp(Math.abs(held.perp) / (geo.current.spacing * 0.45), 0.15, 0.8) * 0.8,
        held.at,
        -Math.sign(held.perp),
      )
    }
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    startSound()
    e.currentTarget.setPointerCapture(e.pointerId)
    const h = hit(e.clientX, e.clientY)
    pointers.current.set(e.pointerId, {
      x: e.clientX,
      y: e.clientY,
      t: e.timeStamp,
      hold: h?.i ?? null,
      anchorAlong: short(e.clientX, e.clientY),
    })
    if (h) pluck(h.i, 0.5, h.u, 1)
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const p = pointers.current.get(e.pointerId)
    if (!p) return
    const events = e.nativeEvent.getCoalescedEvents?.() ?? []
    for (const ev of events.length ? events : [e.nativeEvent]) {
      const { clientX: x, clientY: y, timeStamp: t } = ev
      if (p.hold !== null) {
        bendHeld(p, x, y)
      } else {
        const g = geo.current
        const a0 = long(p.x, p.y)
        const a1 = long(x, y)
        const dt = Math.max(1, t - p.t)
        const velocity = clamp(0.2 + Math.hypot(x - p.x, y - p.y) / dt / 1.5, 0.2, 1)
        const crossed: { i: number; f: number; u: number }[] = []
        for (let i = 0; i < g.count; i++) {
          const c = g.pos[i]
          if (a0 < c === a1 < c) continue
          const f = (c - a0) / (a1 - a0)
          const u = alongOf(short(p.x + (x - p.x) * f, p.y + (y - p.y) * f))
          if (u >= 0 && u <= 1) crossed.push({ i, f, u })
        }
        crossed.sort((m, n) => m.f - n.f)
        for (const c of crossed) pluck(c.i, velocity, c.u, Math.sign(a1 - a0), c.f * dt)
      }
      p.x = x
      p.y = y
      p.t = t
    }
  }

  const ios =
    /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
  const touch = window.matchMedia('(pointer: coarse)').matches

  return (
    <>
      <main
        className="fixed inset-0 overflow-hidden select-none"
        style={{ background: 'radial-gradient(110% 85% at 50% 50%, #1b1714 0%, #0d0b0a 62%, #060505 100%)' }}
      >
        <canvas
          ref={canvas}
          className="absolute inset-0 h-full w-full touch-none"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={(e) => release(e.pointerId)}
          onPointerCancel={(e) => release(e.pointerId)}
          aria-label="Strings: drag across them to strum, press and drag on one to bend it"
        />

        <AnimatePresence>
          {sound !== 'on' && (
            <motion.div
              data-chrome
              className="pointer-events-none absolute inset-0 flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 1.04 }}
              transition={{ duration: 0.25 }}
            >
              <button
                type="button"
                onPointerDown={startSound}
                className="pointer-events-auto flex items-center gap-2.5 rounded-full border border-white/20 bg-black/55 px-5 py-3 font-mono text-[13px] text-white/90 shadow-[0_0_40px_rgba(255,180,90,0.15)] backdrop-blur transition-colors hover:border-white/45"
              >
                <motion.span
                  className="h-2 w-2 rounded-full bg-amber-300"
                  animate={reduced ? undefined : { opacity: [1, 0.25, 1] }}
                  transition={{ duration: 1.4, repeat: Infinity }}
                />
                {sound === 'starting' ? 'starting sound…' : 'tap to start sound'}
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        <div
          data-chrome
          className="pointer-events-none absolute top-4 left-4 max-w-[70vw] font-mono text-[11px] tracking-wide text-white/75 lg:top-auto lg:bottom-5 lg:left-5"
        >
          <a
            href={import.meta.env.BASE_URL}
            className="pointer-events-auto text-white/55 transition-colors hover:text-white"
          >
            everydays
          </a>
          <span className="text-white/35"> / </span>
          day 006 / pluck lab
          <div className="mt-1 text-white/45">
            {sound === 'on' ? 'sound on' : 'sound off'}
            {ios && ' · no sound? turn off silent mode'}
          </div>
        </div>

        <p
          data-chrome
          className="pointer-events-none absolute bottom-4 left-1/2 w-max max-w-[92vw] -translate-x-1/2 text-center font-mono text-[10px] text-white/40 sm:bottom-5"
        >
          {touch ? 'strum with one or more fingers' : 'drag across the strings to strum'} · press on a string and drag
          to bend
        </p>
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
