import { DialRoot, useDialKitController, type DialConfig } from 'dialkit'
import {
  AnimatePresence,
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
  useTransform,
  type MotionValue,
} from 'motion/react'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useTilt } from '../001/useTilt.ts'
import { axesAt, ends, fitLines, influence, type Mode } from './field.ts'

const FONT = "'Roboto Flex Magnetic', sans-serif"
const LINE_HEIGHT = 0.92
const PHRASES = ['magnetic', 'squish', 'hello there', 'variable fonts', 'type is a material', 'boing boing']
const INK = '#ebe6da'
const ACCENT: Record<Mode, string> = { bulge: '#ff7a45', dent: '#7cc8ff' }

const dialConfig = {
  // In em: how far from the pointer letters still feel it.
  'falloff radius': [2.4, 0.5, 6, 0.05],
  'max weight': [920, 200, 1000, 10],
  'max width': [140, 40, 151, 1],
  stiffness: [260, 20, 900, 5],
  damping: [15, 2, 60, 0.5],
  // In em.
  tracking: [0, -0.08, 0.2, 0.005],
  mode: { type: 'select', options: ['bulge', 'dent'], default: 'bulge' },
  'next phrase': { type: 'action' },
} satisfies DialConfig

const segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' })
const graphemes = (s: string) => Array.from(segmenter.segment(s), (g) => g.segment)

type Probe = { x: number; y: number }
type Registered = { el: HTMLElement; target: MotionValue<number> }
type Shared = {
  restW: MotionValue<number>
  peakW: MotionValue<number>
  restD: MotionValue<number>
  peakD: MotionValue<number>
  accent: MotionValue<string>
}

declare global {
  interface Window {
    magnetic?: { probe: (x: number, y: number) => void; release: () => void }
  }
}

function useViewport() {
  const [size, setSize] = useState(() => ({ width: window.innerWidth, height: window.innerHeight }))
  useEffect(() => {
    const onResize = () => setSize({ width: window.innerWidth, height: window.innerHeight })
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return size
}

export function MagneticType() {
  const actionRef = useRef<(path: string) => void>(() => {})
  const dial = useDialKitController('day 005 · magnetic type', dialConfig, {
    onAction: (path) => actionRef.current(path),
  })
  const v = dial.values
  const mode = v.mode as Mode
  const reduced = useReducedMotion() ?? false

  const [phraseIndex, setPhraseIndex] = useState(0)
  const [custom, setCustom] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const text = (custom ?? PHRASES[phraseIndex]).trim() || 'type here'
  const next = useCallback(() => {
    setCustom(null)
    setEditing(false)
    setPhraseIndex((i) => (i + 1) % PHRASES.length)
  }, [])
  useEffect(() => {
    actionRef.current = (path) => {
      if (path === 'next phrase') next()
    }
  }, [next])

  // Layout: break the phrase into lines and size it to fill the view at the resting axes.
  const size = useViewport()
  const measure = useRef<HTMLSpanElement>(null)
  const [fontReady, setFontReady] = useState(false)
  useEffect(() => {
    void document.fonts.load(`400 100px ${FONT}`).then(() => setFontReady(true))
  }, [])
  const end = ends(mode, v['max weight'], v['max width'])
  const [layout, setLayout] = useState<{ lines: string[]; fontSize: number }>({ lines: [], fontSize: 0 })
  useLayoutEffect(() => {
    const m = measure.current
    if (!m) return
    m.style.fontVariationSettings = `'wght' ${end.rest.wght}, 'wdth' ${end.rest.wdth}`
    m.style.letterSpacing = `${v.tracking}em`
    const widthOf = (t: string) => {
      m.textContent = t
      return m.getBoundingClientRect().width / 100
    }
    // Swelling letters push their neighbours out, so bulge mode leaves room at the sides.
    const width = size.width * (mode === 'bulge' ? (size.width < 640 ? 0.78 : 0.68) : size.width < 640 ? 0.9 : 0.8)
    const height = size.height * (size.width < 640 ? 0.6 : 0.66)
    setLayout(fitLines(text.split(/\s+/), widthOf, { width, height, lineHeight: LINE_HEIGHT }))
  }, [text, size.width, size.height, mode, end.rest.wght, end.rest.wdth, v.tracking, fontReady])

  // Axis ends are shared motion values, so letters at rest follow the dials without re-rendering.
  const shared: Shared = {
    restW: useMotionValue(end.rest.wght),
    peakW: useMotionValue(end.peak.wght),
    restD: useMotionValue(end.rest.wdth),
    peakD: useMotionValue(end.peak.wdth),
    accent: useMotionValue(ACCENT[mode]),
  }
  useEffect(() => {
    shared.restW.set(end.rest.wght)
    shared.peakW.set(end.peak.wght)
    shared.restD.set(end.rest.wdth)
    shared.peakD.set(end.peak.wdth)
    shared.accent.set(ACCENT[mode])
  })

  // Reduced motion: a softer, critically damped spring and a smaller swell.
  const spring = reduced
    ? { stiffness: 140, damping: 2 * Math.sqrt(140) * 1.1 }
    : { stiffness: v.stiffness, damping: v.damping }
  const amplitude = reduced ? 0.65 : 1

  // The field: each frame, every letter's target is the bell of its distance to the probe.
  const letters = useRef(new Set<Registered>())
  const register = useCallback((r: Registered, on: boolean) => {
    if (on) letters.current.add(r)
    else letters.current.delete(r)
  }, [])
  const pointer = useRef<Probe | null>(null)
  const gravity = useRef({ x: 0, smooth: 0 })
  const stage = useRef<HTMLElement>(null)
  const block = useRef<HTMLDivElement>(null)
  const params = useRef({ radius: 0, amplitude, tilt: false })
  const { status: tilt, request: requestTilt } = useTilt(true, (x) => {
    gravity.current.x = x
  })
  const radius = v['falloff radius'] * layout.fontSize
  useEffect(() => {
    params.current = { radius, amplitude, tilt: tilt === 'active' }
  }, [radius, amplitude, tilt])

  useEffect(() => {
    let frame = 0
    const tick = () => {
      frame = requestAnimationFrame(tick)
      const { radius, amplitude, tilt } = params.current
      const g = gravity.current
      g.smooth += (g.x - g.smooth) * 0.12
      let probe = pointer.current
      if (!probe && tilt && block.current) {
        // Tilt pushes the bulge sideways along the middle of the text.
        const r = block.current.getBoundingClientRect()
        const sway = Math.max(-1, Math.min(1, g.smooth * 2))
        probe = { x: r.left + r.width / 2 + (sway * r.width) / 2, y: r.top + r.height / 2 }
      }
      for (const { el, target } of letters.current) {
        if (!probe) {
          target.set(0)
          continue
        }
        const r = el.getBoundingClientRect()
        const d = Math.hypot(r.left + r.width / 2 - probe.x, r.top + r.height / 2 - probe.y)
        target.set(amplitude * influence(d, radius))
      }
    }
    frame = requestAnimationFrame(tick)
    window.magnetic = {
      probe: (x, y) => (pointer.current = { x, y }),
      release: () => (pointer.current = null),
    }
    return () => cancelAnimationFrame(frame)
  }, [])

  const down = useRef<{ x: number; y: number; t: number } | null>(null)
  const onPointerDown = (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest('[data-ui]')) return
    pointer.current = { x: e.clientX, y: e.clientY }
    down.current = { x: e.clientX, y: e.clientY, t: performance.now() }
  }
  const onPointerMove = (e: React.PointerEvent) => {
    if (e.pointerType !== 'mouse' && !down.current) return
    pointer.current = { x: e.clientX, y: e.clientY }
  }
  const onPointerUp = (e: React.PointerEvent) => {
    const d = down.current
    down.current = null
    if (e.pointerType !== 'mouse') pointer.current = null
    if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 10 && performance.now() - d.t < 500) {
      if (editing) setEditing(false)
      else next()
    }
  }
  const onPointerLeave = (e: React.PointerEvent) => {
    if (e.pointerType === 'mouse') pointer.current = null
  }

  const touch = typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches
  const pill =
    'pointer-events-auto whitespace-nowrap rounded-full border border-white/15 bg-black/40 px-3 py-2 font-mono text-[11px] text-white/80 backdrop-blur transition-colors hover:border-white/40 hover:text-white focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:outline-none sm:py-1.5'

  return (
    <>
      <main
        ref={stage}
        className="fixed inset-0 touch-none overflow-hidden bg-[#0c0b0a] select-none"
        style={{ backgroundImage: 'radial-gradient(120% 90% at 50% 45%, #1a1714 0%, #0c0b0a 60%, #070606 100%)' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          down.current = null
          pointer.current = null
        }}
        onPointerLeave={onPointerLeave}
      >
        <span
          ref={measure}
          aria-hidden
          className="invisible absolute top-0 left-0 whitespace-pre"
          style={{ fontFamily: FONT, fontSize: 100 }}
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <AnimatePresence mode="popLayout" initial={false}>
            <motion.div
              key={editing ? 'editing' : text}
              ref={block}
              role="img"
              aria-label={text}
              className="text-center whitespace-pre"
              style={{
                fontFamily: FONT,
                fontSize: layout.fontSize,
                lineHeight: LINE_HEIGHT,
                letterSpacing: `${v.tracking}em`,
                visibility: fontReady ? 'visible' : 'hidden',
              }}
              initial={reduced ? { opacity: 0 } : { opacity: 0, scale: 0.94, filter: 'blur(14px)' }}
              animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
              exit={reduced ? { opacity: 0 } : { opacity: 0, scale: 1.04, filter: 'blur(14px)' }}
              transition={{ duration: reduced ? 0.15 : 0.35, ease: [0.2, 0.7, 0.2, 1] }}
            >
              {layout.lines.map((line, l) => (
                <div key={l}>
                  {graphemes(line).map((ch, c) =>
                    ch === ' ' ? (
                      <span key={c}> </span>
                    ) : (
                      <Letter key={`${c}${ch}`} char={ch} spring={spring} shared={shared} register={register} />
                    ),
                  )}
                </div>
              ))}
            </motion.div>
          </AnimatePresence>
        </div>

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
          day 005 / magnetic type
          {reduced && <div className="mt-1 text-white/45">reduced motion: gentle springs</div>}
        </div>

        <div
          data-chrome
          data-ui
          className="pointer-events-none absolute bottom-4 left-1/2 flex w-max max-w-[96vw] -translate-x-1/2 flex-col items-center gap-2 sm:bottom-5"
        >
          <AnimatePresence>
            {editing && (
              <motion.input
                key="input"
                autoFocus
                maxLength={40}
                value={custom ?? ''}
                placeholder="type anything"
                onChange={(e) => setCustom(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === 'Escape') setEditing(false)
                }}
                aria-label="your text"
                className="pointer-events-auto w-[min(80vw,320px)] rounded-full border border-white/20 bg-black/50 px-4 py-2 text-center font-mono text-sm text-white backdrop-blur outline-none placeholder:text-white/35 focus:border-white/45"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 8 }}
              />
            )}
          </AnimatePresence>
          <div className="flex items-center gap-1.5 sm:gap-2">
            <button type="button" className={pill} onClick={next}>
              next<span className="hidden sm:inline"> phrase</span>
            </button>
            <button
              type="button"
              className={pill}
              aria-pressed={editing}
              onClick={() => {
                if (!editing && custom === null) setCustom(PHRASES[phraseIndex])
                setEditing((e) => !e)
              }}
            >
              {editing ? 'done' : 'edit'}
              <span className="hidden sm:inline">{editing ? '' : ' text'}</span>
            </button>
            <div className="pointer-events-auto flex rounded-full border border-white/15 bg-black/40 p-0.5 font-mono text-[11px] backdrop-blur">
              {(['bulge', 'dent'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  aria-pressed={mode === m}
                  onClick={() => dial.setValue('mode', m)}
                  className={`relative rounded-full px-2.5 py-1.5 transition-colors sm:py-1 ${mode === m ? 'text-black' : 'text-white/70 hover:text-white'}`}
                >
                  {mode === m && (
                    <motion.span
                      layoutId="mode"
                      className="absolute inset-0 rounded-full bg-white/85"
                      transition={{ type: 'spring', stiffness: 500, damping: 36 }}
                    />
                  )}
                  <span className="relative">{m}</span>
                </button>
              ))}
            </div>
            {tilt === 'prompt' && (
              <button type="button" className={pill} onClick={() => void requestTilt()}>
                <span className="hidden sm:inline">enable </span>tilt
              </button>
            )}
          </div>
          <p className="font-mono text-[10px] text-white/40">
            {touch
              ? `drag across the letters · tap to change${tilt === 'active' ? ' · tilt to push' : ''}`
              : 'move through the letters · click to change the phrase'}
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

function Letter({
  char,
  spring,
  shared,
  register,
}: {
  char: string
  spring: { stiffness: number; damping: number }
  shared: Shared
  register: (r: Registered, on: boolean) => void
}) {
  const target = useMotionValue(0)
  const i = useSpring(target, spring)
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    const r = { el: ref.current!, target }
    register(r, true)
    return () => register(r, false)
  }, [target, register])
  const fontVariationSettings = useTransform(
    [i, shared.restW, shared.peakW, shared.restD, shared.peakD],
    ([i, rw, pw, rd, pd]: number[]) => {
      const a = axesAt(i, { rest: { wght: rw, wdth: rd }, peak: { wght: pw, wdth: pd } })
      return `'wght' ${a.wght.toFixed(1)}, 'wdth' ${a.wdth.toFixed(2)}`
    },
  )
  const color = useTransform(() => {
    const pct = Math.max(0, Math.min(1, i.get())) * 70
    return `color-mix(in oklab, ${INK}, ${shared.accent.get()} ${pct.toFixed(1)}%)`
  })
  return (
    <motion.span ref={ref} style={{ fontVariationSettings, color }}>
      {char}
    </motion.span>
  )
}
