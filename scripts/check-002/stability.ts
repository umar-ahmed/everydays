// Pop-stability check, in headless Chromium on SwiftShader WebGPU: build a sheet exactly like the
// app does, press one bubble with the finger until it pops, release, and step the real sim with
// no further input. Nothing else may pop, kinetic energy must decay and the sheet must stay flat.
import * as THREE from 'three/webgpu'
import { sheetAspect } from '../../src/days/002/camera.ts'
import { LatticeGenerator, latticeFor, type Shape, type Tiling } from '../../src/days/002/generators/lattice.ts'
import * as simModule from '../../src/days/002/sim.ts'
import { BubbleSim } from '../../src/days/002/sim.ts'
import { FULL_BUDGET, PHONE_BUDGET, buildSheetFromMap, layoutFor } from '../../src/days/002/sheet.ts'

export type Scenario = {
  name: string
  view: { width: number; height: number }
  phone: boolean
  across: number
  size: number
  spacing: number
  tiling: Tiling
  shape: Shape
  /** Edge margin (cm); 0 is the original tight edge. */
  margin: number
  /** Which bubble to press: the one nearest the middle, the middle of the left edge, or a corner. */
  target?: 'center' | 'edge' | 'corner'
}

/** The app's default dials. */
const DEFAULTS = { fill: 0.12, threshold: 0.6, stiffness: 0.75, substeps: 8, pressRadius: 0.75, pressStrength: 0.9 }

export type Sample = { t: number; popped: number; energy: number; maxSpeed: number; height: number; spread: number }
export type StabilityResult = {
  hot?: string[]
  name: string
  bubbles: number
  popAt: number
  /** Seconds of pressing until the pop. */
  pressFor: number
  samples: Sample[]
  failures: string[]
}

const query = new URLSearchParams(typeof location === 'undefined' ? '' : location.search)
/** Frame length; the app steps the real frame time, capped at 1/60 s, so 120 Hz screens give 1/120. */
let seed = 1
const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646
const FRAME = 1 / Number(query.get('hz') ?? 60)
const PINHOLES = Number(query.get('pinholes') ?? 0)
/** Frame-time jitter: each frame is FRAME..2*FRAME (capped at 1/60), like a real display under load. */
const JITTER = query.get('jitter') !== '0'
const SETTLE = Number(query.get('settle') ?? 3)

export async function runScenario(
  renderer: THREE.WebGPURenderer,
  sim: BubbleSim,
  generator: LatticeGenerator,
  sc: Scenario,
  settle = SETTLE,
): Promise<StabilityResult> {
  const failures: string[] = []
  const layout = layoutFor({
    size: sc.size,
    spacing: sc.spacing,
    across: sc.across,
    aspect: sheetAspect(sc.view),
    lattice: latticeFor(sc.tiling),
    margin: sc.margin,
    budget: sc.phone ? PHONE_BUDGET : FULL_BUDGET,
  })
  generator.configure(layout, sc)
  void renderer.compute(generator.compute)
  const raw = new Float32Array(
    await renderer.getArrayBufferAsync(sim.mapRaw.value, null, 0, layout.mapW * layout.mapH * 16),
  )
  const sheet = buildSheetFromMap({ layout, raw, pinholeChance: PINHOLES })
  sim.load(sheet, DEFAULTS.fill)
  sim.setParams({ ...DEFAULTS, tool: 'finger' }, sheet)
  // Experiment switches.
  if (query.has('nofloor')) sim.capGap.value = -1e3
  if (query.has('noback')) sim.backIterations = 0
  if (query.has('iters')) sim.iterations = Number(query.get('iters'))
  if (query.has('omega')) sim.omega.value = Number(query.get('omega'))
  const n = sheet.W * sheet.H

  const aim = { center: [0, 0], edge: [-sheet.width / 2, 0], corner: [-sheet.width / 2, -sheet.depth / 2] }[
    sc.target ?? (query.get('target') as 'center' | 'edge' | 'corner' | null) ?? 'center'
  ]
  const far = (b: number) => Math.hypot(sheet.centers[b * 4] - aim[0], sheet.centers[b * 4 + 1] - aim[1])
  let target = 0
  for (let b = 1; b < sheet.count; b++) if (far(b) < far(target)) target = b
  const [cx, cz] = [sheet.centers[target * 4], sheet.centers[target * 4 + 1]]
  const r = DEFAULTS.pressRadius * sheet.size
  const bottom = Math.max(sheet.domeHeight * (1 - DEFAULTS.pressStrength), 0.01)
  let fingerY = sheet.domeHeight + r + 1
  // Fixed sim ticks like the app; a build without FixedStep (for before/after runs) steps raw frames.
  const Clock = (simModule as { FixedStep?: new () => { dt: number; ticks(dt: number): number } }).FixedStep
  const clock = Clock ? new Clock() : null
  let time = 0

  const read = async () => {
    const [x, v, pops] = await Promise.all([
      renderer.getArrayBufferAsync(sim.x.value, null, 0, n * 16),
      renderer.getArrayBufferAsync(sim.v.value, null, 0, n * 16),
      renderer.getArrayBufferAsync(sim.pops.value, null, 0, sheet.count * 16),
    ])
    const X = new Float32Array(x)
    const V = new Float32Array(v)
    const P = new Float32Array(pops)
    let energy = 0
    let maxSpeed = 0
    let minY = Infinity
    let maxY = -Infinity
    let spread = 0
    for (let k = 0; k < n; k++) {
      const s2 = V[k * 4] ** 2 + V[k * 4 + 1] ** 2 + V[k * 4 + 2] ** 2
      // Mass per vertex ~ its area (h^2), so energy is comparable across sheet sizes.
      energy += 0.5 * s2 * sheet.h * sheet.h
      maxSpeed = Math.max(maxSpeed, Math.sqrt(s2))
      minY = Math.min(minY, X[k * 4 + 1])
      maxY = Math.max(maxY, X[k * 4 + 1])
      // How far any vertex has moved in the sheet plane from its rest spot (roll-up pulls edges in).
      spread = Math.max(spread, Math.hypot(X[k * 4] - sheet.rest[k * 4], X[k * 4 + 2] - sheet.rest[k * 4 + 2]))
    }
    let popped = 0
    for (let b = 0; b < sheet.count; b++) if (P[b * 4] > 0.5) popped++
    return { popped, energy, maxSpeed, height: maxY - minY, spread }
  }

  const poppedCount = async () => {
    const P = new Float32Array(await renderer.getArrayBufferAsync(sim.pops.value, null, 0, sheet.count * 16))
    let popped = 0
    for (let b = 0; b < sheet.count; b++) if (P[b * 4] > 0.5) popped++
    return popped
  }

  const frame = (pressing: boolean) => {
    const want = pressing ? bottom + r : sheet.domeHeight + r + 1
    fingerY += (want - fingerY) * (1 - Math.exp(-FRAME * (pressing ? 14 : 30)))
    sim.finger.value.set(cx, fingerY, cz)
    // The app's path: display frames of any length feed fixed sim ticks.
    const dt = JITTER ? FRAME * (1 + rand()) : FRAME
    if (!clock) {
      time += Math.min(dt, 1 / 60)
      sim.time.value = time
      sim.step(renderer, Math.min(dt, 1 / 60), DEFAULTS.substeps)
      return
    }
    for (let k = clock.ticks(dt); k > 0; k--) {
      time += clock.dt
      sim.time.value = time
      sim.step(renderer, clock.dt, DEFAULTS.substeps)
    }
  }

  // Let the sheet settle under its fill pressure first.
  while (time < 1) frame(false)
  const rest = await read()
  if (rest.popped) failures.push(`${rest.popped} bubbles popped while settling, before any press`)

  // Press until the target pops (at most 4 s), then let go.
  let popAt = -1
  const pressStart = time
  let pressFor = -1
  for (let f = 0; time < pressStart + 10 && popAt < 0; f++) {
    frame(true)
    if (query.has('verbose') && f % 30 === 0) {
      const S = new Float32Array(await renderer.getArrayBufferAsync(sim.state.value, null, 0, sheet.count * 16))
      const X = new Float32Array(await renderer.getArrayBufferAsync(sim.x.value, null, 0, n * 16))
      let capMin = Infinity
      let capMax = -Infinity
      for (let k = 0; k < n; k++) {
        if (sheet.meta[k * 4 + 1] > 0.5 && sheet.meta[k * 4] === target) {
          capMin = Math.min(capMin, X[k * 4 + 1])
          capMax = Math.max(capMax, X[k * 4 + 1])
        }
      }
      console.log(
        `press t=${(time - pressStart).toFixed(2)} target ${target} at (${cx.toFixed(2)},${cz.toFixed(2)}) gauge ${S[target * 4].toFixed(3)} V/V0 ${(S[target * 4 + 1] / sheet.bubbles[target * 4 + 3]).toFixed(3)} fingerY ${fingerY.toFixed(2)} cap y ${capMin.toFixed(3)}..${capMax.toFixed(3)}`,
      )
    }
    if (f % 6 === 5 && (await poppedCount()) > 0) {
      popAt = time
      pressFor = time - pressStart
    }
  }
  if (popAt < 0) failures.push('the pressed bubble never popped')

  const samples: Sample[] = []
  const frames = Math.round(settle / FRAME)
  for (let f = 0; f < frames; f++) {
    frame(false)
    if (f % 12 === 11) samples.push({ t: +(time - popAt).toFixed(2), ...(await read()) })
  }
  // Where the remaining motion is: the fastest vertices, their grid cell and what they belong to.
  const V = new Float32Array(await renderer.getArrayBufferAsync(sim.v.value, null, 0, n * 16))
  const S = new Float32Array(await renderer.getArrayBufferAsync(sim.state.value, null, 0, sheet.count * 16))
  const order = [...Array(n).keys()].sort(
    (a, b) => Math.hypot(V[b * 4], V[b * 4 + 1], V[b * 4 + 2]) - Math.hypot(V[a * 4], V[a * 4 + 1], V[a * 4 + 2]),
  )
  const hot = order.slice(0, 8).map((k) => {
    const inside = sheet.meta[k * 4 + 1] > 0.5
    const b = sheet.meta[k * 4]
    return `v${k} (${k % sheet.W},${Math.floor(k / sheet.W)} of ${sheet.W}x${sheet.H}) |v|=${Math.hypot(V[k * 4], V[k * 4 + 1], V[k * 4 + 2]).toFixed(2)} vy=${V[k * 4 + 1].toFixed(2)} ${inside ? `cap of ${b}${b === target ? ' (pressed)' : ''} gauge ${S[b * 4].toFixed(3)}` : 'flat film'}`
  })
  const last = samples[samples.length - 1]
  const peak = Math.max(...samples.map((s) => s.energy))
  const extra = Math.max(...samples.map((s) => s.popped)) - 1
  if (extra > 0) failures.push(`${extra} other bubbles popped after a single press`)
  if (last.energy > Math.max(peak * 0.05, 1e-3)) {
    failures.push(
      `kinetic energy didn't decay: ${last.energy.toExponential(2)} at the end vs peak ${peak.toExponential(2)}`,
    )
  }
  if (last.height > rest.height + sheet.domeHeight * 0.5) {
    failures.push(`sheet no longer flat: height ${last.height.toFixed(2)} cm vs ${rest.height.toFixed(2)} at rest`)
  }
  if (last.spread > sheet.size)
    failures.push(`sheet drifted/rolled up: a vertex moved ${last.spread.toFixed(2)} cm in plane`)
  return { name: sc.name, bubbles: sheet.count, popAt, pressFor, samples, failures, hot }
}

const DESKTOP = { view: { width: 1280, height: 800 }, phone: false }
const IPHONE = { view: { width: 390, height: 844 }, phone: true }
const DEFAULT_SHEET = { across: 8, size: 1.1, spacing: 0.28, tiling: 'square', shape: 'circle', margin: 0.4 } as const

export const SCENARIOS: Scenario[] = [
  { name: 'desktop defaults, center', ...DESKTOP, ...DEFAULT_SHEET, target: 'center' },
  { name: 'iPhone portrait defaults, center', ...IPHONE, ...DEFAULT_SHEET, target: 'center' },
  {
    name: 'large hexagons 4 across, center',
    view: { width: 1920, height: 1080 },
    phone: false,
    ...DEFAULT_SHEET,
    across: 4,
    size: 2.2,
    spacing: 0.5,
    tiling: 'triangular',
    shape: 'hexagon',
    target: 'center',
  },
  {
    name: 'small bubbles 14 across, center',
    view: { width: 1920, height: 1080 },
    phone: false,
    ...DEFAULT_SHEET,
    across: 14,
    size: 0.6,
    spacing: 0.2,
    target: 'center',
  },
  // Edge and corner bubbles, with the original tight edge (margin 0) and the default margin.
  ...[DESKTOP, IPHONE].flatMap((device) =>
    (['edge', 'corner'] as const).flatMap((target) =>
      [0, 0.4].map((margin): Scenario => ({
        name: `${device.phone ? 'iPhone portrait' : 'desktop'} ${target}, margin ${margin}`,
        ...device,
        ...DEFAULT_SHEET,
        margin,
        target,
      })),
    ),
  ),
]

async function main() {
  const renderer = new THREE.WebGPURenderer()
  await renderer.init()
  const sim = new BubbleSim()
  const generator = new LatticeGenerator(sim.mapRaw)
  const only = query.get('only')
  const results: StabilityResult[] = []
  for (const sc of SCENARIOS) {
    if (only && !sc.name.includes(only)) continue
    results.push(await runScenario(renderer, sim, generator, sc))
  }
  return results
}

const w = window as unknown as { __stability?: StabilityResult[] | { error: string } }
main().then(
  (r) => (w.__stability = r),
  (e: unknown) => (w.__stability = { error: String((e as Error)?.stack ?? e) }),
)
