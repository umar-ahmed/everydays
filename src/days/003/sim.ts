import {
  Fn,
  If,
  abs,
  clamp,
  float,
  floor,
  fract,
  instanceIndex,
  instancedArray,
  int,
  length,
  max,
  min,
  mix,
  mx_noise_float,
  pow,
  select,
  sin,
  smoothstep,
  sqrt,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl'
import type { ComputeNode, Node, StorageBufferNode, WebGPURenderer } from 'three/webgpu'

/** Fastest water (cells per unit time), and deepest cut per step (cells). */
const MAX_SPEED = 6
const MAX_CUT = 0.05

/** Most octaves the generator sums; the Octaves dial gates the rest off. */
const MAX_OCTAVES = 10

export type TerrainParams = {
  seed: number
  octaves: number
  /** Amplitude kept per octave (persistence). */
  roughness: number
  /** 0 smooth fBm hills, 1 ridged mountains. */
  ridge: number
  /** Peak height in world units (the terrain is SIZE wide). */
  heightScale: number
}

export type ErosionParams = {
  rain: number
  /** Sediment carrying capacity per unit of speed and slope. */
  capacity: number
  erosion: number
  deposition: number
  evaporation: number
  /** Thermal weathering: share of the excess over the talus angle moved per step. */
  thermal: number
}

/** World size of the terrain square. */
export const SIZE = 10

/** Cells per side: phones get a quarter of the cells. */
export const gridFor = (compact: boolean) => (compact ? 256 : 512)

type I = Node<'int'>
type F = Node<'float'>

/**
 * Grid-based hydraulic erosion on the GPU, after the virtual-pipes model of Mei, Decaudin and
 * Hu, "Fast Hydraulic Erosion Simulation and Visualization on GPU" (2007), plus a thermal
 * (talus) pass. Everything runs in grid units: cells are 1 apart and heights are in cells.
 *
 * Per step: rain + outflow flux through virtual pipes to the four neighbours, water depth and
 * velocity from the net flux, erosion or deposition toward the flow's carrying capacity,
 * semi-Lagrangian sediment transport with evaporation, then thermal weathering.
 */
export class ErosionSim {
  readonly n: number
  /** Cell size in world units. */
  readonly cell: number
  /** Per cell: terrain height, water depth, suspended sediment, initial height (cells). */
  readonly state: StorageBufferNode<'vec4'>
  readonly state2: StorageBufferNode<'vec4'>
  /** Outflow flux to the left, right, top and bottom neighbours. */
  readonly flux: StorageBufferNode<'vec4'>
  /** Water velocity (x, z) in cells per unit time, and the mean water depth. */
  readonly velocity: StorageBufferNode<'vec4'>

  readonly seed = uniform(1)
  readonly octaves = uniform(7)
  readonly roughness = uniform(0.5)
  readonly ridge = uniform(0.5)
  readonly heightCells = uniform(100)

  readonly rain = uniform(0.012)
  readonly capacity = uniform(1)
  readonly erosion = uniform(0.05)
  readonly deposition = uniform(0.05)
  readonly evaporation = uniform(0.015)
  readonly thermal = uniform(0.2)
  readonly dt = uniform(0.02)
  readonly step = uniform(0)

  private readonly generate: ComputeNode
  private readonly kernels: ComputeNode[]

  constructor(n: number) {
    this.n = n
    this.cell = SIZE / (n - 1)
    const count = n * n
    this.state = instancedArray(count, 'vec4')
    this.state2 = instancedArray(count, 'vec4')
    this.flux = instancedArray(count, 'vec4')
    this.velocity = instancedArray(count, 'vec4')
    const { state, state2, flux, velocity } = this
    const N = int(n)

    const xy = (i: I) => [i.mod(N), i.div(N)] as const
    const iclamp = (v: I) => select(v.lessThan(0), int(0), select(v.greaterThan(N.sub(1)), N.sub(1), v))
    const idx = (x: I, y: I) => iclamp(y).mul(N).add(iclamp(x))
    const hash = (p: Node<'vec2'>) => fract(sin(p.x.mul(127.1).add(p.y.mul(311.7))).mul(43758.5453))

    // Terrain: fBm and ridged noise mixed by the Ridge dial, sharpened, with the edges lowered
    // so rivers drain off the map instead of pooling at the border.
    this.generate = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const uv = vec2(float(x), float(y)).div(float(n - 1))
      const offset = vec3(this.seed.mul(17.13), this.seed.mul(-9.71), this.seed.mul(3.37))
      // A low-frequency warp breaks up the grid-aligned look of plain noise.
      const p0 = vec3(uv.mul(3), 0).add(offset)
      const warp = vec3(mx_noise_float(p0.mul(0.7).add(11.3)), mx_noise_float(p0.mul(0.7).add(27.1)), 0).mul(0.35)
      const p = p0.add(warp)
      const fbm = float(0).toVar()
      const ridged = float(0).toVar()
      const norm = float(0).toVar()
      const amp = float(1).toVar()
      const freq = float(1).toVar()
      const weight = float(1).toVar()
      for (let k = 0; k < MAX_OCTAVES; k++) {
        If(float(k).lessThan(this.octaves), () => {
          const v = mx_noise_float(p.mul(freq).add(vec3(k * 7.31, k * 3.17, 0)))
          fbm.addAssign(v.mul(amp))
          // Ridges: folded noise, each octave weighted by the one below (Musgrave).
          const r = float(1).sub(abs(v)).toVar()
          r.assign(r.mul(r).mul(weight))
          weight.assign(clamp(r.mul(2), 0, 1))
          ridged.addAssign(r.mul(amp))
          norm.addAssign(amp)
          amp.mulAssign(this.roughness)
          freq.mulAssign(2.03)
        })
      }
      const hills = fbm.div(norm).mul(0.9).add(0.5)
      const mountains = ridged.div(norm).mul(1.25)
      const h = clamp(mix(hills, mountains, this.ridge), 0, 1.5)
      const edge = max(abs(uv.x.sub(0.5)), abs(uv.y.sub(0.5))).mul(2)
      const shaped = pow(h, float(1.35)).mul(float(1).sub(smoothstep(0.55, 1, edge).mul(0.85)))
      const height = shaped.mul(this.heightCells)
      state.element(i).assign(vec4(height, 0, 0, height))
      flux.element(i).assign(vec4(0))
      velocity.element(i).assign(vec4(0))
    })().compute(count)

    const g = 9.81
    // 1. Rain, then outflow flux through the four virtual pipes, scaled so a cell never sends
    //    out more water than it holds.
    const fluxPass = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const s = state.element(i)
      const jitter = hash(vec2(float(x).add(this.step.mul(0.618)), float(y).add(this.step.mul(1.37))))
      const d = s.y.add(this.rain.mul(this.dt).mul(jitter.mul(1.6).add(0.2)))
      const level = s.x.add(d)
      const out = (jx: I, jy: I, f: F) => {
        const nb = state.element(idx(jx, jy))
        // Off the map is a drain: the edge of the world is lower than anything on it.
        const edge = jx.lessThan(0).or(jx.greaterThanEqual(N)).or(jy.lessThan(0)).or(jy.greaterThanEqual(N))
        const other = select(edge, float(-1e3), nb.x.add(nb.y))
        return max(f.add(this.dt.mul(g).mul(level.sub(other))), 0)
      }
      const f0 = flux.element(i)
      const fl = out(x.sub(1), y, f0.x)
      const fr = out(x.add(1), y, f0.y)
      const ft = out(x, y.sub(1), f0.z)
      const fb = out(x, y.add(1), f0.w)
      const total = fl.add(fr).add(ft).add(fb)
      const k = min(float(1), d.div(max(total.mul(this.dt), 1e-8)))
      flux.element(i).assign(vec4(fl, fr, ft, fb).mul(k))
      state.element(i).assign(vec4(s.x, d, s.z, s.w))
    })().compute(count)

    // 2. Water depth from the net flux, and velocity from the flux through the cell.
    const waterPass = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const s = state.element(i)
      const f = flux.element(i)
      const inside = (jx: I, jy: I) =>
        jx.greaterThanEqual(0).and(jx.lessThan(N)).and(jy.greaterThanEqual(0)).and(jy.lessThan(N))
      const fromL = select(inside(x.sub(1), y), flux.element(idx(x.sub(1), y)).y, float(0))
      const fromR = select(inside(x.add(1), y), flux.element(idx(x.add(1), y)).x, float(0))
      const fromT = select(inside(x, y.sub(1)), flux.element(idx(x, y.sub(1))).w, float(0))
      const fromB = select(inside(x, y.add(1)), flux.element(idx(x, y.add(1))).z, float(0))
      const inflow = fromL.add(fromR).add(fromT).add(fromB)
      const outflow = f.x.add(f.y).add(f.z).add(f.w)
      const d2 = max(s.y.add(this.dt.mul(inflow.sub(outflow))), 0)
      const mean = s.y.add(d2).mul(0.5)
      const wx = fromL.sub(f.x).add(f.y).sub(fromR).mul(0.5)
      const wz = fromT.sub(f.z).add(f.w).sub(fromB).mul(0.5)
      // Capped: a thin film racing down a cliff would otherwise carve without limit.
      const raw = vec2(wx, wz).div(max(mean, 0.01))
      const v = raw.mul(min(float(1), float(MAX_SPEED).div(max(length(raw), 1e-6))))
      velocity.element(i).assign(vec4(v, mean, 0))
      state.element(i).assign(vec4(s.x, d2, s.z, s.w))
    })().compute(count)

    // 3. Erode or deposit toward the carrying capacity of the flow.
    const erodePass = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const s = state.element(i)
      const hl = state.element(idx(x.sub(1), y)).x
      const hr = state.element(idx(x.add(1), y)).x
      const ht = state.element(idx(x, y.sub(1))).x
      const hb = state.element(idx(x, y.add(1))).x
      const grad = vec2(hr.sub(hl), hb.sub(ht)).mul(0.5)
      // Sine of the local tilt, floored so flat beds still carry a little.
      const tilt = max(length(grad).div(sqrt(grad.dot(grad).add(1))), 0.05)
      const v = velocity.element(i)
      // Shallow sheet flow carries less than a channel: capacity ramps in with depth.
      const depthLimit = clamp(v.z.mul(4), 0, 1)
      const capacity = this.capacity.mul(tilt).mul(length(v.xy)).mul(depthLimit)
      const take = this.erosion.mul(capacity.sub(s.z)).mul(this.dt)
      const drop = this.deposition.mul(s.z.sub(capacity)).mul(this.dt)
      const change = select(capacity.greaterThan(s.z), take.negate(), drop)
      // A little per step at most, so the bed follows the flow instead of outrunning it.
      const dh = clamp(change, float(-MAX_CUT), s.z)
      state2.element(i).assign(vec4(s.x.add(dh), s.y, s.z.sub(dh), s.w))
    })().compute(count)

    // 4. Carry sediment along the flow (semi-Lagrangian), and evaporate.
    const transportPass = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const s = state2.element(i)
      const v = velocity.element(i).xy
      const back = vec2(float(x), float(y)).sub(v.mul(this.dt))
      const bx = clamp(back.x, 0, float(n - 1))
      const by = clamp(back.y, 0, float(n - 1))
      const x0 = int(floor(bx))
      const y0 = int(floor(by))
      const tx = fract(bx)
      const ty = fract(by)
      const sed = (dx: number, dy: number) => state2.element(idx(x0.add(dx), y0.add(dy))).z
      const carried = mix(mix(sed(0, 0), sed(1, 0), tx), mix(sed(0, 1), sed(1, 1), tx), ty)
      const water = s.y.mul(max(float(1).sub(this.evaporation.mul(this.dt)), 0))
      state.element(i).assign(vec4(s.x, water, carried, s.w))
    })().compute(count)

    // 5. Thermal weathering: material slides to lower neighbours where the slope exceeds the
    //    talus angle. Each pair moves the same amount both ways, so mass is conserved.
    const talus = Math.tan((55 * Math.PI) / 180)
    const thermalPass = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const s = state.element(i)
      const net = float(0).toVar()
      for (const [dx, dy] of [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
      ]) {
        const h = state.element(idx(x.add(dx), y.add(dy))).x
        const diff = s.x.sub(h)
        // Out when this cell is higher by more than the talus slope, in when it's lower.
        const slide = max(abs(diff).sub(talus), 0).mul(this.thermal).mul(0.125)
        net.addAssign(select(diff.greaterThan(0), slide.negate(), slide))
      }
      state2.element(i).assign(vec4(s.x.add(net), s.y, s.z, s.w))
    })().compute(count)

    // Thermal writes state2; copy back so every step starts from state.
    const copyBack = Fn(() => {
      const i = int(instanceIndex)
      state.element(i).assign(state2.element(i))
    })().compute(count)

    this.kernels = [fluxPass, waterPass, erodePass, transportPass, thermalPass, copyBack]
  }

  setTerrain(p: TerrainParams) {
    this.seed.value = p.seed
    this.octaves.value = p.octaves
    this.roughness.value = p.roughness
    this.ridge.value = p.ridge
    this.heightCells.value = p.heightScale / this.cell
  }

  setErosion(p: ErosionParams) {
    this.rain.value = p.rain
    this.capacity.value = p.capacity
    this.erosion.value = p.erosion
    this.deposition.value = p.deposition
    this.evaporation.value = p.evaporation
    this.thermal.value = p.thermal
  }

  /** Fresh terrain from the noise, dry, with nothing eroded. */
  reset(renderer: WebGPURenderer) {
    void renderer.compute(this.generate)
  }

  /** Run `iterations` erosion steps in one submission. */
  run(renderer: WebGPURenderer, iterations: number) {
    const list: ComputeNode[] = []
    for (let k = 0; k < iterations; k++) list.push(...this.kernels)
    // One step counter per submission is enough to vary the rain pattern between frames.
    this.step.value = (this.step.value + 1) % 100000
    void renderer.compute(list)
  }
}
