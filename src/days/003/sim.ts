import {
  Fn,
  If,
  abs,
  clamp,
  exp,
  float,
  floor,
  fract,
  instanceIndex,
  instancedArray,
  int,
  uint,
  length,
  max,
  min,
  mix,
  mx_noise_float,
  packUnorm4x8,
  pow,
  select,
  sin,
  smoothstep,
  sqrt,
  tan,
  unpackUnorm4x8,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl'
import * as THREE from 'three/webgpu'
import type { ComputeNode, Node, StorageBufferNode, WebGPURenderer } from 'three/webgpu'

/**
 * Fastest water (cells per unit time) and deepest cut per step (cells) on the 512 grid. Heights
 * are in cells, so at other resolutions these and the rain scale with the cell count, and the
 * same world-space terrain erodes alike, just at finer or coarser detail. The carrying capacity
 * stays put: the pipe flow's speed already grows with the water depth in cells.
 */
const MAX_SPEED = 6
const MAX_CUT = 0.05
const BASE_GRID = 512
/** Deepest a channel cuts below its banks (cells, 512 grid) before it widens instead. */
const MAX_INCISION = 8

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
  /** 0 steady even rain, 1 drifting storms that leave some regions dry. */
  rainVariation: number
  /** Sediment carrying capacity per unit of speed and slope. */
  capacity: number
  erosion: number
  deposition: number
  evaporation: number
  /** Thermal weathering: share of the excess over the talus angle moved per step. */
  thermal: number
  /** Spread of the talus angle across the map (0 uniform, 1 about ±18°). */
  talusVariation: number
  /** Spread of rock hardness: 0 uniform, 1 bands and blocks up to ~10x harder than others. */
  hardness: number
  /** Thickness of one hard + soft strata pair, world units. */
  strataScale: number
  /** Dip of the strata, degrees. */
  strataTilt: number
  /** Strength of the weak fault/joint lines. */
  joints: number
  /** Thickness of the mobile top layer whose colour moves with the sediment (cells, 512 grid). */
  layer: number
}

/** World size of the terrain square. */
export const SIZE = 10

/** Terrain resolution choices (cells per side). Phones default to 256, desktops to 512. */
export const RESOLUTIONS = [128, 256, 512, 1024, 2048, 4096] as const
export const defaultResolution = (compact: boolean) => (compact ? 256 : 512)

/** GPU buffers the sim allocates: n * n vec4s of f32 each, plus four of packed rgba8 colour. */
export const SIM_BUFFERS = 6
export const COLOR_BUFFERS = 4
export const bufferBytes = (n: number) => n * n * 16
export const simBytes = (n: number) => SIM_BUFFERS * bufferBytes(n) + COLOR_BUFFERS * n * n * 4

/** Largest resolution up to `requested` whose buffers fit the device's limits. */
export function fitResolution(
  requested: number,
  limits: { maxBufferSize: number; maxStorageBufferBindingSize: number },
) {
  const cap = Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize)
  const fits = RESOLUTIONS.filter((r) => r <= requested && bufferBytes(r) <= cap)
  return fits.at(-1) ?? RESOLUTIONS[0]
}

type I = Node<'int'>
type F = Node<'float'>
type V4 = Node<'vec4'>
type U = Node<'uint'>

const RAD = Math.PI / 180

/**
 * Grid-based hydraulic erosion on the GPU, after the virtual-pipes model of Mei, Decaudin and
 * Hu, "Fast Hydraulic Erosion Simulation and Visualization on GPU" (2007), extended to eight
 * pipes, plus a thermal (talus) pass. Everything runs in grid units: cells are 1 apart and
 * heights are in cells.
 *
 * Per step: rain + outflow flux through virtual pipes to the eight neighbours, water depth and
 * velocity from the net flux, erosion (divided by the rock's hardness) or deposition toward the
 * flow's carrying capacity, semi-Lagrangian sediment transport with evaporation, then thermal
 * weathering against a talus angle that varies with the rock.
 *
 * The rock is layered: tilted hard and soft strata (looked up at the current surface height, so
 * cutting down through them makes benches and steps), blocky 3D noise, and weak joint lines.
 *
 * Colour moves with the material. Each cell has a mobile top layer (up to `layer` cells thick)
 * whose material is tracked as (relief it came from, stratum 0 soft..1 hard, rockiness), with the
 * layer's cover 0..1 in the fourth channel; below it is bedrock, whose material is wherever the
 * surface is now. Eroded material carries its source's mix into the suspended sediment, which is
 * advected with the water and blended into the layer where it settles; thermal slumping carries
 * it downhill the same way. The renderer maps materials through the palette, so presets still
 * swap live. Materials are packed rgba8 with stochastic rounding, so small mixing steps add up
 * instead of rounding away.
 */
export class ErosionSim {
  readonly n: number
  /** Cell size in world units. */
  readonly cell: number
  /** Cells per world unit relative to the 512 grid. */
  readonly scale: number
  /** Per cell: terrain height, water depth, suspended sediment, initial height (cells). */
  readonly state: StorageBufferNode<'vec4'>
  readonly state2: StorageBufferNode<'vec4'>
  /** Outflow flux to the left, right, top and bottom neighbours. */
  readonly flux: StorageBufferNode<'vec4'>
  /** Outflow flux to the top-left, top-right, bottom-left and bottom-right neighbours. */
  readonly flux2: StorageBufferNode<'vec4'>
  /** Water velocity (x, z) in cells per unit time, and the mean water depth. */
  readonly velocity: StorageBufferNode<'vec4'>
  /** Per cell: rock noise, joint line (0..1), talus noise, current rain factor. */
  readonly rock: StorageBufferNode<'vec4'>
  /** Top-layer material and cover, packed rgba8 (see above); thermal writes `surface2`. */
  readonly surface: StorageBufferNode<'uint'>
  readonly surface2: StorageBufferNode<'uint'>
  /** Suspended sediment's material: erosion writes `sedColor`, transport `sedColor2`. */
  readonly sedColor: StorageBufferNode<'uint'>
  readonly sedColor2: StorageBufferNode<'uint'>

  readonly seed = uniform(1)
  readonly octaves = uniform(7)
  readonly roughness = uniform(0.5)
  readonly ridge = uniform(0.5)
  readonly heightCells = uniform(100)

  readonly rain = uniform(0.012)
  readonly rainVariation = uniform(0.6)
  readonly capacity = uniform(1)
  readonly erosion = uniform(0.05)
  readonly deposition = uniform(0.05)
  readonly evaporation = uniform(0.015)
  readonly thermal = uniform(0.2)
  readonly talusVariation = uniform(0.6)
  readonly hardness = uniform(0.6)
  /** Strata phase per cell of height, and the dip as a slope along `strataDir`. */
  readonly strataK = uniform(1)
  readonly strataTan = uniform(0.2)
  readonly strataDir = uniform(new THREE.Vector2(1, 0))
  readonly joints = uniform(0.4)
  /** Top-layer thickness in cells. */
  readonly layer = uniform(0.5)
  readonly dt = uniform(0.02)
  /** Erosion steps run so far (wraps), for the drifting weather. */
  readonly step = uniform(0)

  private readonly generate: ComputeNode
  private readonly paint: ComputeNode
  private readonly weather: ComputeNode
  private readonly kernels: ComputeNode[]

  constructor(n: number) {
    this.n = n
    this.cell = SIZE / (n - 1)
    this.scale = (n - 1) / (BASE_GRID - 1)
    const scale = this.scale
    const count = n * n
    // Every buffer starts zeroed and is filled on the GPU, so they can share one upload array
    // (at 4096 each is 256 MiB).
    const zeros = new Float32Array(count * 4)
    const buffer = () => instancedArray(zeros, 'vec4') as StorageBufferNode<'vec4'>
    this.state = buffer()
    this.state2 = buffer()
    this.flux = buffer()
    this.flux2 = buffer()
    this.velocity = buffer()
    this.rock = buffer()
    const words = new Uint32Array(zeros.buffer, 0, count)
    const packed = () => instancedArray(words, 'uint') as StorageBufferNode<'uint'>
    this.surface = packed()
    this.surface2 = packed()
    this.sedColor = packed()
    this.sedColor2 = packed()
    const { state, state2, flux, flux2, velocity, rock, surface, surface2, sedColor, sedColor2 } = this
    const N = int(n)

    const xy = (i: I) => [i.mod(N), i.div(N)] as const
    const iclamp = (v: I) => select(v.lessThan(0), int(0), select(v.greaterThan(N.sub(1)), N.sub(1), v))
    const idx = (x: I, y: I) => iclamp(y).mul(N).add(iclamp(x))
    const hash = (p: Node<'vec2'>) => fract(sin(p.x.mul(127.1).add(p.y.mul(311.7))).mul(43758.5453))
    const inside = (jx: I, jy: I) =>
      jx.greaterThanEqual(0).and(jx.lessThan(N)).and(jy.greaterThanEqual(0)).and(jy.lessThan(N))
    const offset = vec3(this.seed.mul(17.13), this.seed.mul(-9.71), this.seed.mul(3.37))
    /** Per-cell, per-step dither in [0, 1) (salted per kernel) for stochastic rounding. */
    const dither = (x: I, y: I, salt: number) =>
      hash(
        vec2(
          float(x).add(this.step.mul(0.618)).add(salt),
          float(y)
            .add(this.step.mul(1.37))
            .sub(salt * 0.7),
        ),
      )
    /** Quantize to rgba8 rounding up or down at random in proportion, so the mean is exact. */
    const pack = (v: V4, d: F) =>
      packUnorm4x8(
        min(
          floor(
            clamp(v, 0, 1)
              .mul(255)
              .add(vec4(d, fract(d.add(0.38)), fract(d.add(0.62)), fract(d.add(0.81)))),
          ),
          255,
        ).div(255),
      )
    const unpack = (u: U) => unpackUnorm4x8(u) as unknown as V4
    /** Bedrock's material where the surface is now: its relief, stratum, and rockiness. */
    const bedrock = (h: F, band: F, rocky: F) =>
      vec3(clamp(h.div(this.heightCells), 0, 1), band.mul(0.5).add(0.5), rocky)

    // Terrain: fBm and ridged noise mixed by the Ridge dial, sharpened, with the edges lowered
    // so rivers drain off the map instead of pooling at the border. Also lays down the rock.
    this.generate = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const uv = vec2(float(x), float(y)).div(float(n - 1))
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

      // Rock: blocky noise through the volume (sampled at the starting surface), two families
      // of broken joint lines, and the talus angle's own noise.
      const q = vec3(uv.mul(7), shaped.mul(2.5)).add(offset.mul(1.7))
      const blocks = clamp(
        mx_noise_float(q)
          .mul(1.3)
          .add(mx_noise_float(q.mul(2.7).add(5.1)).mul(0.5)),
        -1,
        1,
      )
      const angle = this.seed.mul(2.399)
      const d1 = vec2(sin(angle), sin(angle.add(Math.PI / 2)))
      const d2 = vec2(sin(angle.add(1.15)), sin(angle.add(1.15 + Math.PI / 2)))
      const bend = mx_noise_float(vec3(uv.mul(2.5), 3.3).add(offset)).mul(0.9)
      const line = (d: Node<'vec2'>, f: number, salt: number) => {
        const c = uv.dot(d).mul(f).add(bend)
        const near = float(1).sub(smoothstep(0.015, 0.06, abs(fract(c).sub(0.5))))
        // Joints come and go along their length.
        return near.mul(smoothstep(-0.05, 0.25, mx_noise_float(vec3(uv.mul(3.1), salt).add(offset))))
      }
      const joint = max(line(d1, 6, 7.7), line(d2, 4.5, 9.1))
      const talusNoise = clamp(
        mx_noise_float(vec3(uv.mul(5), 1.7).add(offset))
          .mul(1.2)
          .add(mx_noise_float(vec3(uv.mul(19), 4.2).add(offset)).mul(0.6)),
        -1,
        1,
      )
      state.element(i).assign(vec4(height, 0, 0, height))
      state2.element(i).assign(vec4(height, 0, 0, height))
      flux.element(i).assign(vec4(0))
      flux2.element(i).assign(vec4(0))
      velocity.element(i).assign(vec4(0))
      rock.element(i).assign(vec4(blocks, joint, talusNoise, 1))
    })().compute(count)

    // A fresh map has no loose top layer and nothing in suspension. (A separate kernel: one
    // shader may only bind eight storage buffers by default.)
    this.paint = Fn(() => {
      const i = int(instanceIndex)
      surface.element(i).assign(uint(0))
      surface2.element(i).assign(uint(0))
      sedColor.element(i).assign(uint(0))
      sedColor2.element(i).assign(uint(0))
    })().compute(count)

    // Weather, once per batch of steps: storm cells drift across the map and change shape, so
    // some regions get soaked while others stay dry for a while.
    this.weather = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const uv = vec2(float(x), float(y)).div(float(n - 1))
      const t = this.step
      const drift = vec2(0.7, 0.35).mul(t.mul(4e-5))
      const storms = mx_noise_float(vec3(uv.mul(2.2).add(drift), t.mul(1e-4)).add(offset))
        .mul(1.5)
        .add(mx_noise_float(vec3(uv.mul(6.5).sub(drift.mul(1.7)), t.mul(2.5e-4)).add(offset.mul(0.7))).mul(0.6))
      const r = rock.element(i)
      rock.element(i).assign(vec4(r.xyz, max(float(1).add(this.rainVariation.mul(storms)), 0)))
    })().compute(count)

    const g = 9.81
    const r2 = Math.SQRT1_2
    // Diagonal pipes are √2 long and get half the cross-section, which makes the 3x3 stencil
    // close to isotropic, so water no longer prefers the grid axes.
    const DIAG = 0.5 * r2
    // 1. Rain, then outflow flux through the eight virtual pipes, scaled so a cell never sends
    //    out more water than it holds.
    const fluxPass = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const s = state.element(i)
      const jitter = hash(vec2(float(x).add(this.step.mul(0.618)), float(y).add(this.step.mul(1.37))))
      const d = s.y.add(this.rain.mul(this.dt).mul(jitter.add(0.5)).mul(rock.element(i).w))
      const level = s.x.add(d)
      const out = (dx: number, dy: number, f: F, w: number) => {
        const jx = x.add(dx)
        const jy = y.add(dy)
        const nb = state.element(idx(jx, jy))
        // Off the map is a drain: the edge of the world is lower than anything on it.
        const other = select(inside(jx, jy), nb.x.add(nb.y), float(-1e3))
        return max(f.add(this.dt.mul(g * w).mul(level.sub(other))), 0)
      }
      const f0 = flux.element(i)
      const g0 = flux2.element(i)
      const axis = vec4(out(-1, 0, f0.x, 1), out(1, 0, f0.y, 1), out(0, -1, f0.z, 1), out(0, 1, f0.w, 1))
      const diag = vec4(out(-1, -1, g0.x, DIAG), out(1, -1, g0.y, DIAG), out(-1, 1, g0.z, DIAG), out(1, 1, g0.w, DIAG))
      const total = axis.x.add(axis.y).add(axis.z).add(axis.w).add(diag.x).add(diag.y).add(diag.z).add(diag.w)
      const k = min(float(1), d.div(max(total.mul(this.dt), 1e-8)))
      flux.element(i).assign(axis.mul(k))
      flux2.element(i).assign(diag.mul(k))
      state.element(i).assign(vec4(s.x, d, s.z, s.w))
    })().compute(count)

    // 2. Water depth from the net flux, and velocity from the flux through the cell.
    const waterPass = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const s = state.element(i)
      const f = flux.element(i)
      const fd = flux2.element(i)
      const from = (dx: number, dy: number, pick: (v: V4) => F, diag: boolean) => {
        const jx = x.add(dx)
        const jy = y.add(dy)
        return select(inside(jx, jy), pick((diag ? flux2 : flux).element(idx(jx, jy))), float(0))
      }
      // Net outflow toward each neighbour: what this cell sends minus what that one sends back.
      const nL = f.x.sub(from(-1, 0, (v) => v.y, false))
      const nR = f.y.sub(from(1, 0, (v) => v.x, false))
      const nT = f.z.sub(from(0, -1, (v) => v.w, false))
      const nB = f.w.sub(from(0, 1, (v) => v.z, false))
      const nTL = fd.x.sub(from(-1, -1, (v) => v.w, true))
      const nTR = fd.y.sub(from(1, -1, (v) => v.z, true))
      const nBL = fd.z.sub(from(-1, 1, (v) => v.y, true))
      const nBR = fd.w.sub(from(1, 1, (v) => v.x, true))
      const net = nL.add(nR).add(nT).add(nB).add(nTL).add(nTR).add(nBL).add(nBR)
      const d2 = max(s.y.sub(this.dt.mul(net)), 0)
      const mean = s.y.add(d2).mul(0.5)
      const wx = nR.sub(nL).add(nTR.add(nBR).sub(nTL).sub(nBL).mul(r2)).mul(0.5)
      const wz = nB.sub(nT).add(nBL.add(nBR).sub(nTL).sub(nTR).mul(r2)).mul(0.5)
      // Capped: a thin film racing down a cliff would otherwise carve without limit.
      const raw = vec2(wx, wz).div(max(mean, 0.01))
      const v = raw.mul(min(float(1), float(MAX_SPEED * scale).div(max(length(raw), 1e-6))))
      velocity.element(i).assign(vec4(v, mean, 0))
      state.element(i).assign(vec4(s.x, d2, s.z, s.w))
    })().compute(count)

    // Talus angle (as a slope): bedrock stands steep, varying with noise and hardness; fresh
    // deposits (above where the ground started) are scree at its angle of repose.
    const talusOf = (r: V4, logHardness: F, height: F, initial: F) => {
      const rockAngle = float(57 * RAD)
        .add(this.talusVariation.mul(r.z).mul(12 * RAD))
        .add(clamp(logHardness, -1, 1).mul(6 * RAD))
      const loose = smoothstep(0.05 * scale, 0.6 * scale, height.sub(initial))
      return tan(mix(rockAngle, float(34 * RAD), loose))
    }

    // 3. Erode or deposit toward the carrying capacity of the flow. Hard rock gives up its
    //    material slowly, and a channel that is already deep for its banks stops cutting down
    //    and widens instead, so long runs don't end as knife-edge fins between slot gullies.
    const erodePass = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const s = state.element(i)
      const h = (dx: number, dy: number) => state.element(idx(x.add(dx), y.add(dy))).x
      const hl = h(-1, 0)
      const hr = h(1, 0)
      const ht = h(0, -1)
      const hb = h(0, 1)
      const htl = h(-1, -1)
      const htr = h(1, -1)
      const hbl = h(-1, 1)
      const hbr = h(1, 1)
      // Sobel gradient: rotation-invariant enough that diagonal slopes read true.
      const grad = vec2(
        hr.sub(hl).mul(2).add(htr.sub(htl)).add(hbr.sub(hbl)),
        hb.sub(ht).mul(2).add(hbl.sub(htl)).add(hbr.sub(htr)),
      ).div(8)
      // Sine of the local tilt, floored so flat beds still carry a little.
      const tilt = max(length(grad).div(sqrt(grad.dot(grad).add(1))), 0.05)
      const v = velocity.element(i)
      // Shallow sheet flow carries less than a channel: capacity ramps in with depth.
      const depthLimit = clamp(v.z.mul(4 / scale), 0, 1)
      const capacity = this.capacity.mul(tilt).mul(length(v.xy)).mul(depthLimit)
      const banks = hl.add(hr).add(ht).add(hb).mul(0.15).add(htl.add(htr).add(hbl).add(hbr).mul(0.1))
      const incised = clamp(float(1).sub(banks.sub(s.x).div(MAX_INCISION * scale)), 0, 1)
      const r = rock.element(i)
      const band = this.strataBand(float(x), float(y), s.x, r)
      const logHardness = this.hardnessFrom(band, r)
      const hardness = exp(logHardness)
      const take = this.erosion.mul(capacity.sub(s.z)).mul(this.dt).mul(incised).div(hardness)
      const drop = this.deposition.mul(s.z.sub(capacity)).mul(this.dt)
      const change = select(capacity.greaterThan(s.z), take.negate(), drop)
      // A little per step at most, so the bed follows the flow instead of outrunning it.
      const dh = clamp(change, float(-MAX_CUT * scale), s.z)
      state2.element(i).assign(vec4(s.x.add(dh), s.y, s.z.sub(dh), s.w))
      // Each cell's talus angle, for the thermal pass to read for itself and its neighbours.
      velocity.element(i).assign(vec4(v.xyz, talusOf(r, logHardness, s.x, s.w)))

      // Colour: what's cut comes off the loose layer first, then the bedrock under it, and
      // mixes into the suspended load by mass; what settles mixes into the layer by mass.
      const top = this.layer
      const layer = unpack(surface.element(i))
      const load = unpack(sedColor2.element(i)).xyz
      const layerMass = layer.w.mul(top)
      const cut = max(dh.negate(), 0)
      const settled = max(dh, 0)
      const fromLayer = min(cut, layerMass)
      const rocky = smoothstep(0.51, 0.8, tilt)
      const picked = layer.xyz.mul(fromLayer).add(bedrock(s.x, band, rocky).mul(cut.sub(fromLayer)))
      const loadMass = s.z.add(cut)
      const newLoad = select(loadMass.greaterThan(1e-6), load.mul(s.z).add(picked).div(max(loadMass, 1e-6)), load)
      const kept = layerMass.sub(fromLayer)
      const mass = kept.add(settled)
      const newLayer = select(
        mass.greaterThan(1e-6),
        layer.xyz.mul(kept).add(load.mul(settled)).div(max(mass, 1e-6)),
        layer.xyz,
      )
      surface.element(i).assign(pack(vec4(newLayer, min(mass.div(top), 1)), dither(x, y, 0.13)))
      sedColor.element(i).assign(pack(vec4(newLoad, 0), dither(x, y, 0.29)))
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
      // The load's material comes along with it, weighted by how much each corner carries.
      const tint = vec3(0).toVar()
      const weight = float(0).toVar()
      for (const [dx, dy, w] of [
        [0, 0, float(1).sub(tx).mul(float(1).sub(ty))],
        [1, 0, tx.mul(float(1).sub(ty))],
        [0, 1, float(1).sub(tx).mul(ty)],
        [1, 1, tx.mul(ty)],
      ] as const) {
        const m = w.mul(sed(dx, dy))
        tint.addAssign(unpack(sedColor.element(idx(x0.add(dx), y0.add(dy)))).xyz.mul(m))
        weight.addAssign(m)
      }
      const own = unpack(sedColor.element(i)).xyz
      const moved = select(weight.greaterThan(1e-6), tint.div(max(weight, 1e-6)), own)
      sedColor2.element(i).assign(pack(vec4(moved, 0), dither(x, y, 0.47)))
    })().compute(count)

    // 5. Thermal weathering: material slides to lower neighbours (all eight) where the slope
    //    exceeds the talus angle. Each pair uses the higher cell's angle and moves the same
    //    amount both ways, so mass is conserved. (No diffusive soil creep: the noise's finest
    //    octaves are about a cell tall, and creep strong enough to touch fins melts them too.)
    const thermalPass = Fn(() => {
      const i = int(instanceIndex)
      const [x, y] = xy(i)
      const s = state.element(i)
      const own = velocity.element(i).w
      const net = float(0).toVar()
      // Colour: what slides in brings the source's loose layer first, then its bedrock (a
      // slide only starts on ground steeper than talus, so that bedrock counts as rock).
      const top = this.layer
      const layer = unpack(surface.element(i))
      const gone = float(0).toVar()
      const came = float(0).toVar()
      const brought = vec3(0).toVar()
      for (const [dx, dy] of [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1],
      ]) {
        const dist = Math.hypot(dx, dy)
        const j = idx(x.add(dx), y.add(dy))
        const nb = state.element(j)
        const diff = s.x.sub(nb.x)
        // The higher cell's material decides: a cliff stands at its rock's angle whatever lies
        // below it.
        const talus = select(diff.greaterThan(0), own, velocity.element(j).w).mul(dist)
        // Out when this cell is higher by more than the talus slope, in when it's lower.
        const slide = max(abs(diff).sub(talus), 0)
          .mul(this.thermal)
          .mul(0.1 / dist)
        net.addAssign(select(diff.greaterThan(0), slide.negate(), slide))
        const incoming = select(diff.greaterThan(0), float(0), slide)
        gone.addAssign(select(diff.greaterThan(0), slide, float(0)))
        const source = unpack(surface.element(j))
        const jx = float(x.add(dx))
        const jy = float(y.add(dy))
        const rockThere = bedrock(nb.x, this.strataBand(jx, jy, nb.x, rock.element(j)), float(1))
        const fromLayer = clamp(source.w.mul(top).div(max(incoming, 1e-6)), 0, 1)
        brought.addAssign(mix(rockThere, source.xyz, fromLayer).mul(incoming))
        came.addAssign(incoming)
      }
      state2.element(i).assign(vec4(s.x.add(net), s.y, s.z, s.w))
      const kept = max(layer.w.mul(top).sub(gone), 0)
      const mass = kept.add(came)
      const mixed = select(mass.greaterThan(1e-6), layer.xyz.mul(kept).add(brought).div(max(mass, 1e-6)), layer.xyz)
      surface2.element(i).assign(pack(vec4(mixed, min(mass.div(top), 1)), dither(x, y, 0.71)))
    })().compute(count)

    // Thermal writes state2; copy back so every step starts from state.
    const copyBack = Fn(() => {
      const i = int(instanceIndex)
      state.element(i).assign(state2.element(i))
      surface.element(i).assign(surface2.element(i))
    })().compute(count)

    this.kernels = [fluxPass, waterPass, erodePass, transportPass, thermalPass, copyBack]
  }

  /**
   * Log hardness of the rock at cell (x, y) and height h (cells), from its `rock` entry: tilted
   * strata bands (hard and soft, undulating with the rock noise), the noise itself, and joint
   * lines that are weaker than anything around them. Erosion divides by exp() of this.
   */
  hardnessLog(x: F, y: F, h: F, r: V4) {
    return this.hardnessFrom(this.strataBand(x, y, h, r), r)
  }

  /** `hardnessLog` from an already looked-up strata band. */
  hardnessFrom(band: F, r: V4) {
    return band.mul(0.9).add(r.x.mul(0.6)).sub(r.y.mul(this.joints).mul(1.6)).mul(this.hardness).mul(1.2)
  }

  /** -1 soft to 1 hard: which stratum the point at height h (cells) sits in. */
  strataBand(x: F, y: F, h: F, r: V4) {
    const phase = h.add(vec2(x, y).dot(this.strataDir).mul(this.strataTan)).mul(this.strataK).add(r.x.mul(1.2))
    return smoothstep(-0.35, 0.35, sin(phase)).mul(2).sub(1)
  }

  setTerrain(p: TerrainParams) {
    this.seed.value = p.seed
    this.octaves.value = p.octaves
    this.roughness.value = p.roughness
    this.ridge.value = p.ridge
    this.heightCells.value = p.heightScale / this.cell
    const azimuth = p.seed * 1.618
    this.strataDir.value.set(Math.cos(azimuth), Math.sin(azimuth))
  }

  setErosion(p: ErosionParams) {
    this.rain.value = p.rain * this.scale
    this.rainVariation.value = p.rainVariation
    this.capacity.value = p.capacity
    this.erosion.value = p.erosion
    this.deposition.value = p.deposition
    this.evaporation.value = p.evaporation
    this.thermal.value = p.thermal
    this.talusVariation.value = p.talusVariation
    this.hardness.value = p.hardness
    this.strataK.value = (2 * Math.PI * this.cell) / Math.max(p.strataScale, 1e-3)
    this.strataTan.value = Math.tan(p.strataTilt * RAD)
    this.joints.value = p.joints
    this.layer.value = p.layer * this.scale
  }

  /** Fresh terrain from the noise, dry, with nothing eroded. */
  reset(renderer: WebGPURenderer) {
    void renderer.compute([this.generate, this.paint])
  }

  /** Run `iterations` erosion steps in one submission. */
  run(renderer: WebGPURenderer, iterations: number) {
    const list: ComputeNode[] = [this.weather]
    for (let k = 0; k < iterations; k++) list.push(...this.kernels)
    void renderer.compute(list)
    this.step.value = (this.step.value + iterations) % 1e6
  }

  /** Free the GPU buffers and compute pipelines. */
  dispose() {
    for (const node of [this.generate, this.paint, this.weather, ...this.kernels]) node.dispose()
    for (const b of [this.state, this.state2, this.flux, this.flux2, this.velocity, this.rock]) b.value.dispose()
    for (const b of [this.surface, this.surface2, this.sedColor, this.sedColor2]) b.value.dispose()
  }
}
