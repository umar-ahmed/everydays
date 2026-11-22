import {
  Fn,
  If,
  Loop,
  abs,
  clamp,
  cross,
  float,
  instanceIndex,
  instancedArray,
  int,
  length,
  max,
  mix,
  normalize,
  pow,
  select,
  uniform,
  vec3,
  vec4,
} from 'three/tsl'
import type { ComputeNode, WebGPURenderer } from 'three/webgpu'
import { CELL, MAX_BUBBLES, MAX_VERTS, type Sheet } from './sheet.ts'

/**
 * Bubble wrap as an XPBD cloth on the GPU.
 *
 * Every substep runs four kernels:
 *   bubbles   per-bubble enclosed volume -> adiabatic gas pressure; latch a pop past the burst threshold
 *   predict   gravity + pressure force (gauge pressure x vertex area along the normal), explicit velocity
 *   solve     one Jacobi XPBD pass over stretch/shear/bend distance constraints, gathered per vertex
 *   finalize  finger/roller and table collisions, velocity from positions
 * One substep with one iteration follows "Small Steps in Physics Simulation" (Macklin et al. 2019),
 * so XPBD's lambda starts at zero every substep and needs no per-constraint storage.
 */

const GAMMA = 1.4
const NEIGHBOURS: [number, number, 'stretch' | 'bend'][] = [
  [1, 0, 'stretch'],
  [-1, 0, 'stretch'],
  [0, 1, 'stretch'],
  [0, -1, 'stretch'],
  [1, 1, 'stretch'],
  [1, -1, 'stretch'],
  [-1, 1, 'stretch'],
  [-1, -1, 'stretch'],
  [2, 0, 'bend'],
  [-2, 0, 'bend'],
  [0, 2, 'bend'],
  [0, -2, 'bend'],
]

export type SimParams = {
  fill: number
  threshold: number
  stiffness: number
  substeps: number
  pressRadius: number
  tool: 'finger' | 'roller'
}

export class BubbleSim {
  readonly x = instancedArray(MAX_VERTS, 'vec4')
  readonly v = instancedArray(MAX_VERTS, 'vec4')
  readonly p = instancedArray(MAX_VERTS, 'vec4')
  readonly q = instancedArray(MAX_VERTS, 'vec4')
  readonly normal = instancedArray(MAX_VERTS, 'vec4')
  readonly rest = instancedArray(MAX_VERTS, 'vec4')
  /** Rest-pose normals (w = 0: no compression), for the Debug > rest pose view. */
  readonly restNormal = instancedArray(MAX_VERTS, 'vec4')
  readonly meta = instancedArray(MAX_VERTS, 'vec4')
  /** Pop events, one texel per bubble: popped, gauge pressure at burst, burst time, rest volume. */
  readonly pops = instancedArray(MAX_BUBBLES, 'vec4')
  /** Live state per bubble: gauge pressure, volume, gas content (atm), pinhole flag. */
  readonly state = instancedArray(MAX_BUBBLES, 'vec4')

  readonly uW = uniform(0, 'int')
  readonly uH = uniform(0, 'int')
  readonly uCols = uniform(0, 'int')
  readonly uH2 = uniform(1)
  readonly dt = uniform(1 / 600)
  readonly time = uniform(0)
  readonly gravity = uniform(60)
  /** Pressure acceleration per atm of gauge pressure, for a vertex of rest area. */
  readonly kPressure = uniform(1200)
  readonly fill = uniform(0.12)
  readonly threshold = uniform(0.6)
  readonly damping = uniform(2)
  readonly aStretch = uniform(1e-7)
  readonly aCompress = uniform(1e-4)
  readonly aBendBase = uniform(2e-6)
  readonly aBendDome = uniform(1e-4)
  readonly aBendPopped = uniform(5e-3)
  readonly finger = uniform(vec3(0, 100, 0))
  readonly fingerRadius = uniform(0.5)
  readonly roller = uniform(0)
  readonly rollerHalf = uniform(3)
  readonly friction = uniform(0.6)
  readonly omega = uniform(1.9)
  /** Pinhole venting, atm of gas per second per atm of excess pressure. */
  readonly leakRate = uniform(80)

  private readonly kernels: ComputeNode[]
  private readonly normals: ComputeNode
  private readonly solveQP: ComputeNode
  /** Jacobi passes per substep; odd, so the last pass lands in q for finalize. */
  iterations = 7

  constructor() {
    const { x, v, p, q, rest, meta, pops, state, uW, uH, uCols } = this

    const bubbles = Fn(() => {
      const b = int(instanceIndex)
      const cx = b.mod(uCols)
      const cy = b.div(uCols)
      const volume = float(0).toVar()
      Loop(CELL * CELL, ({ i }) => {
        const ix = cx.mul(CELL).add(i.mod(CELL))
        const iy = cy.mul(CELL).add(i.div(CELL))
        const i0 = iy.mul(uW).add(ix)
        const p0 = x.element(i0).xyz
        const p1 = x.element(i0.add(1)).xyz
        const p2 = x.element(i0.add(uW).add(1)).xyz
        const p3 = x.element(i0.add(uW)).xyz
        const a = p2.sub(p0)
        const c = p3.sub(p1)
        const area = a.x.mul(c.z).sub(a.z.mul(c.x)).mul(0.5)
        volume.addAssign(area.mul(p0.y.add(p1.y).add(p2.y).add(p3.y)).mul(0.25))
      })
      const pop = pops.element(b)
      const v0 = pop.w
      const st = state.element(b)
      const pinhole = st.w
      // Gas content in atm at rest volume. Sealed bubbles hold the fill; pinholed ones leak.
      const gas = select(pinhole.greaterThan(0.5), st.z, this.fill.add(1))
      // Adiabatic squeeze: p V^gamma stays constant, in atmospheres; gauge = absolute - 1.
      const squeeze = v0.div(max(volume, v0.mul(0.03)))
      // A vented bubble can sit slightly below atmospheric; the clamp keeps it from sucking hard.
      const gauge = max(gas.mul(pow(squeeze, GAMMA)).sub(1), -0.15)
      If(pop.x.lessThan(0.5).and(gauge.greaterThan(this.threshold)), () => {
        pop.assign(vec4(1, gauge, this.time, v0))
      })
      const intact = pop.x.lessThan(0.5)
      // A pinhole only vents once squeezed past the fill pressure, so it hisses flat when pressed.
      const leak = select(pinhole.greaterThan(0.5).and(intact), max(gauge.sub(this.fill.mul(1.2)), 0).mul(this.leakRate), float(0))
      // Venting continues until the inside is at atmospheric for the current volume.
      const left = max(gas.sub(leak.mul(this.dt)), pow(squeeze, -GAMMA))
      st.assign(vec4(select(intact, gauge, float(0)), volume, left, pinhole))
    })().compute(MAX_BUBBLES)

    const around = (i: ReturnType<typeof int>, buf: typeof x = x) => {
      const ix = i.mod(uW)
      const iy = i.div(uW)
      const il = select(ix.greaterThan(0), i.sub(1), i)
      const ir = select(ix.lessThan(uW.sub(1)), i.add(1), i)
      const iu = select(iy.greaterThan(0), i.sub(uW), i)
      const id = select(iy.lessThan(uH.sub(1)), i.add(uW), i)
      // Points up for a flat sheet; length is roughly the vertex's area.
      return cross(buf.element(id).xyz.sub(buf.element(iu).xyz), buf.element(ir).xyz.sub(buf.element(il).xyz)).mul(0.25)
    }

    const predict = Fn(() => {
      const i = int(instanceIndex)
      const info = meta.element(i)
      const gauge = state.element(int(info.x)).x
      const push = around(i).mul(gauge.mul(this.kPressure).mul(info.y).div(this.uH2))
      const acc = vec3(0, this.gravity.negate(), 0).add(push)
      const vel = v.element(i).xyz.add(acc.mul(this.dt)).mul(float(1).sub(this.damping.mul(this.dt)))
      v.element(i).assign(vec4(vel, 0))
      p.element(i).assign(vec4(x.element(i).xyz.add(vel.mul(this.dt)), 1))
    })().compute(MAX_VERTS)

    const makeSolve = (src: typeof p, dst: typeof p) => Fn(() => {
      const i = int(instanceIndex)
      const ix = i.mod(uW)
      const iy = i.div(uW)
      const pi = src.element(i).xyz
      const ri = rest.element(i)
      const info = meta.element(i)
      const popped = pops.element(int(info.x)).x
      const aBend = select(
        info.y.lessThan(0.5),
        this.aBendBase,
        select(popped.greaterThan(0.5), this.aBendPopped, this.aBendDome),
      )
      const dt2 = this.dt.mul(this.dt)
      const sum = vec3(0).toVar()
      const count = float(0).toVar()
      for (const [dx, dy, kind] of NEIGHBOURS) {
        const jx = ix.add(dx)
        const jy = iy.add(dy)
        const ok = jx.greaterThanEqual(0).and(jx.lessThan(uW)).and(jy.greaterThanEqual(0)).and(jy.lessThan(uH))
        If(ok, () => {
          const j = jy.mul(uW).add(jx)
          const d = src.element(j).xyz.sub(pi)
          const dist = max(length(d), 1e-6)
          const c = dist.sub(length(rest.element(j).xyz.sub(ri.xyz)))
          // Film resists stretching hard and compression barely, so an unpressurised dome buckles.
          const alpha = kind === 'bend' ? aBend : select(c.greaterThan(0), this.aStretch, this.aCompress)
          sum.addAssign(d.div(dist).mul(c.div(float(2).add(alpha.div(dt2)))))
          count.addAssign(1)
        })
      }
      // Jacobi: average this vertex's corrections, over-relaxed (Macklin et al. 2014).
      dst.element(i).assign(vec4(pi.add(sum.mul(this.omega.div(max(count, 1)))), 1))
    })().compute(MAX_VERTS)
    const solvePQ = makeSolve(p, q)
    const solveQP = makeSolve(q, p)

    const finalize = Fn(() => {
      const i = int(instanceIndex)
      const prev = x.element(i).xyz
      const pos = q.element(i).xyz.toVar()
      const c = this.finger
      const r = this.fingerRadius
      If(this.roller.lessThan(0.5), () => {
        const d = pos.sub(c)
        const dist = length(d)
        If(dist.lessThan(r), () => {
          pos.assign(c.add(d.mul(r.div(max(dist, 1e-5)))))
        })
      }).Else(() => {
        const d = vec3(0, pos.y.sub(c.y), pos.z.sub(c.z))
        const dist = length(d)
        If(dist.lessThan(r).and(abs(pos.x.sub(c.x)).lessThan(this.rollerHalf)), () => {
          pos.assign(vec3(pos.x, c.yz.add(d.yz.mul(r.div(max(dist, 1e-5))))))
        })
      })
      If(pos.y.lessThan(0), () => {
        pos.assign(vec3(mix(pos.x, prev.x, this.friction), 0, mix(pos.z, prev.z, this.friction)))
      })
      v.element(i).assign(vec4(pos.sub(prev).div(this.dt), 0))
      x.element(i).assign(vec4(pos, 1))
    })().compute(MAX_VERTS)

    this.normals = Fn(() => {
      const i = int(instanceIndex)
      const n = around(i)
      // w: local compression (lost area vs rest), read only by the wrinkle shader.
      const squeeze = clamp(float(1).sub(length(n).div(max(length(around(i, rest)), 1e-8))), 0, 1)
      this.normal.element(i).assign(vec4(normalize(n), squeeze))
    })().compute(MAX_VERTS)

    this.kernels = [bubbles, predict, solvePQ, finalize]
    this.solveQP = solveQP
  }

  load(sheet: Sheet, fill: number) {
    const n = sheet.W * sheet.H
    const write = (node: { value: { array: ArrayLike<number> & { set(a: ArrayLike<number>): void; fill(v: number): void }; needsUpdate: boolean } }, data?: Float32Array) => {
      const arr = node.value.array
      arr.fill(0)
      if (data) arr.set(data)
      node.value.needsUpdate = true
    }
    const positions = new Float32Array(n * 4)
    for (let k = 0; k < n; k++) {
      positions[k * 4] = sheet.rest[k * 4]
      positions[k * 4 + 1] = sheet.rest[k * 4 + 1]
      positions[k * 4 + 2] = sheet.rest[k * 4 + 2]
      positions[k * 4 + 3] = 1
    }
    write(this.x, positions)
    write(this.p, positions)
    write(this.q, positions)
    write(this.v)
    write(this.normal)
    write(this.rest, sheet.rest)
    write(this.restNormal, restNormals(sheet))
    write(this.meta, sheet.meta)
    write(this.pops, sheet.bubbles)
    const state = new Float32Array(sheet.cols * sheet.rows * 4)
    for (let b = 0; b < sheet.cols * sheet.rows; b++) {
      state[b * 4 + 2] = 1 + fill
      state[b * 4 + 3] = sheet.pinholes[b]
    }
    write(this.state, state)
    this.uW.value = sheet.W
    this.uH.value = sheet.H
    this.uCols.value = sheet.cols
    // Dispatch only the live sheet; three bounds-checks each kernel against its count.
    this.kernels[0].count = sheet.cols * sheet.rows
    for (const k of [...this.kernels.slice(1), this.solveQP, this.normals]) k.count = n
    this.uH2.value = sheet.h * sheet.h
  }

  setParams(params: SimParams, sheet: Sheet) {
    this.fill.value = params.fill
    this.threshold.value = Math.max(params.threshold, params.fill + 0.1)
    // Film stiffness 0..1 maps to stretch compliance 1e-4 .. 1e-9 (log scale).
    this.aStretch.value = 10 ** (-4 - 5 * params.stiffness)
    this.aCompress.value = this.aStretch.value * 2000
    this.fingerRadius.value = params.pressRadius * sheet.size
    this.roller.value = params.tool === 'roller' ? 1 : 0
    this.rollerHalf.value = sheet.size * 1.6
  }

  step(renderer: WebGPURenderer, frameDt: number, substeps: number) {
    const n = Math.max(1, Math.round(substeps))
    this.dt.value = frameDt / n
    const list: ComputeNode[] = []
    const [bubbles, predict, solvePQ, finalize] = this.kernels
    const passes = Math.max(1, this.iterations | 1)
    for (let s = 0; s < n; s++) {
      list.push(bubbles, predict, solvePQ)
      for (let k = 1; k < passes; k += 2) list.push(this.solveQP, solvePQ)
      list.push(finalize)
    }
    list.push(this.normals)
    renderer.compute(list)
  }
}

/** CPU twin of the normals kernel, run on the rest positions. */
function restNormals({ rest, W, H }: Sheet) {
  const out = new Float32Array(W * H * 4)
  const at = (ix: number, iy: number, c: number) => rest[(iy * W + ix) * 4 + c]
  for (let iy = 0; iy < H; iy++) {
    for (let ix = 0; ix < W; ix++) {
      const l = Math.max(ix - 1, 0)
      const r = Math.min(ix + 1, W - 1)
      const u = Math.max(iy - 1, 0)
      const d = Math.min(iy + 1, H - 1)
      const ax = at(ix, d, 0) - at(ix, u, 0)
      const ay = at(ix, d, 1) - at(ix, u, 1)
      const az = at(ix, d, 2) - at(ix, u, 2)
      const bx = at(r, iy, 0) - at(l, iy, 0)
      const by = at(r, iy, 1) - at(l, iy, 1)
      const bz = at(r, iy, 2) - at(l, iy, 2)
      const nx = ay * bz - az * by
      const ny = az * bx - ax * bz
      const nz = ax * by - ay * bx
      const len = Math.hypot(nx, ny, nz) || 1
      const k = (iy * W + ix) * 4
      out[k] = nx / len
      out[k + 1] = ny / len
      out[k + 2] = nz / len
    }
  }
  return out
}
