import {
  Fn,
  If,
  Loop,
  abs,
  acos,
  clamp,
  cross,
  float,
  instanceIndex,
  instancedArray,
  int,
  length,
  max,
  min,
  mix,
  normalize,
  pow,
  select,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl'
import type { ComputeNode, WebGPURenderer } from 'three/webgpu'
import { MAX_BUBBLES, MAX_MAP, MAX_QUADS, MAX_VERTS, type Sheet } from './sheet.ts'

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

/**
 * Fixed sim ticks from variable frame times. The solver (a fixed count of over-relaxed Jacobi
 * passes per substep) is tuned for one substep length; stepping the raw frame time made 120 Hz
 * displays and slow motion run shorter substeps, where solver residue turns into velocity
 * (residue / dt) and domes never settle. Every tick is exactly `dt`; slow motion runs fewer ticks.
 */
export class FixedStep {
  readonly dt = 1 / 60
  /** One tick per frame at most: below 60 fps the sim slows down rather than doubling GPU work. */
  readonly maxTicks = 1
  private acc = 0

  /** How many ticks to run for a frame of `frameDt` seconds at `speed` (1 = real time). */
  ticks(frameDt: number, speed = 1) {
    this.acc += Math.min(frameDt, 0.1) * speed
    const n = Math.min(Math.floor(this.acc / this.dt + 1e-6), this.maxTicks)
    this.acc = Math.min(this.acc - n * this.dt, this.dt)
    return n
  }

  reset() {
    this.acc = 0
  }
}

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
  /** Generator output and the canonical bubble map (see generators/index.ts for the format). */
  readonly mapRaw = instancedArray(MAX_MAP, 'vec4')
  readonly map = instancedArray(MAX_MAP, 'vec4')
  /** Per bubble: center x, center z, inradius, equivalent diameter. */
  readonly centers = instancedArray(MAX_BUBBLES, 'vec4')
  readonly mapW = uniform(1, 'int')
  readonly mapH = uniform(1, 'int')
  readonly texel = uniform(0.05)
  readonly half = uniform(vec2(1, 1))
  /** Per bubble: (first, count) into quadList, the quads its volume is summed over. */
  readonly quadRange = instancedArray(MAX_BUBBLES, 'vec2')
  readonly quadList = instancedArray(MAX_QUADS, 'float')
  /** Live state per bubble: gauge pressure, volume, gas content (atm), pinhole flag. */
  readonly state = instancedArray(MAX_BUBBLES, 'vec4')
  /** Per bubble: height of its seal ring this substep (the back film's level under the cap). */
  readonly base = instancedArray(MAX_BUBBLES, 'vec4')
  /**
   * Debug channels, written only while `debug` is 1. diag: stretch strain (signed, mean over the
   * axial edges), bend deviation (rad), |v| (cm/s), total solver correction in the last substep
   * (cm). corr: that correction split into stretch, bend, pressure push and table/finger clamps
   * (cm). contact: clamped onto the back film (0/1), penetration depth before the clamp (cm).
   */
  readonly diag = instancedArray(MAX_VERTS, 'vec4')
  readonly corr = instancedArray(MAX_VERTS, 'vec4')
  readonly contact = instancedArray(MAX_VERTS, 'vec4')
  readonly debug = uniform(0)

  readonly uW = uniform(0, 'int')
  readonly uH = uniform(0, 'int')
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
  /** Cap vertices stay at least this far (cm) above the back film. */
  readonly capGap = uniform(0.004)
  /**
   * The flat back film the domes are sealed onto, for rendering only: bonded vertices (outside
   * every footprint) copy the cloth, and a harmonic fill spans each footprint at its seal-ring
   * height, kept under the cap. The sim must not read it: the seal ring is part of the cloth the
   * cap pulls on, so a floor or volume base derived from it closes a loop (cap lifts ring, ring
   * lifts floor, floor shoves cap) that a free sheet edge amplifies into a runaway. The sheet
   * lies on the table, which carries the back film, so caps rest on and gas is measured from the
   * table plane.
   */
  readonly back = instancedArray(MAX_VERTS, 'vec4')
  readonly back2 = instancedArray(MAX_VERTS, 'vec4')
  /**
   * Jacobi relaxation. Over-relaxing (any omega > 1) pumps energy into the film: stretch is
   * near-rigid and compression very soft, so a constraint near its rest length overshoots
   * back and forth across the switch every pass. Plain averaging dissipates.
   */
  readonly omega = uniform(1)
  /** Pinhole venting, atm of gas per second per atm of excess pressure. */
  readonly leakRate = uniform(80)

  private readonly kernels: ComputeNode[]
  private readonly normals: ComputeNode
  private readonly backFill: ComputeNode[]
  /** Harmonic-fill passes per frame (even); warm-started from the last frame. */
  backIterations = 16
  private readonly solveQP: ComputeNode
  /** Jacobi passes per substep; odd, so the last pass lands in q for finalize. */
  iterations = 9

  constructor() {
    const { x, v, p, q, rest, meta, pops, state, uW, uH, quadRange, quadList } = this

    const bubbles = Fn(() => {
      const b = int(instanceIndex)
      const range = quadRange.element(b)
      const first = int(range.x)
      const volume = float(0).toVar()
      const ring = float(0).toVar()
      const ringCount = float(0).toVar()
      // Each bubble sums its own quads (any footprint shape or layout), as listed by the sheet.
      Loop({ start: int(0), end: int(range.y), type: 'int', condition: '<' }, ({ i }) => {
        const quad = int(quadList.element(first.add(i)))
        const ix = quad.mod(uW.sub(1))
        const iy = quad.div(uW.sub(1))
        const i0 = iy.mul(uW).add(ix)
        const p0 = x.element(i0).xyz
        const p1 = x.element(i0.add(1)).xyz
        const p2 = x.element(i0.add(uW).add(1)).xyz
        const p3 = x.element(i0.add(uW)).xyz
        const a = p2.sub(p0)
        const c = p3.sub(p1)
        const qa = a.x.mul(c.z).sub(a.z.mul(c.x)).mul(0.5)
        volume.addAssign(qa.mul(p0.y.add(p1.y).add(p2.y).add(p3.y)).mul(0.25))
        // Corners outside the footprint are the seal ring the back film is bonded to.
        for (const [k, pk] of [
          [i0, p0],
          [i0.add(1), p1],
          [i0.add(uW).add(1), p2],
          [i0.add(uW), p3],
        ] as const) {
          const sealed = meta.element(k).y.lessThan(0.5)
          ring.addAssign(select(sealed, pk.y, float(0)))
          ringCount.addAssign(select(sealed, float(1), float(0)))
        }
      })
      // Seal-ring height, for drawing the back film; the gas is measured from the table.
      this.base.element(b).assign(vec4(ring.div(max(ringCount, 1)), 0, 0, 0))
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
      const leak = select(
        pinhole.greaterThan(0.5).and(intact),
        max(gauge.sub(this.fill.mul(1.2)), 0).mul(this.leakRate),
        float(0),
      )
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
      const vel = v
        .element(i)
        .xyz.add(acc.mul(this.dt))
        .mul(float(1).sub(this.damping.mul(this.dt)))
      v.element(i).assign(vec4(vel, 0))
      p.element(i).assign(vec4(x.element(i).xyz.add(vel.mul(this.dt)), 1))
      If(this.debug.greaterThan(0.5), () => {
        this.corr.element(i).assign(vec4(0, 0, length(push).mul(this.dt.mul(this.dt)), 0))
      })
    })().compute(MAX_VERTS)

    const makeSolve = (src: typeof p, dst: typeof p) =>
      Fn(() => {
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
        const sumStretch = vec3(0).toVar()
        const sumBend = vec3(0).toVar()
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
            ;(kind === 'bend' ? sumBend : sumStretch).addAssign(d.div(dist).mul(c.div(float(2).add(alpha.div(dt2)))))
            count.addAssign(1)
          })
        }
        // Jacobi: average this vertex's corrections (Macklin et al. 2014).
        const scale = this.omega.div(max(count, 1))
        dst.element(i).assign(vec4(pi.add(sumStretch.add(sumBend).mul(scale)), 1))
        If(this.debug.greaterThan(0.5), () => {
          const k = this.corr.element(i)
          k.assign(vec4(k.x.add(length(sumStretch).mul(scale)), k.y.add(length(sumBend).mul(scale)), k.z, k.w))
        })
      })().compute(MAX_VERTS)
    const solvePQ = makeSolve(p, q)
    const solveQP = makeSolve(q, p)

    const finalize = Fn(() => {
      const i = int(instanceIndex)
      const prev = x.element(i).xyz
      const pos = q.element(i).xyz.toVar()
      const solved = q.element(i).xyz
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
      const clamped = pos.toVar()
      // A collapsed cap rests on the back film, which lies on the table.
      const onCap = meta.element(i).y.greaterThan(0.5)
      const depth = select(onCap, max(this.capGap.sub(pos.y), 0), float(0))
      If(onCap.and(pos.y.lessThan(this.capGap)), () => {
        pos.assign(vec3(pos.x, this.capGap, pos.z))
      })
      If(this.debug.greaterThan(0.5), () => {
        // v still holds the predicted velocity here, so prev + v dt is the predicted position.
        const predicted = prev.add(v.element(i).xyz.mul(this.dt))
        const k = this.corr.element(i)
        k.assign(vec4(k.xyz, length(clamped.sub(solved))))
        this.contact.element(i).assign(vec4(select(depth.greaterThan(0), float(1), float(0)), depth, 0, 0))
        const dg = this.diag.element(i)
        dg.assign(vec4(dg.xyz, length(solved.sub(predicted))))
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
      If(this.debug.greaterThan(0.5), () => {
        const ix = i.mod(uW)
        const iy = i.div(uW)
        const pi = x.element(i).xyz
        const ri = rest.element(i).xyz
        const strain = float(0).toVar()
        const edges = float(0).toVar()
        for (const [dx, dy] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ]) {
          const jx = ix.add(dx)
          const jy = iy.add(dy)
          If(jx.greaterThanEqual(0).and(jx.lessThan(uW)).and(jy.greaterThanEqual(0)).and(jy.lessThan(uH)), () => {
            const j = jy.mul(uW).add(jx)
            const restLen = max(length(rest.element(j).xyz.sub(ri)), 1e-6)
            strain.addAssign(length(x.element(j).xyz.sub(pi)).div(restLen).sub(1))
            edges.addAssign(1)
          })
        }
        // Bend: change of the angle between the two edges through this vertex, along x and z.
        const bend = float(0).toVar()
        for (const [dx, dy] of [
          [1, 0],
          [0, 1],
        ]) {
          const ok = ix
            .sub(dx)
            .greaterThanEqual(0)
            .and(ix.add(dx).lessThan(uW))
            .and(iy.sub(dy).greaterThanEqual(0))
            .and(iy.add(dy).lessThan(uH))
          If(ok, () => {
            const a = dy ? i.sub(uW) : i.sub(1)
            const b = dy ? i.add(uW) : i.add(1)
            const angle = (buf: typeof x) => {
              const e1 = normalize(buf.element(i).xyz.sub(buf.element(a).xyz))
              const e2 = normalize(buf.element(b).xyz.sub(buf.element(i).xyz))
              return acos(clamp(e1.dot(e2), -1, 1))
            }
            bend.assign(max(bend, abs(angle(x).sub(angle(rest)))))
          })
        }
        const dg = this.diag.element(i)
        dg.assign(vec4(strain.div(max(edges, 1)), bend, length(v.element(i).xyz), dg.w))
      })
    })().compute(MAX_VERTS)

    const fill = (src: typeof x, dst: typeof x) =>
      Fn(() => {
        const i = int(instanceIndex)
        const ix = i.mod(uW)
        const iy = i.div(uW)
        const l = src.element(select(ix.greaterThan(0), i.sub(1), i)).xyz
        const r = src.element(select(ix.lessThan(uW.sub(1)), i.add(1), i)).xyz
        const u = src.element(select(iy.greaterThan(0), i.sub(uW), i)).xyz
        const d = src.element(select(iy.lessThan(uH.sub(1)), i.add(uW), i)).xyz
        const info = meta.element(i)
        const spread = l.add(r).add(u).add(d).mul(0.25)
        // Flat at the seal ring, but never above the cap it lies under.
        const flat = vec3(spread.x, min(this.base.element(int(info.x)).x, x.element(i).y.sub(this.capGap)), spread.z)
        dst.element(i).assign(vec4(select(info.y.greaterThan(0.5), flat, x.element(i).xyz), 0))
      })().compute(MAX_VERTS)
    this.backFill = [fill(this.back, this.back2), fill(this.back2, this.back)]

    this.kernels = [bubbles, predict, solvePQ, finalize]
    this.solveQP = solveQP
  }

  load(sheet: Sheet, fill: number) {
    const n = sheet.W * sheet.H
    const write = (
      node: {
        value: {
          array: ArrayLike<number> & { set(a: ArrayLike<number>): void; fill(v: number): void }
          needsUpdate: boolean
        }
      },
      data?: Float32Array,
    ) => {
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
    // The back film starts flat on the table under the whole sheet.
    const flat = new Float32Array(n * 4)
    for (let k = 0; k < n; k++) {
      flat[k * 4] = sheet.rest[k * 4]
      flat[k * 4 + 2] = sheet.rest[k * 4 + 2]
    }
    write(this.back, flat)
    write(this.back2, flat)
    write(this.normal)
    write(this.rest, sheet.rest)
    write(this.restNormal, sheet.restNormal)
    write(this.meta, sheet.meta)
    write(this.pops, sheet.bubbles)
    write(this.map, sheet.map)
    write(this.centers, sheet.centers)
    this.mapW.value = sheet.mapW
    this.mapH.value = sheet.mapH
    this.texel.value = sheet.texel
    this.half.value.set(sheet.width / 2, sheet.depth / 2)
    write(this.quadRange, sheet.quadRange)
    write(this.quadList, sheet.quadList)
    const state = new Float32Array(sheet.count * 4)
    for (let b = 0; b < sheet.count; b++) {
      state[b * 4 + 2] = 1 + fill
      state[b * 4 + 3] = sheet.pinholes[b]
    }
    write(this.state, state)
    write(this.base)
    write(this.diag)
    write(this.corr)
    write(this.contact)
    this.uW.value = sheet.W
    this.uH.value = sheet.H
    // Dispatch only the live sheet; three bounds-checks each kernel against its count.
    this.kernels[0].count = Math.max(sheet.count, 1)
    for (const k of [...this.kernels.slice(1), this.solveQP, this.normals, ...this.backFill]) k.count = n
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
    for (let k = 0; k < this.backIterations; k += 2) list.push(...this.backFill)
    void renderer.compute(list)
  }
}
