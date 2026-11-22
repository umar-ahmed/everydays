import {
  Fn,
  clamp,
  cross,
  float,
  floor,
  fract,
  instanceIndex,
  instancedArray,
  int,
  max,
  min,
  mix,
  normalize,
  select,
  smoothstep,
  sqrt,
  uniform,
  vec3,
  vec4,
} from 'three/tsl'
import type { ComputeNode, Node, WebGPURenderer } from 'three/webgpu'
import { BACK_OFFSET } from './camera.ts'
import { MAP_SCALE, type Sheet } from './sheet.ts'
import type { BubbleSim } from './sim.ts'

type I = Node<'int'>
const imin = (a: I, b: I) => select(a.lessThan(b), a, b)
const iclamp = (a: I, lo: I, hi: I) => select(a.lessThan(lo), lo, imin(a, hi))

import { MAX_RENDER_VERTS, renderSize } from './layout.ts'

export { DETAILS, MAX_DETAIL, MAX_RENDER_VERTS, renderSize } from './layout.ts'

/**
 * A render surface denser than the cloth sim. Its rest shape is sampled from the bubble map's
 * signed distance at render resolution (with a soft foot on the rim, which the sim itself can't
 * have: it stops non-circular bubbles from popping), and the sim's displacement from its own rest
 * pose is added on top through a Catmull-Rom (C1) interpolation, so domes keep round silhouettes.
 */
export class RenderSurface {
  /** xyz position, w compression (wrinkle input). */
  readonly pos = instancedArray(MAX_RENDER_VERTS, 'vec4')
  /** xyz normal, w compression. */
  readonly normal = instancedArray(MAX_RENDER_VERTS, 'vec4')
  /** Rest x, fine rest height, rest z, domeness 0..1. */
  readonly rest = instancedArray(MAX_RENDER_VERTS, 'vec4')
  /** Bubble id (-1 for flat film), inside 0/1, signed distance (cm), unused. */
  readonly info = instancedArray(MAX_RENDER_VERTS, 'vec4')
  /** The flat back film at render resolution (drawn only across footprints and a seal band). */
  readonly backPos = instancedArray(MAX_RENDER_VERTS, 'vec4')
  readonly backNormal = instancedArray(MAX_RENDER_VERTS, 'vec4')

  readonly detail = uniform(2, 'int')
  readonly RW = uniform(1, 'int')
  readonly RH = uniform(1, 'int')
  readonly h = uniform(0.1)
  readonly foot = uniform(0.1)
  /** 1 renders the rest shape with no sim displacement (Debug > Simulation > rest pose). */
  readonly restPose = uniform(0)

  private readonly staticPass: ComputeNode
  private readonly surfacePass: ComputeNode
  private readonly normalPass: ComputeNode
  private readonly backPass: ComputeNode
  private readonly backNormalPass: ComputeNode
  private dirty = true

  constructor(sim: BubbleSim) {
    const { pos, normal, rest, info } = this
    const ixy = (i: I) => [i.mod(this.RW), i.div(this.RW)] as const

    this.staticPass = Fn(() => {
      const i = int(instanceIndex)
      const [rx, ry] = ixy(i)
      const x = float(rx).mul(this.h).sub(sim.half.x)
      const z = float(ry).mul(this.h).sub(sim.half.y)
      // Map coordinates of this render vertex: exact texels at 1x and 2x, between texels at 4x.
      const mx = float(rx).mul(MAP_SCALE).div(float(this.detail))
      const my = float(ry).mul(MAP_SCALE).div(float(this.detail))
      const tx = int(floor(mx))
      const ty = int(floor(my))
      const fx = fract(mx)
      const fy = fract(my)
      const at = (dx: number, dy: number) =>
        sim.map.element(
          imin(ty.add(dy), sim.mapH.sub(1))
            .mul(sim.mapW)
            .add(imin(tx.add(dx), sim.mapW.sub(1))),
        )
      const sd = mix(mix(at(0, 0).y, at(1, 0).y, fx), mix(at(0, 1).y, at(1, 1).y, fx), fy)
      const nx = imin(tx.add(int(fx.add(0.5))), sim.mapW.sub(1))
      const ny = imin(ty.add(int(fy.add(0.5))), sim.mapH.sub(1))
      const near = sim.map.element(ny.mul(sim.mapW).add(nx))
      const id = near.x
      const inside = id.greaterThanEqual(0).and(sd.lessThan(0))
      const c = sim.centers.element(int(max(id, 0)))
      // Same dome height rule as the sheet builder (0.4x the equivalent diameter, lower for narrow shapes).
      const rEq = c.w.mul(0.5)
      const domeH = rEq.mul(0.76).mul(min(float(1), c.z.div(max(rEq, 1e-6)).mul(0.45).add(0.55)))
      const t = clamp(sd.negate().div(max(c.z, 1e-6)), 0, 1)
      const cap = sqrt(float(1).sub(float(1).sub(t).mul(float(1).sub(t))))
      const y = select(inside, domeH.mul(cap).mul(smoothstep(0, this.foot, sd.negate())), float(0))
      rest.element(i).assign(vec4(x, y, z, select(inside, t, float(0))))
      info.element(i).assign(vec4(select(inside, id, float(-1)), select(inside, float(1), float(0)), sd, 0))
    })().compute(1)

    // Catmull-Rom weights for the four sim samples around a fraction t of a segment.
    const weights = (t: Node<'float'>) => {
      const t2 = t.mul(t)
      const t3 = t2.mul(t)
      return [
        t3.negate().add(t2.mul(2)).sub(t).mul(0.5),
        t3.mul(3).sub(t2.mul(5)).add(2).mul(0.5),
        t3.mul(-3).add(t2.mul(4)).add(t).mul(0.5),
        t3.sub(t2).mul(0.5),
      ]
    }

    // Rest shape plus the sim's displacement from its own rest pose, Catmull-Rom interpolated.
    const surface = (
      out: typeof pos,
      restAt: (i: I) => Node<'vec3'>,
      simPos: (j: I) => Node<'vec3'>,
      simRest: (j: I) => Node<'vec3'>,
      squeezeOf: ((j: I) => Node<'float'>) | null,
    ) =>
      Fn(() => {
        const i = int(instanceIndex)
        const [rx, ry] = ixy(i)
        const sx = float(rx).div(float(this.detail))
        const sy = float(ry).div(float(this.detail))
        const ix = int(floor(sx))
        const iy = int(floor(sy))
        const wx = weights(fract(sx))
        const wy = weights(fract(sy))
        const simIndex = (dx: number, dy: number) =>
          iclamp(iy.add(dy), int(0), sim.uH.sub(1))
            .mul(sim.uW)
            .add(iclamp(ix.add(dx), int(0), sim.uW.sub(1)))
        const disp = vec3(0).toVar()
        for (let dy = -1; dy <= 2; dy++) {
          for (let dx = -1; dx <= 2; dx++) {
            const j = simIndex(dx, dy)
            disp.addAssign(
              simPos(j)
                .sub(simRest(j))
                .mul(wx[dx + 1].mul(wy[dy + 1])),
            )
          }
        }
        // Compression for the wrinkle layer: bilinear over the containing sim quad.
        const fx = fract(sx)
        const fy = fract(sy)
        const squeeze = squeezeOf
          ? mix(
              mix(squeezeOf(simIndex(0, 0)), squeezeOf(simIndex(1, 0)), fx),
              mix(squeezeOf(simIndex(0, 1)), squeezeOf(simIndex(1, 1)), fx),
              fy,
            )
          : float(0)
        const still = this.restPose.greaterThan(0.5)
        out.element(i).assign(vec4(restAt(i).add(select(still, vec3(0), disp)), select(still, float(0), squeeze)))
      })().compute(1)

    const normals = (src: typeof pos, dst: typeof pos) =>
      Fn(() => {
        const i = int(instanceIndex)
        const [rx, ry] = ixy(i)
        const il = select(rx.greaterThan(0), i.sub(1), i)
        const ir = select(rx.lessThan(this.RW.sub(1)), i.add(1), i)
        const iu = select(ry.greaterThan(0), i.sub(this.RW), i)
        const id = select(ry.lessThan(this.RH.sub(1)), i.add(this.RW), i)
        const n = cross(src.element(id).xyz.sub(src.element(iu).xyz), src.element(ir).xyz.sub(src.element(il).xyz))
        dst.element(i).assign(vec4(normalize(n), src.element(i).w))
      })().compute(1)

    this.surfacePass = surface(
      pos,
      (i) => rest.element(i).xyz,
      (j) => sim.x.element(j).xyz,
      (j) => sim.rest.element(j).xyz,
      (j) => sim.normal.element(j).w,
    )
    this.normalPass = normals(pos, normal)
    // The back film's rest shape is flat; its sim counterpart is the harmonic-filled base.
    const flat = (v: Node<'vec4'>) => vec3(v.x, 0, v.z)
    this.backPass = surface(
      this.backPos,
      (i) => flat(rest.element(i)).sub(vec3(0, BACK_OFFSET, 0)),
      (j) => sim.back.element(j).xyz,
      (j) => flat(sim.rest.element(j)),
      null,
    )
    this.backNormalPass = normals(this.backPos, this.backNormal)
  }

  /** Point at a sheet and detail level. Returns the render grid size for the index buffer. */
  configure(sheet: Sheet, detail: number) {
    const { RW, RH } = renderSize(sheet.W, sheet.H, detail)
    this.detail.value = detail
    this.RW.value = RW
    this.RH.value = RH
    this.h.value = sheet.h / detail
    // The soft foot is about one sim segment wide: enough to round the rim at render resolution.
    this.foot.value = sheet.h
    for (const k of [this.staticPass, this.surfacePass, this.normalPass, this.backPass, this.backNormalPass])
      k.count = RW * RH
    this.dirty = true
    return { RW, RH }
  }

  update(renderer: WebGPURenderer) {
    if (this.dirty) {
      renderer.compute(this.staticPass)
      this.dirty = false
    }
    renderer.compute([this.surfacePass, this.normalPass, this.backPass, this.backNormalPass])
  }
}

/** Triangles for an RW x RH render grid, alternating the diagonal for symmetric domes. */
export function renderIndex(RW: number, RH: number) {
  const index = new Uint32Array((RW - 1) * (RH - 1) * 6)
  let n = 0
  for (let y = 0; y < RH - 1; y++) {
    for (let x = 0; x < RW - 1; x++) {
      const a = y * RW + x
      const b = a + 1
      const c = a + RW + 1
      const d = a + RW
      if ((x + y) % 2 === 0) index.set([a, d, c, a, c, b], n)
      else index.set([a, d, b, b, d, c], n)
      n += 6
    }
  }
  return index
}

/**
 * Triangles for the back film: only render quads whose corners come within a seal band of a
 * bubble footprint. Elsewhere the two films are bonded and the cap layer stands for both.
 */
export function backIndex(sheet: Sheet, detail: number) {
  const { RW, RH } = renderSize(sheet.W, sheet.H, detail)
  const band = sheet.h * 0.75
  const sd = new Float32Array(RW * RH)
  for (let ry = 0; ry < RH; ry++) {
    for (let rx = 0; rx < RW; rx++) {
      const mx = (rx * MAP_SCALE) / detail
      const my = (ry * MAP_SCALE) / detail
      const tx = Math.floor(mx)
      const ty = Math.floor(my)
      const fx = mx - tx
      const fy = my - ty
      const at = (dx: number, dy: number) =>
        sheet.map[(Math.min(ty + dy, sheet.mapH - 1) * sheet.mapW + Math.min(tx + dx, sheet.mapW - 1)) * 4 + 1]
      sd[ry * RW + rx] = (at(0, 0) * (1 - fx) + at(1, 0) * fx) * (1 - fy) + (at(0, 1) * (1 - fx) + at(1, 1) * fx) * fy
    }
  }
  const out: number[] = []
  for (let y = 0; y < RH - 1; y++) {
    for (let x = 0; x < RW - 1; x++) {
      const a = y * RW + x
      const b = a + 1
      const c = a + RW + 1
      const d = a + RW
      if (Math.min(sd[a], sd[b], sd[c], sd[d]) > band) continue
      if ((x + y) % 2 === 0) out.push(a, d, c, a, c, b)
      else out.push(a, d, b, b, d, c)
    }
  }
  return new Uint32Array(out)
}
