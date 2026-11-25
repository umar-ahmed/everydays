import {
  Fn,
  If,
  Loop,
  abs,
  cos,
  exp,
  float,
  instanceIndex,
  instancedArray,
  int,
  log,
  max,
  mix,
  normalize,
  positionGeometry,
  select,
  sin,
  sqrt,
  storage,
  texture,
  uint,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl'
import * as THREE from 'three/webgpu'
import type { Node, StorageBufferNode } from 'three/webgpu'
import {
  CMF,
  EPS,
  LAMBDA_MAX,
  LAMBDA_MIN,
  LENS_OFFSET,
  MAX_EDGES,
  MAX_SHAPES,
  MIRROR_REFLECTANCE,
  WHITE_BALANCE,
  XYZ_TO_SRGB,
  compile,
  type Optics,
  type Scene,
  type View,
} from './optics.ts'

type F = Node<'float'>
type V2 = Node<'vec2'>
type U = Node<'uint'>

/** Segments per path: the emitted ray plus up to this many bounces. */
export const MAX_BOUNCES = 16
const SLOTS = MAX_BOUNCES + 1
/** Line half-width in device pixels; the profile across it is a tent. */
const HALF_WIDTH = 1
/** Brightness of a beam spanning a twentieth of the view at power 1, before exposure. */
const BRIGHTNESS = 0.05

export type Display = { exposure: number }
export type Tracer = {
  readonly maxPaths: number
  /** Paths traced so far since the last change. */
  readonly paths: number
  /** Stops tracing and keeps the current picture (for captures). */
  paused: boolean
  setScene(scene: Scene, optics: Optics, pathsPerFrame: number): void
  setDisplay(d: Display): void
  dispose(): void
}

const pcg = (s: U) => {
  const state = s.mul(uint(747796405)).add(uint(2891336453))
  const word = state
    .shiftRight(state.shiftRight(uint(28)).add(uint(4)))
    .bitXor(state)
    .mul(uint(277803737))
  return word.shiftRight(uint(22)).bitXor(word)
}

const lobes = (l: F, fit: readonly (readonly number[])[]) =>
  fit.reduce<F>((sum, [w, mu, s1, s2]) => {
    const t = l.sub(mu).div(select(l.lessThan(mu), float(s1), float(s2)))
    return sum.add(exp(t.mul(t).mul(-0.5)).mul(w))
  }, float(0))

/** White-balanced linear sRGB of a wavelength (see `spectralRgb`). */
const spectral = (l: F) => {
  const x = lobes(l, CMF.x)
  const y = lobes(l, CMF.y)
  const z = lobes(l, CMF.z)
  const m = XYZ_TO_SRGB
  return vec3(
    x.mul(m[0]).add(y.mul(m[1])).add(z.mul(m[2])).mul(WHITE_BALANCE[0]),
    x.mul(m[3]).add(y.mul(m[4])).add(z.mul(m[5])).mul(WHITE_BALANCE[1]),
    x.mul(m[6]).add(y.mul(m[7])).add(z.mul(m[8])).mul(WHITE_BALANCE[2]),
  )
}

const rgb = (t: ReturnType<typeof texture>) => (t as unknown as Node<'vec4'>).rgb

const cross = (a: V2, b: V2) => a.x.mul(b.y).sub(a.y.mul(b.x))

/**
 * Progressive 2D light tracing on the GPU. Each frame a compute pass traces `pathsPerFrame`
 * light paths, one per thread, and writes every segment it travels (with its colour) to a
 * buffer. The segments are drawn as thin additive quads into a half-float frame, which is added
 * to a float32 accumulation (ping-ponged, so it never needs float blending). The display pass
 * averages the accumulation, exposes and tone maps it.
 *
 * The fluence picture this draws is what a 2D world with faintly dusty air would show: a path
 * lights every pixel it crosses in proportion to the length inside it.
 */
export class GpuTracer implements Tracer {
  readonly maxPaths: number
  paths = 0
  paused = false
  private renderer: THREE.WebGPURenderer
  private container: HTMLElement
  private resize = new ResizeObserver(() => this.fit())
  private disposed = false

  private segs: StorageBufferNode<'vec4'>
  private cols: StorageBufferNode<'vec4'>
  private edges: StorageBufferNode<'vec4'>
  private circles: StorageBufferNode<'vec4'>
  private edgeSurfaces: StorageBufferNode<'vec4'>
  private circleSurfaces: StorageBufferNode<'vec4'>
  private kernel: THREE.ComputeNode
  private lines: THREE.Mesh
  private geometry: THREE.InstancedBufferGeometry
  private lineScene = new THREE.Scene()
  private camera = new THREE.OrthographicCamera()
  private frameRT: THREE.RenderTarget
  private acc: [THREE.RenderTarget, THREE.RenderTarget]
  private accumulate: THREE.QuadMesh
  private display: THREE.QuadMesh
  private prevTex: ReturnType<typeof texture>
  private accTex: ReturnType<typeof texture>

  private u = {
    flash: uniform(new THREE.Vector2()),
    rot: uniform(0),
    cone: uniform(0),
    aperture: uniform(0),
    slots: uniform(1, 'int'),
    ior: uniform(1.5),
    dispersion: uniform(0),
    haze: uniform(0),
    walls: uniform(0),
    albedo: uniform(0),
    weight: uniform(0),
    view: uniform(new THREE.Vector2(1, 1)),
    edgeCount: uniform(0, 'int'),
    circleCount: uniform(0, 'int'),
    jitterA: uniform(0, 'uint'),
    jitterB: uniform(0, 'uint'),
    seed: uniform(0, 'uint'),
    pixel: uniform(0.001),
    scale: uniform(0),
    exposure: uniform(1),
  }

  private view: View = { hw: 1, hh: 1 }
  private pathsPerFrame = 1
  private frames = 0
  private dirty = true
  private scene: Scene | null = null
  private optics: Optics | null = null
  private onStats: (paths: number) => void

  constructor(container: HTMLElement, maxPaths: number, onStats: (paths: number) => void) {
    this.container = container
    this.maxPaths = maxPaths
    this.onStats = onStats
    this.renderer = new THREE.WebGPURenderer({ antialias: false })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, maxPaths > 20000 ? 2 : 1.5))
    this.renderer.autoClear = false
    this.renderer.toneMapping = THREE.NeutralToneMapping
    const canvas = this.renderer.domElement
    canvas.style.display = 'block'
    canvas.style.width = '100%'
    canvas.style.height = '100%'
    container.appendChild(canvas)

    const count = maxPaths * SLOTS
    this.segs = instancedArray(count, 'vec4') as unknown as StorageBufferNode<'vec4'>
    this.cols = instancedArray(count, 'vec4') as unknown as StorageBufferNode<'vec4'>
    this.edges = instancedArray(new Float32Array(MAX_EDGES * 4), 'vec4') as unknown as StorageBufferNode<'vec4'>
    this.circles = instancedArray(new Float32Array(MAX_SHAPES * 4), 'vec4') as unknown as StorageBufferNode<'vec4'>
    this.edgeSurfaces = instancedArray(new Float32Array(MAX_EDGES * 4), 'vec4') as unknown as StorageBufferNode<'vec4'>
    this.circleSurfaces = instancedArray(
      new Float32Array(MAX_SHAPES * 4),
      'vec4',
    ) as unknown as StorageBufferNode<'vec4'>
    this.kernel = this.buildKernel()

    const plane = new THREE.PlaneGeometry(1, 1)
    this.geometry = new THREE.InstancedBufferGeometry()
    this.geometry.index = plane.index
    this.geometry.setAttribute('position', plane.getAttribute('position'))
    this.geometry.instanceCount = 0
    this.lines = new THREE.Mesh(this.geometry, this.buildLineMaterial(count))
    this.lines.frustumCulled = false
    this.lineScene.add(this.lines)

    const rt = (type: THREE.TextureDataType) =>
      new THREE.RenderTarget(1, 1, {
        type,
        depthBuffer: false,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
      })
    this.frameRT = rt(THREE.HalfFloatType)
    this.acc = [rt(THREE.FloatType), rt(THREE.FloatType)]

    this.prevTex = texture(this.acc[0].texture)
    const frameTex = texture(this.frameRT.texture)
    const add = new THREE.MeshBasicNodeMaterial()
    add.colorNode = vec4(rgb(this.prevTex).add(rgb(frameTex)), 1)
    add.toneMapped = false
    this.accumulate = new THREE.QuadMesh(add)

    this.accTex = texture(this.acc[1].texture)
    const show = new THREE.MeshBasicNodeMaterial()
    const light = max(rgb(this.accTex).mul(this.u.scale), vec3(0))
    show.colorNode = vec4(light.add(vec3(0.004, 0.0045, 0.008)), 1)
    this.display = new THREE.QuadMesh(show)
  }

  async init() {
    await this.renderer.init()
    if (!(this.renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend) throw new Error('webgpu-unavailable')
    if (this.disposed) return
    this.resize.observe(this.container)
    this.fit()
    await this.renderer.setAnimationLoop(this.frame)
  }

  private buildKernel() {
    const { u, segs, cols, edges, circles, edgeSurfaces, circleSurfaces } = this
    return Fn(() => {
      const path = instanceIndex
      const rng = pcg(path.add(pcg(u.seed.add(uint(1))))).toVar()
      const rand = () => {
        rng.assign(pcg(rng))
        return float(rng)
          .mul(1 / 2 ** 32)
          .toVar()
      }
      // Wavelength and emission angle from the R2 low-discrepancy sequence over the frame's
      // paths (in fixed point, exact for any path count), rotated by a random offset per frame.
      const r2 = (k: number, j: U) => float(path.mul(uint(k)).add(j)).mul(1 / 2 ** 32)
      const lambda = r2(3242174889, u.jitterA)
        .mul(LAMBDA_MAX - LAMBDA_MIN)
        .add(LAMBDA_MIN)
        .toVar()
      const a = u.rot.add(r2(2447445413, u.jitterB).sub(0.5).mul(u.cone))
      const lens = rand().sub(0.5).mul(u.aperture)
      const axis = vec2(cos(u.rot), sin(u.rot))
      const o = u.flash.add(axis.mul(LENS_OFFSET)).add(vec2(axis.y.negate(), axis.x).mul(lens)).toVar()
      const d = vec2(cos(a), sin(a)).toVar()
      const n = u.ior.add(u.dispersion.div(lambda.mul(0.001).mul(lambda.mul(0.001))))
      const color = spectral(lambda).mul(u.weight).toVar()
      const throughput = float(1).toVar()
      const alive = int(1).toVar()
      const base = int(path).mul(u.slots)

      Loop({ start: int(0), end: u.slots, type: 'int', condition: '<' }, ({ i }) => {
        const slot = base.add(i)
        If(alive.equal(0), () => {
          segs.element(slot).assign(vec4(0))
          cols.element(slot).assign(vec4(0))
        }).Else(() => {
          // Walls: the view's edges, normals inward. 0 wall, 1 glass, 2 mirror, 3 diffuse.
          const tx = select(d.x.greaterThan(0), u.view.x.sub(o.x), u.view.x.negate().sub(o.x)).div(d.x)
          const ty = select(d.y.greaterThan(0), u.view.y.sub(o.y), u.view.y.negate().sub(o.y)).div(d.y)
          const xFirst = abs(d.y)
            .lessThan(1e-9)
            .or(abs(d.x).greaterThan(1e-9).and(tx.lessThan(ty)))
          const tHit = max(select(xFirst, tx, ty), 0).toVar()
          const nrm = select(xFirst, vec2(d.x.sign().negate(), 0), vec2(0, d.y.sign().negate())).toVar()
          const kind = int(0).toVar()
          const surf = vec4(0).toVar()

          Loop({ start: int(0), end: u.edgeCount, type: 'int', condition: '<' }, ({ i: e }) => {
            const seg = edges.element(e)
            const p0 = seg.xy
            const ed = seg.zw.sub(seg.xy)
            const den = cross(d, ed)
            const q = p0.sub(o)
            const t = cross(q, ed).div(den)
            const s = cross(q, d).div(den)
            If(
              abs(den)
                .greaterThan(1e-12)
                .and(t.greaterThan(EPS))
                .and(t.lessThan(tHit))
                .and(s.greaterThanEqual(0))
                .and(s.lessThanEqual(1)),
              () => {
                tHit.assign(t)
                nrm.assign(normalize(vec2(ed.y, ed.x.negate())))
                surf.assign(edgeSurfaces.element(e))
                kind.assign(int(surf.x).add(1))
              },
            )
          })

          Loop({ start: int(0), end: u.circleCount, type: 'int', condition: '<' }, ({ i: c }) => {
            const circle = circles.element(c)
            const q = o.sub(circle.xy)
            const b = q.dot(d)
            const disc = b.mul(b).sub(q.dot(q).sub(circle.z.mul(circle.z)))
            const sq = sqrt(max(disc, 0))
            const near = b.negate().sub(sq)
            const t = select(near.greaterThan(EPS), near, b.negate().add(sq))
            If(disc.greaterThanEqual(0).and(t.greaterThan(EPS)).and(t.lessThan(tHit)), () => {
              tHit.assign(t)
              nrm.assign(q.add(d.mul(t)).div(circle.z))
              surf.assign(circleSurfaces.element(c))
              kind.assign(int(surf.x).add(1))
            })
          })

          // Free flight through the hazy air: scatter before the surface, if it comes first.
          const free = log(float(1).sub(rand())).negate().div(max(u.haze, 1e-6))
          const scatter = u.haze.greaterThan(0).and(free.lessThan(tHit))
          const t = select(scatter, free, tHit)
          const end = o.add(d.mul(t)).toVar()
          segs.element(slot).assign(vec4(o, end))
          cols.element(slot).assign(vec4(color.mul(throughput), 0))

          const r1 = rand()
          const r2b = rand()
          If(scatter, () => {
            const phi = r1.mul(Math.PI * 2)
            d.assign(vec2(cos(phi), sin(phi)))
          })
            .ElseIf(kind.equal(0).or(kind.equal(3)), () => {
              // Russian roulette on the albedo keeps every drawn segment at full strength.
              const bump = lambda.sub(surf.y).div(max(surf.z, 1e-3))
              const tint = select(
                surf.z.greaterThan(0),
                surf.w.add(
                  float(1)
                    .sub(surf.w)
                    .mul(exp(bump.mul(bump).mul(-0.5))),
                ),
                float(1),
              )
              If(r1.greaterThanEqual(select(kind.equal(0), u.walls, u.albedo.mul(tint))), () => {
                alive.assign(0)
              })
              // Bounce off the side the light arrived from, cosine-weighted about that normal.
              const nn = select(d.dot(nrm).lessThan(0), nrm, nrm.negate())
              const s = r2b.mul(2).sub(1)
              const k = sqrt(max(float(1).sub(s.mul(s)), 0))
              d.assign(vec2(nn.x.mul(k).sub(nn.y.mul(s)), nn.y.mul(k).add(nn.x.mul(s))))
            })
            .ElseIf(kind.equal(2), () => {
              d.assign(d.sub(nrm.mul(d.dot(nrm).mul(2))))
              throughput.mulAssign(MIRROR_REFLECTANCE)
            })
            .Else(() => {
              const entering = d.dot(nrm).lessThan(0)
              const nn = select(entering, nrm, nrm.negate())
              const n1 = select(entering, float(1), n)
              const n2 = select(entering, n, float(1))
              const cosi = d.dot(nn).negate()
              const eta = n1.div(n2)
              const k = float(1).sub(eta.mul(eta).mul(float(1).sub(cosi.mul(cosi))))
              const cost = sqrt(max(k, 0))
              const rs = n1
                .mul(cosi)
                .sub(n2.mul(cost))
                .div(n1.mul(cosi).add(n2.mul(cost)))
              const rp = n1
                .mul(cost)
                .sub(n2.mul(cosi))
                .div(n1.mul(cost).add(n2.mul(cosi)))
              const R = rs.mul(rs).add(rp.mul(rp)).mul(0.5)
              const reflect = k.lessThan(0).or(r1.lessThan(R))
              const reflected = d.add(nn.mul(cosi.mul(2)))
              const refracted = d.mul(eta).add(nn.mul(eta.mul(cosi).sub(cost)))
              d.assign(normalize(select(reflect, reflected, refracted)))
            })
          o.assign(end.add(d.mul(EPS)))
        })
      })
    })().compute(this.maxPaths)
  }

  private buildLineMaterial(count: number) {
    const { u } = this
    const segs = storage(this.segs.value, 'vec4', count).toReadOnly()
    const cols = storage(this.cols.value, 'vec4', count).toReadOnly()
    const m = new THREE.NodeMaterial()
    const seg = segs.element(instanceIndex)
    const p0 = seg.xy
    const p1 = seg.zw
    const along = p1.sub(p0)
    const len = along.length()
    const side = vec2(along.y.negate(), along.x).div(max(len, 1e-9))
    const p = mix(p0, p1, positionGeometry.x.add(0.5)).add(
      side.mul(positionGeometry.y.mul(2 * HALF_WIDTH).mul(u.pixel)),
    )
    // World is y-down with the view's half extents in `view`; zero-length segments go off-screen.
    const clip = vec4(p.x.div(u.view.x), p.y.div(u.view.y).negate(), 0, 1)
    m.vertexNode = select(len.greaterThan(1e-7), clip, vec4(2, 2, 2, 1))
    const tent = float(1).sub(abs(varying(positionGeometry.y.mul(2))))
    const tint = varying(cols.element(instanceIndex).xyz)
    m.fragmentNode = vec4(tint.mul(tent).div(HALF_WIDTH), 1)
    m.blending = THREE.CustomBlending
    m.blendEquation = THREE.AddEquation
    m.blendSrc = THREE.OneFactor
    m.blendDst = THREE.OneFactor
    m.blendSrcAlpha = THREE.OneFactor
    m.blendDstAlpha = THREE.OneFactor
    m.transparent = true
    m.depthTest = false
    m.depthWrite = false
    m.toneMapped = false
    m.side = THREE.DoubleSide
    return m
  }

  private fit() {
    const w = Math.max(1, this.container.clientWidth)
    const h = Math.max(1, this.container.clientHeight)
    this.renderer.setSize(w, h, false)
    const ratio = this.renderer.getPixelRatio()
    const pw = Math.round(w * ratio)
    const ph = Math.round(h * ratio)
    for (const rt of [this.frameRT, ...this.acc]) rt.setSize(pw, ph)
    this.dirty = true
  }

  /** View half extents for the current canvas, as the overlay sees them. */
  get size() {
    return { width: this.container.clientWidth, height: this.container.clientHeight }
  }

  setScene(scene: Scene, optics: Optics, pathsPerFrame: number) {
    this.scene = scene
    this.optics = optics
    this.pathsPerFrame = Math.max(1, Math.min(this.maxPaths, Math.round(pathsPerFrame)))
    this.dirty = true
  }

  setDisplay(d: Display) {
    this.u.exposure.value = d.exposure
  }

  private upload(view: View, heightPx: number) {
    const { u, scene, optics } = this
    if (!scene || !optics) return
    const c = compile(scene.shapes)
    ;(this.edges.value.array as Float32Array).set(c.edges)
    this.edges.value.needsUpdate = true
    ;(this.circles.value.array as Float32Array).set(c.circles)
    this.circles.value.needsUpdate = true
    u.edgeCount.value = c.edgeCount
    ;(this.edgeSurfaces.value.array as Float32Array).set(c.edgeSurfaces)
    this.edgeSurfaces.value.needsUpdate = true
    ;(this.circleSurfaces.value.array as Float32Array).set(c.circleSurfaces)
    this.circleSurfaces.value.needsUpdate = true
    u.circleCount.value = c.circleCount
    u.flash.value.set(scene.flashlight.x, scene.flashlight.y)
    u.rot.value = (optics.rotation * Math.PI) / 180
    u.cone.value = (optics.cone * Math.PI) / 180
    u.aperture.value = optics.aperture
    u.slots.value = Math.min(MAX_BOUNCES, Math.max(0, Math.round(optics.bounces))) + 1
    u.ior.value = optics.ior
    u.dispersion.value = optics.dispersion
    u.haze.value = optics.haze
    u.walls.value = optics.walls
    u.albedo.value = optics.albedo
    u.view.value.set(view.hw, view.hh)
    u.weight.value = (optics.power * BRIGHTNESS * heightPx) / this.pathsPerFrame
  }

  private frame = () => {
    if (this.disposed || this.paused || !this.scene) return
    const { renderer, u } = this
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    if (w < 1 || h < 1) return
    const aspect = w / h
    this.view = aspect >= 1 ? { hw: aspect, hh: 1 } : { hw: 1, hh: 1 / aspect }
    const ph = this.frameRT.height
    if (this.dirty) {
      this.dirty = false
      this.upload(this.view, ph)
      this.frames = 0
      this.paths = 0
      for (const rt of this.acc) {
        renderer.setRenderTarget(rt)
        renderer.setClearColor(0x000000, 0)
        renderer.clear()
      }
    }
    u.pixel.value = (2 * this.view.hh) / ph
    u.seed.value = (u.seed.value + 1) >>> 0
    u.jitterA.value = (Math.random() * 2 ** 32) >>> 0
    u.jitterB.value = (Math.random() * 2 ** 32) >>> 0
    this.kernel.count = this.pathsPerFrame
    void renderer.compute(this.kernel)
    this.geometry.instanceCount = this.pathsPerFrame * u.slots.value

    renderer.setRenderTarget(this.frameRT)
    renderer.setClearColor(0x000000, 0)
    renderer.clear()
    renderer.render(this.lineScene, this.camera)

    const [prev, next] = this.acc
    this.prevTex.value = prev.texture
    renderer.setRenderTarget(next)
    this.accumulate.render(renderer)
    this.acc = [next, prev]
    this.frames++
    this.paths += this.pathsPerFrame

    this.accTex.value = next.texture
    u.scale.value = u.exposure.value / this.frames
    renderer.setRenderTarget(null)
    this.display.render(renderer)
    this.onStats(this.paths)
  }

  dispose() {
    this.disposed = true
    this.resize.disconnect()
    void this.renderer.setAnimationLoop(null)
    this.geometry.dispose()
    this.frameRT.dispose()
    for (const rt of this.acc) rt.dispose()
    void this.renderer.dispose()
    this.renderer.domElement.remove()
  }
}
