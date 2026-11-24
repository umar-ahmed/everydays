import {
  clamp,
  dot,
  exp,
  float,
  floor,
  fract,
  fwidth,
  int,
  max,
  min,
  mix,
  mx_noise_float,
  normalize,
  pow,
  sin,
  vec2,
  positionWorld,
  select,
  smoothstep,
  transformNormalToView,
  unpackUnorm4x8,
  uniform,
  vec3,
  vertexIndex,
} from 'three/tsl'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import * as THREE from 'three/webgpu'
import type { Node } from 'three/webgpu'
import { Look, PALETTES, type PaletteName } from './palettes.ts'
import {
  ErosionSim,
  RESOLUTIONS,
  SIZE,
  bufferBytes,
  defaultResolution,
  fitResolution,
  simBytes,
  type ErosionParams,
  type TerrainParams,
} from './sim.ts'

export type Params = TerrainParams &
  ErosionParams & {
    eroding: boolean
    iterations: number
    view: 'shaded' | 'water' | 'erosion' | 'hardness' | 'color'
    /** Show the colour the erosion has moved (it is tracked either way). */
    colorTransport: boolean
    /** Strata contrast and saturation boost for the moved colour, 0..1. */
    vividness: number
    /** Cells per side of the heightfield. */
    resolution: number
    palette: PaletteName
    /** Snow: line (fraction of peak height), softness, slope limit (degrees), aspect bias, sparkle. */
    snowLine: number
    snowSoftness: number
    snowSlope: number
    snowAspect: number
    sparkle: number
  }

/** Phones get a smaller default grid and render mesh. */
export const compactDevice = () => typeof screen !== 'undefined' && Math.min(screen.width, screen.height) < 600

/** Most vertices per side of the render mesh; finer sims are shaded per pixel from the grid. */
const meshMax = () => (compactDevice() ? 512 : 1024)
/** Cell-steps per frame the Iterations dial is measured against: finer grids run fewer steps a
 * frame (fractionally, down to one every few frames) so orbiting stays smooth. */
const FRAME_BUDGET = 1024 * 1024

/** Direction toward the sun (the key light sits at -6, 5, 3). */
const SUN = new THREE.Vector3(-6, 5, 3).normalize()

export type Stats = { steps: number }
/** The grid size actually running, and why it isn't the one asked for (if it isn't). */
export type SizeNotice = { requested: number; actual: number; reason: string | null }

export type Limits = { maxBufferSize: number; maxStorageBufferBindingSize: number }

/** The adapter's buffer limits, or null without WebGPU. */
export async function gpuLimits(): Promise<Limits | null> {
  if (!('gpu' in navigator) || !navigator.gpu) return null
  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return null
    return {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    }
  } catch {
    return null
  }
}

const mib = (bytes: number) =>
  bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`

const terrainKey = (p: TerrainParams) => `${p.seed}/${p.octaves}/${p.roughness}/${p.ridge}/${p.heightScale}`

type GPUErrorScopes = {
  pushErrorScope(filter: 'validation' | 'out-of-memory'): void
  popErrorScope(): Promise<{ message: string } | null>
}

export class TerrainApp {
  private renderer: THREE.WebGPURenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200)
  private controls: OrbitControls
  /** Set once `init` resolves; replaced (after the old one is freed) when the size changes. */
  sim!: ErosionSim
  private mesh: THREE.Mesh | null = null
  private look = new Look()
  private palette = ''
  private params: Params
  private key = ''
  private view = uniform(0, 'int')
  private steps = 0
  private owed = 0
  private disposed = false
  private building = false
  private requested = 0
  private limits: Limits
  private resize = new ResizeObserver(() => this.fit())
  private container: HTMLElement
  private onStats: (s: Stats) => void
  private onSize: (s: SizeNotice) => void
  private lastStats = 0

  constructor(
    container: HTMLElement,
    params: Params,
    limits: Limits,
    onStats: (s: Stats) => void,
    onSize: (s: SizeNotice) => void,
  ) {
    this.container = container
    this.params = params
    this.limits = limits
    this.onStats = onStats
    this.onSize = onSize
    // Ask for the adapter's full buffer limits: the defaults (128 MB per storage binding) stop
    // at 2048, and a 4096 grid needs 256 MB per buffer.
    this.renderer = new THREE.WebGPURenderer({
      antialias: true,
      requiredLimits: { ...limits },
    } as ConstructorParameters<typeof THREE.WebGPURenderer>[0])
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    // Some browsers drop the whole device when a big grid runs them out of memory.
    this.renderer.onDeviceLost = () =>
      this.onSize({
        requested: this.requested,
        actual: this.requested,
        reason: 'The GPU was reset (most likely out of memory). Reload the page and pick a smaller resolution.',
      })
    // Neutral keeps the palettes' hues and saturation (AgX greyed the presets out).
    this.renderer.toneMapping = THREE.NeutralToneMapping
    const canvas = this.renderer.domElement
    canvas.style.display = 'block'
    canvas.style.width = '100%'
    canvas.style.height = '100%'
    canvas.style.touchAction = 'none'
    container.appendChild(canvas)
    this.controls = new OrbitControls(this.camera, canvas)
  }

  async init() {
    await this.renderer.init()
    if (!(this.renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend) throw new Error('webgpu-unavailable')
    if (this.disposed) return
    const { scene, camera, controls } = this

    // Late-afternoon sun over a hazy sky: warm key, cool sky fill, ground bounce.
    const sky = new THREE.Color(0.3, 0.5, 0.82)
    scene.background = sky
    scene.fog = new THREE.Fog(sky, SIZE * 1.4, SIZE * 4)
    const sun = new THREE.DirectionalLight(0xfff1dc, 3.2)
    sun.position.set(-6, 5, 3)
    scene.add(sun, new THREE.HemisphereLight(0xbcd3ff, 0x5a4a38, 0.9))

    camera.position.set(SIZE * 0.7, SIZE * 0.42, SIZE * 0.72)
    controls.target.set(0, SIZE * 0.06, 0)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = SIZE * 0.25
    controls.maxDistance = SIZE * 2.5
    controls.maxPolarAngle = Math.PI * 0.47
    controls.update()

    this.requested = this.requestedSize(this.params)
    await this.rebuild()
    if (this.disposed) return
    this.resize.observe(this.container)
    this.fit()
    await this.renderer.setAnimationLoop(this.frame)
  }

  private requestedSize(p: Params) {
    return p.resolution || defaultResolution(compactDevice())
  }

  /**
   * Allocate a sim at `n` and fill it, watching for errors: a size can pass the limit check and
   * still not fit in the GPU's memory. Returns null (with everything freed) if it didn't fit.
   */
  private async allocate(n: number) {
    const device = (this.renderer.backend as unknown as { device: GPUErrorScopes }).device
    device.pushErrorScope('out-of-memory')
    device.pushErrorScope('validation')
    let sim: ErosionSim | null = null
    let failed = false
    try {
      sim = new ErosionSim(n)
      sim.setTerrain(this.params)
      sim.setErosion(this.params)
      // Generating writes every buffer, so they are all allocated by the time this returns.
      sim.reset(this.renderer)
    } catch {
      failed = true
    }
    const invalid = await device.popErrorScope()
    const oom = await device.popErrorScope()
    if (!failed && !invalid && !oom && sim) return sim
    sim?.dispose()
    return null
  }

  /**
   * Swap in a sim (and mesh) at the requested size, or the largest size below it that this GPU
   * can hold. The old one is freed first so a big grid doesn't need room for both. The noise is
   * in sheet coordinates, so the same mountains come back, just finer or coarser.
   */
  private async rebuild() {
    if (this.building) return
    this.building = true
    for (;;) {
      const target = this.requested
      if (this.mesh) {
        this.scene.remove(this.mesh)
        this.mesh.geometry.dispose()
        ;(this.mesh.material as THREE.Material).dispose()
        this.mesh = null
        this.sim.dispose()
      }
      let n = fitResolution(target, this.limits)
      const cap = Math.min(this.limits.maxBufferSize, this.limits.maxStorageBufferBindingSize)
      let reason =
        n < target
          ? `${target}² needs ${mib(bufferBytes(target))} per buffer; this GPU allows ${mib(cap)}, so it's running ${n}².`
          : null
      let sim = await this.allocate(n)
      while (!sim && n > RESOLUTIONS[0]) {
        const lower = RESOLUTIONS[RESOLUTIONS.indexOf(n as (typeof RESOLUTIONS)[number]) - 1]
        reason = `${n}² ran out of GPU memory (it needs ${mib(simBytes(n))}), so it's running ${lower}².`
        n = lower
        sim = await this.allocate(n)
      }
      if (!sim) throw new Error('terrain-allocation-failed')
      this.sim = sim
      this.mesh = this.makeTerrain()
      this.scene.add(this.mesh)
      if (this.disposed) break
      // Another size was picked while this one was building: go again.
      if (this.requested !== target) continue
      // The dial gets moved to the size that runs; that shouldn't rebuild again.
      this.requested = n
      this.onSize({ requested: target, actual: n, reason })
      break
    }
    this.building = false
    this.apply(this.params, true)
  }

  private makeTerrain() {
    const { sim } = this
    const n = sim.n
    // The mesh is at most meshMax() per side; its vertices sample the grid bilinearly. Normals,
    // water and colour are looked up per pixel from the full grid, so a 4096 sim keeps its
    // detail on a 1024 mesh.
    const m = Math.min(n, meshMax())
    const geometry = new THREE.BufferGeometry()
    // Positions come from the sim; the attribute only sizes the vertex range.
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(m * m * 3), 3))
    const index = new Uint32Array((m - 1) * (m - 1) * 6)
    let k = 0
    for (let y = 0; y < m - 1; y++) {
      for (let x = 0; x < m - 1; x++) {
        const a = y * m + x
        index.set([a, a + m, a + 1, a + 1, a + m, a + m + 1], k)
        k += 6
      }
    }
    geometry.setIndex(new THREE.BufferAttribute(index, 1))

    const N = int(n)
    type I = Node<'int'>
    type F = Node<'float'>
    const iclamp = (v: I) => select(v.lessThan(0), int(0), select(v.greaterThan(N.sub(1)), N.sub(1), v))
    const cellAt = (x: I, y: I) => sim.state.element(iclamp(y).mul(N).add(iclamp(x)))
    /** Bilinear lookup of the state at fractional cell coordinates. */
    const sample = (cx: F, cy: F) => {
      const x0 = int(floor(cx))
      const y0 = int(floor(cy))
      const tx = fract(cx)
      const ty = fract(cy)
      return mix(
        mix(cellAt(x0, y0), cellAt(x0.add(1), y0), tx),
        mix(cellAt(x0, y0.add(1)), cellAt(x0.add(1), y0.add(1)), tx),
        ty,
      )
    }
    const cell = sim.cell
    const vi = int(vertexIndex)
    const stride = (n - 1) / (m - 1)
    const vx = float(vi.mod(int(m))).mul(stride)
    const vy = float(vi.div(int(m))).mul(stride)
    const vs = sample(vx, vy)
    const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.9 })
    material.positionNode = vec3(
      vx.mul(cell).sub(SIZE / 2),
      vs.x.add(vs.y.mul(0.6)).mul(cell),
      vy.mul(cell).sub(SIZE / 2),
    )

    // Per pixel: the grid position under this fragment.
    const c = positionWorld.xz
      .add(SIZE / 2)
      .div(cell)
      .clamp(0, n - 1)
    const s = sample(c.x, c.y)
    // Normal of the terrain surface (not the water): central differences at the four cells
    // around the pixel, blended bilinearly so it is smooth at any zoom.
    const fx = int(floor(c.x))
    const fy = int(floor(c.y))
    const h = (dx: number, dy: number) => cellAt(fx.add(dx), fy.add(dy)).x
    const grad = (dx: number, dy: number) => vec2(h(dx + 1, dy).sub(h(dx - 1, dy)), h(dx, dy + 1).sub(h(dx, dy - 1)))
    const tx = fract(c.x)
    const ty = fract(c.y)
    const g = mix(mix(grad(0, 0), grad(1, 0), tx), mix(grad(0, 1), grad(1, 1), tx), ty).mul(cell)
    const normal = normalize(vec3(g.x.negate(), cell * 2, g.y.negate()))
    material.normalNode = transformNormalToView(normal)
    const height = s.x
    const water = s.y
    const change = s.x.sub(s.w)
    const near = sim.rock.element(
      iclamp(int(floor(c.y.add(0.5))))
        .mul(N)
        .add(iclamp(int(floor(c.x.add(0.5))))),
    )
    const stratum = sim.strataBand(c.x, c.y, height, near)
    // The loose top layer the erosion moves around, bilinear like the rest.
    const layerAt = (x: I, y: I) =>
      unpackUnorm4x8(sim.surface.element(iclamp(y).mul(N).add(iclamp(x)))) as unknown as Node<'vec4'>
    const lx = int(floor(c.x))
    const ly = int(floor(c.y))
    const layer = mix(
      mix(layerAt(lx, ly), layerAt(lx.add(1), ly), fract(c.x)),
      mix(layerAt(lx, ly.add(1)), layerAt(lx.add(1), ly.add(1)), fract(c.x)),
      fract(c.y),
    )
    const hardness = sim.hardnessLog(c.x, c.y, height, near)

    const { look } = this
    // Slope: 1 flat, 0 vertical.
    const flat = normal.y
    const steep = smoothstep(0.86, 0.6, flat)
    const relief = height.div(max(sim.heightCells, 1))
    const t = clamp(relief, 0, 1)
    // Height ramp for gentle ground: four stops at 0, 0.3, 0.6 and 1 of the peak height.
    const rampAt = (u: F) =>
      mix(
        mix(mix(look.ramp[0], look.ramp[1], smoothstep(0, 0.3, u)), look.ramp[2], smoothstep(0.3, 0.6, u)),
        look.ramp[3],
        smoothstep(0.6, 1, u),
      )
    const ramp = rampAt(t)
    // Rock by stratum (0 soft, 1 hard): hard bands paler, soft darker, and with Vividness the
    // soft ones also take the cut-rock hue, so strata read as coloured bands.
    const rockOf = (st: F) =>
      mix(look.rock, mix(look.rockCut, look.rock, st), look.vivid).add(
        st
          .mul(2)
          .sub(1)
          .mul(0.5)
          .mul(look.strata.add(look.vivid.mul(look.colorOn).mul(0.2))),
      )
    // Contour lines every 5% of the peak height, antialiased in screen space.
    const iso = relief.mul(20)
    const isoDist = min(fract(iso), float(1).sub(fract(iso))).div(max(fwidth(iso), 1e-4))
    const ground = ramp.mul(float(1).sub(look.contours.mul(float(1).sub(clamp(isoDist, 0, 1)))))
    // Exposed rock shows the same strata the erosion cuts through, warmer where the water cut
    // through them.
    const rock = mix(rockOf(stratum.mul(0.5).add(0.5)), look.rockCut, smoothstep(-0.3, -3, change).mul(0.6))
    const deposited = smoothstep(0.02, 0.35, change)
    let albedo = mix(ground, rock, steep.mul(look.rockiness))
    // Moved material, mapped through the same palette from where it came from (its relief,
    // stratum and rockiness), then pushed apart in saturation so streaks and fans read.
    const moved = mix(rampAt(layer.x), rockOf(layer.y), layer.z.mul(look.rockiness))
    const lum = dot(moved, vec3(0.299, 0.587, 0.114))
    const vivid = max(mix(vec3(lum), moved, look.vivid.mul(1.2).add(1)), 0)
    const cover = smoothstep(0.05, 0.7, layer.w)
    // Without colour transport, deposits are plain palette sediment as before.
    albedo = mix(albedo, look.sand, deposited.mul(float(1).sub(steep.mul(0.7))).mul(float(1).sub(look.colorOn)))
    albedo = mix(albedo, vivid, cover.mul(look.colorOn))
    const wet = smoothstep(0.002, 0.05, water)

    // Snow: above a noisy snow line, lower on slopes facing away from the sun, only where the
    // ground is flatter than the slope limit (ledges hold it, sheer cliffs don't), and melted
    // where water runs.
    const wobble = mx_noise_float(positionWorld.mul(1.3))
      .mul(0.65)
      .add(mx_noise_float(positionWorld.mul(5.1)).mul(0.35))
    const shade = dot(normalize(vec3(normal.x, 0, normal.z).add(vec3(1e-4, 0, 0))), vec3(-SUN.x, 0, -SUN.z).normalize())
    const sideways = float(1).sub(flat).mul(4).clamp(0, 1)
    const line = look.snowLine
      .sub(shade.mul(sideways).mul(look.snowAspect).mul(0.12))
      .add(wobble.mul(look.snowSoftness).mul(0.9))
    const soft = max(look.snowSoftness, 0.005)
    const snow = smoothstep(line.sub(soft), line.add(soft), relief)
      .mul(smoothstep(look.snowSlope.sub(0.06), look.snowSlope.add(0.04), flat))
      .mul(float(1).sub(wet))
    albedo = mix(albedo, vec3(0.92, 0.94, 0.98), snow)

    // Wet ground darkens; standing and running water tints by depth.
    albedo = albedo.mul(mix(float(1), float(0.55), wet))
    const depth = float(1).sub(exp(water.mul(-1.2)))
    const waterColor = mix(look.water, look.water.mul(0.3), depth)
    const shaded = mix(albedo, waterColor, depth.mul(0.92))
    // Sun-facing glints in the snow.
    const cellHash = fract(sin(dot(floor(positionWorld.xz.mul(160)), vec2(12.9898, 78.233))).mul(43758.5453))
    const glint = smoothstep(0.985, 1, cellHash)
      .mul(pow(max(dot(normal, vec3(SUN.x, SUN.y, SUN.z)), 0), float(6)))
      .mul(snow)
      .mul(look.sparkle)
    material.emissiveNode = vec3(glint.mul(2.5))
    // Debug views: water depth; terrain change (blue cut, orange filled); rock hardness (dark
    // soft, pale hard, joints red).
    const waterView = mix(vec3(0.08), vec3(0.2, 0.6, 1), clamp(water.mul(2), 0, 1))
    const erosionView = select(
      change.lessThan(0),
      mix(vec3(0.85), vec3(0.15, 0.35, 0.9), clamp(change.negate().div(3), 0, 1)),
      mix(vec3(0.85), vec3(0.95, 0.55, 0.15), clamp(change.div(1.5), 0, 1)),
    )
    const hardView = mix(vec3(0.12, 0.1, 0.09), vec3(0.95, 0.9, 0.8), clamp(hardness.mul(0.35).add(0.5), 0, 1)).add(
      vec3(0.6, -0.1, -0.1).mul(near.y.mul(sim.joints)),
    )
    // Moved colour on its own, over dark bedrock.
    const colorView = mix(vec3(0.1), vivid, cover)
    const viewMode = this.view
    material.colorNode = select(
      viewMode.equal(0),
      shaded,
      select(
        viewMode.equal(1),
        waterView,
        select(viewMode.equal(2), erosionView, select(viewMode.equal(3), hardView, colorView)),
      ),
    )
    material.roughnessNode = mix(mix(float(0.92), float(0.25), clamp(wet.add(depth), 0, 1)), float(0.6), snow)

    const mesh = new THREE.Mesh(geometry, material)
    mesh.frustumCulled = false
    return mesh
  }

  setParams(params: Params) {
    this.apply(params, false)
  }

  private apply(p: Params, force: boolean) {
    this.params = p
    const requested = this.requestedSize(p)
    if (requested !== this.requested) {
      this.requested = requested
      void this.rebuild()
    }
    const { look } = this
    if (p.palette !== this.palette) {
      this.palette = p.palette
      look.setPalette(PALETTES[p.palette])
    }
    look.snowLine.value = p.snowLine
    look.snowSoftness.value = p.snowSoftness
    look.snowSlope.value = Math.cos((p.snowSlope * Math.PI) / 180)
    look.snowAspect.value = p.snowAspect
    look.sparkle.value = p.sparkle
    look.colorOn.value = p.colorTransport ? 1 : 0
    look.vivid.value = p.vividness
    this.view.value = ['shaded', 'water', 'erosion', 'hardness', 'color'].indexOf(p.view)
    // A rebuild re-applies everything (with a fresh terrain) once the new sim is ready.
    if (this.building) return
    this.sim.setTerrain(p)
    this.sim.setErosion(p)
    const key = terrainKey(p)
    if (force || key !== this.key) {
      this.key = key
      this.reset()
    }
  }

  /** Regenerate the terrain and start erosion over. */
  reset() {
    if (this.building) return
    this.sim.reset(this.renderer)
    this.steps = 0
    this.owed = 0
    this.onStats({ steps: 0 })
  }

  /** Run erosion steps without waiting for frames (for the OG card and tests). */
  fastForward(steps: number) {
    if (this.building) return
    for (let done = 0; done < steps; done += 50) this.sim.run(this.renderer, Math.min(50, steps - done))
    this.steps += steps
    this.onStats({ steps: this.steps })
  }

  /** Wait for the GPU to finish the queued work (reads back one cell). */
  async sync() {
    await this.renderer.getArrayBufferAsync(this.sim.state.value, null, 0, 16)
  }

  private fit = () => {
    const w = this.container.clientWidth || 1
    const h = this.container.clientHeight || 1
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  private frame = () => {
    const { params } = this
    if (this.building || !this.mesh) return
    if (params.eroding && params.iterations > 0) {
      this.owed += params.iterations * Math.min(1, FRAME_BUDGET / (this.sim.n * this.sim.n))
      const run = Math.floor(this.owed)
      if (run > 0) {
        this.owed -= run
        this.sim.run(this.renderer, run)
        this.steps += run
      }
      const now = performance.now()
      if (now - this.lastStats > 200) {
        this.lastStats = now
        this.onStats({ steps: this.steps })
      }
    }
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  dispose() {
    this.disposed = true
    void this.renderer.setAnimationLoop(null)
    this.resize.disconnect()
    this.controls.dispose()
    if (this.mesh) this.sim.dispose()
    void this.renderer.dispose()
    this.renderer.domElement.remove()
  }
}
