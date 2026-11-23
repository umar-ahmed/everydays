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
  uniform,
  varying,
  vec3,
  vertexIndex,
} from 'three/tsl'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import * as THREE from 'three/webgpu'
import type { Node } from 'three/webgpu'
import { Look, PALETTES, type PaletteName } from './palettes.ts'
import { ErosionSim, SIZE, defaultResolution, maxResolution, type ErosionParams, type TerrainParams } from './sim.ts'

export type Params = TerrainParams &
  ErosionParams & {
    eroding: boolean
    iterations: number
    view: 'shaded' | 'water' | 'erosion'
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

/** Phones get a smaller default grid and stop at 512 (see sim.ts). */
export const compactDevice = () => typeof screen !== 'undefined' && Math.min(screen.width, screen.height) < 600

/** Direction toward the sun (the key light sits at -6, 5, 3). */
const SUN = new THREE.Vector3(-6, 5, 3).normalize()

export type Stats = { steps: number }

export async function webgpuAvailable() {
  if (!('gpu' in navigator) || !navigator.gpu) return false
  try {
    return (await navigator.gpu.requestAdapter()) !== null
  } catch {
    return false
  }
}

const terrainKey = (p: TerrainParams) => `${p.seed}/${p.octaves}/${p.roughness}/${p.ridge}/${p.heightScale}`

export class TerrainApp {
  private renderer: THREE.WebGPURenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200)
  private controls: OrbitControls
  sim: ErosionSim
  private mesh: THREE.Mesh | null = null
  private look = new Look()
  private palette = ''
  private params: Params
  private key = ''
  private view = uniform(0, 'int')
  private steps = 0
  private disposed = false
  private resize = new ResizeObserver(() => this.fit())
  private container: HTMLElement
  private onStats: (s: Stats) => void
  private lastStats = 0

  constructor(container: HTMLElement, params: Params, onStats: (s: Stats) => void) {
    this.container = container
    this.params = params
    this.onStats = onStats
    this.sim = new ErosionSim(this.resolutionFor(params.resolution))
    this.renderer = new THREE.WebGPURenderer({ antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
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

    this.mesh = this.makeTerrain()
    scene.add(this.mesh)
    this.apply(this.params, true)
    this.resize.observe(this.container)
    this.fit()
    await this.renderer.setAnimationLoop(this.frame)
  }

  private makeTerrain() {
    const { sim } = this
    const n = sim.n
    // Positions come from the sim; the attribute only sizes the vertex range.
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(n * n * 3), 3))
    const index = new Uint32Array((n - 1) * (n - 1) * 6)
    let k = 0
    for (let y = 0; y < n - 1; y++) {
      for (let x = 0; x < n - 1; x++) {
        const a = y * n + x
        index.set([a, a + n, a + 1, a + 1, a + n, a + n + 1], k)
        k += 6
      }
    }
    geometry.setIndex(new THREE.BufferAttribute(index, 1))

    const N = int(n)
    const i = int(vertexIndex)
    const x = i.mod(N)
    const y = i.div(N)
    const iclamp = (v: typeof x) => select(v.lessThan(0), int(0), select(v.greaterThan(N.sub(1)), N.sub(1), v))
    const at = (dx: number, dy: number) =>
      sim.state.element(
        iclamp(y.add(dy))
          .mul(N)
          .add(iclamp(x.add(dx))),
      )
    const s = sim.state.element(i)
    const cell = sim.cell
    const world = (u: Node<'float'>) => u.mul(cell)
    const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.9 })
    material.positionNode = vec3(
      float(x)
        .mul(cell)
        .sub(SIZE / 2),
      world(s.x.add(s.y.mul(0.6))),
      float(y)
        .mul(cell)
        .sub(SIZE / 2),
    )
    // Normal of the terrain surface (not the water) from central differences, in world units.
    const gx = at(1, 0).x.sub(at(-1, 0).x).mul(cell)
    const gz = at(0, 1).x.sub(at(0, -1).x).mul(cell)
    const normal = varying(normalize(vec3(gx.negate(), cell * 2, gz.negate())))
    material.normalNode = transformNormalToView(normal)
    const height = varying(s.x)
    const water = varying(s.y)
    const change = varying(s.x.sub(s.w))

    const { look } = this
    // Slope: 1 flat, 0 vertical.
    const flat = normal.y
    const steep = smoothstep(0.86, 0.6, flat)
    const relief = height.div(max(sim.heightCells, 1))
    const t = clamp(relief, 0, 1)
    // Height ramp for gentle ground: four stops at 0, 0.3, 0.6 and 1 of the peak height.
    const ramp = mix(
      mix(mix(look.ramp[0], look.ramp[1], smoothstep(0, 0.3, t)), look.ramp[2], smoothstep(0.3, 0.6, t)),
      look.ramp[3],
      smoothstep(0.6, 1, t),
    )
    // Contour lines every 5% of the peak height, antialiased in screen space.
    const iso = relief.mul(20)
    const isoDist = min(fract(iso), float(1).sub(fract(iso))).div(max(fwidth(iso), 1e-4))
    const ground = ramp.mul(float(1).sub(look.contours.mul(float(1).sub(clamp(isoDist, 0, 1)))))
    // Exposed rock shows strata: bands in height, warmer where the water cut through them.
    const band = smoothstep(0.3, 0.7, fract(positionWorld.y.mul(5.5)))
      .sub(0.5)
      .mul(look.strata)
    const rock = mix(look.rock, look.rockCut, smoothstep(-0.3, -3, change)).add(band.mul(steep))
    const deposited = smoothstep(0.02, 0.35, change)
    let albedo = mix(ground, rock, steep.mul(look.rockiness))
    albedo = mix(albedo, look.sand, deposited.mul(float(1).sub(steep.mul(0.7))))
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
    // Debug views: water depth, and terrain change (blue cut, orange filled).
    const waterView = mix(vec3(0.08), vec3(0.2, 0.6, 1), clamp(water.mul(2), 0, 1))
    const erosionView = select(
      change.lessThan(0),
      mix(vec3(0.85), vec3(0.15, 0.35, 0.9), clamp(change.negate().div(3), 0, 1)),
      mix(vec3(0.85), vec3(0.95, 0.55, 0.15), clamp(change.div(1.5), 0, 1)),
    )
    const viewMode = this.view
    material.colorNode = select(viewMode.equal(0), shaded, select(viewMode.equal(1), waterView, erosionView))
    material.roughnessNode = mix(mix(float(0.92), float(0.25), clamp(wet.add(depth), 0, 1)), float(0.6), snow)

    const mesh = new THREE.Mesh(geometry, material)
    mesh.frustumCulled = false
    return mesh
  }

  setParams(params: Params) {
    this.params = params
    this.apply(params, false)
  }

  private resolutionFor(requested: number) {
    const compact = compactDevice()
    return Math.min(requested || defaultResolution(compact), maxResolution(compact))
  }

  /** A new sim and mesh at another grid size; the noise is in sheet coordinates, so the same
   * mountains come back, just finer or coarser. */
  private rebuild(n: number) {
    if (this.mesh) {
      this.scene.remove(this.mesh)
      this.mesh.geometry.dispose()
      ;(this.mesh.material as THREE.Material).dispose()
    }
    this.sim = new ErosionSim(n)
    this.mesh = this.makeTerrain()
    this.scene.add(this.mesh)
  }

  private apply(p: Params, force: boolean) {
    const n = this.resolutionFor(p.resolution)
    if (n !== this.sim.n) {
      this.rebuild(n)
      force = true
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
    this.sim.setTerrain(p)
    this.sim.setErosion(p)
    this.view.value = ['shaded', 'water', 'erosion'].indexOf(p.view)
    const key = terrainKey(p)
    if (force || key !== this.key) {
      this.key = key
      this.reset()
    }
  }

  /** Regenerate the terrain and start erosion over. */
  reset() {
    this.sim.reset(this.renderer)
    this.steps = 0
    this.onStats({ steps: 0 })
  }

  /** Run erosion steps without waiting for frames (for the OG card and tests). */
  fastForward(steps: number) {
    for (let done = 0; done < steps; done += 50) this.sim.run(this.renderer, Math.min(50, steps - done))
    this.steps += steps
    this.onStats({ steps: this.steps })
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
    if (params.eroding && params.iterations > 0) {
      this.sim.run(this.renderer, params.iterations)
      this.steps += params.iterations
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
    void this.renderer.dispose()
    this.renderer.domElement.remove()
  }
}
