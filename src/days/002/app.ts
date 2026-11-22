import * as THREE from 'three/webgpu'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { PopSynth } from './audio.ts'
import { FOV, TABLE_GAP, fitCamera } from './camera.ts'
import { FilmLook } from './materials.ts'
import { buildSheet, MAX_VERTS, type Sheet, type SheetConfig } from './sheet.ts'
import type { Shape } from './shapes.ts'
import { BubbleSim } from './sim.ts'

export type Params = SheetConfig & {
  fill: number
  threshold: number
  stiffness: number
  substeps: number
  tool: 'finger' | 'roller'
  pressRadius: number
  pressStrength: number
  thickness: number
  iridescence: number
  haze: number
  wrinkle: number
  wrinkleScale: number
  sharpness: number
  light: number
  rim: number
  volume: number
  pitch: number
  reducedMotion: boolean
  simulation: 'running' | 'paused' | 'rest pose'
  view: 'shaded' | 'normals' | 'wireframe'
}

export type Stats = { popped: number; total: number }

export async function webgpuAvailable() {
  if (!('gpu' in navigator) || !navigator.gpu) return false
  try {
    return (await navigator.gpu.requestAdapter()) !== null
  } catch {
    return false
  }
}

const sameSheet = (a: SheetConfig, b: SheetConfig) =>
  a.shape === b.shape &&
  a.size === b.size &&
  a.spacing === b.spacing &&
  a.cols === b.cols &&
  a.pinholeChance === b.pinholeChance

export class BubbleWrapApp {
  readonly synth = new PopSynth()
  private renderer: THREE.WebGPURenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 500)
  private sim = new BubbleSim()
  private geometry = new THREE.BufferGeometry()
  private sheet: Sheet
  private params: Params
  private key = new THREE.DirectionalLight(0xffffff, 1)
  private rim = new THREE.DirectionalLight(0xdfe9ff, 1)
  private fill = new THREE.DirectionalLight(0xfff3e6, 0.35)
  private ring: THREE.Mesh
  private mesh: THREE.Mesh | null = null
  private look = new FilmLook(this.sim)

  private pointer = new THREE.Vector2()
  private hovering = false
  private pressed = false
  private fingerY = 10
  private raycaster = new THREE.Raycaster()
  private plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
  private hit = new THREE.Vector3()

  private seen = new Uint8Array(0)
  private generation = 0
  private readback: Promise<void> | null = null
  private elapsed = 0
  private last = performance.now()
  private disposed = false
  private resize = new ResizeObserver(() => this.fit())

  private container: HTMLElement
  private onStats: (s: Stats) => void

  constructor(container: HTMLElement, params: Params, onStats: (s: Stats) => void) {
    this.container = container
    this.onStats = onStats
    this.params = params
    this.sheet = buildSheet(params)
    this.renderer = new THREE.WebGPURenderer({ antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.toneMapping = THREE.NeutralToneMapping
    this.renderer.toneMappingExposure = 1.05
    const canvas = this.renderer.domElement
    canvas.style.display = 'block'
    canvas.style.width = '100%'
    canvas.style.height = '100%'
    canvas.style.touchAction = 'none'
    container.appendChild(canvas)

    this.ring = new THREE.Mesh(
      new THREE.RingGeometry(0.9, 1, 64).rotateX(-Math.PI / 2),
      new THREE.MeshBasicNodeMaterial({ color: 0xffffff, transparent: true, opacity: 0.28, depthWrite: false }),
    )
    this.ring.visible = false
  }

  async init() {
    await this.renderer.init()
    if (!(this.renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend) throw new Error('webgpu-unavailable')
    if (this.disposed) return

    const { scene } = this
    scene.background = new THREE.Color(0x1d1e21)
    const pmrem = new THREE.PMREMGenerator(this.renderer)
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture

    // Subdivided so no single triangle reaches far behind the camera and gets clipped by the
    // near plane, which costs the clipped triangle depth interpolation precision.
    const table = new THREE.Mesh(new THREE.PlaneGeometry(600, 600, 60, 60).rotateX(-Math.PI / 2), this.look.table)
    table.position.y = -TABLE_GAP
    scene.add(table)

    this.key.position.set(-7, 14, 9)
    this.rim.position.set(5, 4, -14)
    this.fill.position.set(10, 3, 6)
    scene.add(this.key, this.rim, this.fill)

    this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_VERTS * 3), 3))
    const mesh = new THREE.Mesh(this.geometry, this.look.film)
    mesh.frustumCulled = false
    mesh.renderOrder = 1
    scene.add(mesh, this.ring)
    this.mesh = mesh

    this.loadSheet(this.sheet)
    this.apply(this.params)

    const canvas = this.renderer.domElement
    canvas.addEventListener('pointerdown', this.onDown)
    canvas.addEventListener('pointermove', this.onMove)
    canvas.addEventListener('pointerleave', this.onLeave)
    window.addEventListener('pointerup', this.onUp)
    window.addEventListener('pointercancel', this.onUp)
    this.resize.observe(this.container)
    this.fit()
    this.renderer.setAnimationLoop(this.frame)
  }

  setParams(params: Params) {
    const rebuild = !sameSheet(params, this.params)
    this.params = params
    if (rebuild) {
      this.sheet = buildSheet(params)
      this.loadSheet(this.sheet)
      this.fit()
    }
    this.apply(params)
  }

  reset() {
    this.loadSheet(this.sheet)
  }

  private loadSheet(sheet: Sheet) {
    this.generation++
    this.sim.load(sheet, this.params.fill)
    const { look } = this
    look.cell.value = sheet.width / sheet.cols
    look.cols.value = sheet.cols
    look.rows.value = sheet.rows
    look.half.value.set(sheet.width / 2, sheet.depth / 2)
    look.domeR.value = sheet.size / 2
    look.lens.value = sheet.domeHeight
    // Domes focus the key light onto the table on the side away from it.
    const away = new THREE.Vector2(-this.key.position.x, -this.key.position.z).normalize()
    look.spot.value.copy(away).multiplyScalar(sheet.domeHeight * 0.55)
    this.geometry.setIndex(new THREE.BufferAttribute(sheet.index, 1))
    this.geometry.setDrawRange(0, sheet.index.length)
    this.seen = new Uint8Array(sheet.cols * sheet.rows)
    this.onStats({ popped: 0, total: sheet.cols * sheet.rows })
  }

  private apply(p: Params) {
    this.sim.setParams(p, this.sheet)
    const { look } = this
    look.thickness.value = p.thickness
    look.iridescence.value = p.iridescence
    look.haze.value = p.haze
    look.wrinkle.value = p.wrinkle
    look.wrinkleScale.value = p.wrinkleScale
    look.sharpness.value = p.sharpness
    look.caustics.value = p.light
    this.key.intensity = 2.2 * p.light
    this.fill.intensity = 0.5 * p.light
    this.rim.intensity = p.rim
    this.scene.environmentIntensity = 0.9 * p.light
    this.synth.volume = p.volume
    this.synth.pitch = p.pitch
    look.restPose.value = p.simulation === 'rest pose' ? 1 : 0
    if (this.mesh) {
      this.mesh.material = p.view === 'normals' ? look.normals : p.view === 'wireframe' ? look.wireframe : look.film
    }
  }

  private fit = () => {
    const w = this.container.clientWidth || 1
    const h = this.container.clientHeight || 1
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  private placeCamera(t: number) {
    const fit = fitCamera(this.sheet.width, this.sheet.depth, this.camera.aspect)
    if (this.camera.near !== fit.near || this.camera.far !== fit.far) {
      this.camera.near = fit.near
      this.camera.far = fit.far
      this.camera.updateProjectionMatrix()
    }
    const sway = this.params.reducedMotion ? 0 : 1
    const az = THREE.MathUtils.degToRad(2.2 * Math.sin(t * 0.21)) * sway
    const el = fit.elevation + THREE.MathUtils.degToRad(1.4 * Math.sin(t * 0.13)) * sway
    const { dist } = fit
    this.camera.position.set(Math.sin(az) * Math.cos(el) * dist, Math.sin(el) * dist, Math.cos(az) * Math.cos(el) * dist)
    this.camera.lookAt(...fit.target)
  }

  private onDown = (e: PointerEvent) => {
    this.synth.unlock()
    this.renderer.domElement.setPointerCapture(e.pointerId)
    this.setPointer(e)
    this.pressed = true
  }
  private onMove = (e: PointerEvent) => this.setPointer(e)
  private onUp = () => {
    this.pressed = false
  }
  private onLeave = (e: PointerEvent) => {
    if (e.pointerType === 'mouse' && !this.pressed) this.hovering = false
  }
  private setPointer(e: PointerEvent) {
    const r = this.renderer.domElement.getBoundingClientRect()
    this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    this.hovering = true
  }

  private frame = () => {
    const now = performance.now()
    // Capped at 1/60 s: below 60 fps the sheet slows down rather than going soft, since
    // the solver's stiffness depends on the substep length.
    const dt = Math.min(Math.max((now - this.last) / 1000, 1 / 240), 1 / 60)
    this.last = now
    this.elapsed += dt
    const { sheet, params, sim } = this
    // Paused and rest pose skip every dispatch and readback. The clock above still ticks
    // each frame, so going back to running starts from a normal-sized step.
    const live = params.simulation === 'running'

    this.placeCamera(this.elapsed)
    this.camera.updateMatrixWorld()

    const r = params.pressRadius * sheet.size
    this.plane.constant = -sheet.domeHeight * 0.5
    this.raycaster.setFromCamera(this.pointer, this.camera)
    const onSheet = this.hovering && this.raycaster.ray.intersectPlane(this.plane, this.hit) !== null
    // A pressing finger: the bottom of the sphere sinks to a depth set by Press strength.
    const bottom = Math.max(sheet.domeHeight * (1 - params.pressStrength), 0.01)
    const pressing = this.pressed && onSheet && live
    const target = pressing ? bottom + r : sheet.domeHeight + r + 1
    this.fingerY += (target - this.fingerY) * (1 - Math.exp(-dt * (pressing ? 14 : 30)))
    sim.finger.value.set(this.hit.x, this.fingerY, this.hit.z)
    sim.time.value = this.elapsed

    this.ring.visible = onSheet && live
    this.ring.position.set(this.hit.x, sheet.domeHeight + 0.04, this.hit.z)
    this.ring.scale.set(params.tool === 'roller' ? sim.rollerHalf.value : r, 1, r)

    if (live) sim.step(this.renderer, dt, params.substeps)
    else this.synth.hiss(0)
    this.renderer.render(this.scene, this.camera)
    if (live) this.pollPops()
  }

  private pollPops() {
    if (this.readback) return
    const gen = this.generation
    const { sheet, sim } = this
    this.readback = Promise.all([
      this.renderer.getArrayBufferAsync(sim.pops.value),
      this.renderer.getArrayBufferAsync(sim.state.value),
    ])
      .then(([popBuf, stateBuf]) => {
        if (this.disposed || gen !== this.generation) return
        const pops = new Float32Array(popBuf)
        const state = new Float32Array(stateBuf)
        const total = sheet.cols * sheet.rows
        const cellSize = sheet.width / sheet.cols
        let popped = 0
        let fresh = 0
        let venting = 0
        for (let b = 0; b < total; b++) {
          if (state[b * 4 + 3] > 0.5 && pops[b * 4] < 0.5) {
            venting += Math.max(state[b * 4] - this.params.fill * 1.2, 0)
          }
          if (pops[b * 4] < 0.5) continue
          popped++
          if (this.seen[b]) continue
          this.seen[b] = 1
          fresh++
          const x = ((b % sheet.cols) + 0.5) * cellSize - sheet.width / 2
          this.synth.play({ pressure: pops[b * 4 + 1], size: sheet.size, pan: x / (sheet.width / 2) })
        }
        this.synth.hiss(Math.sqrt(venting) * 1.4)
        if (fresh) this.onStats({ popped, total })
      })
      .catch(() => {})
      .finally(() => {
        this.readback = null
      })
  }

  dispose() {
    this.disposed = true
    this.renderer.setAnimationLoop(null)
    this.resize.disconnect()
    const canvas = this.renderer.domElement
    canvas.removeEventListener('pointerdown', this.onDown)
    canvas.removeEventListener('pointermove', this.onMove)
    canvas.removeEventListener('pointerleave', this.onLeave)
    window.removeEventListener('pointerup', this.onUp)
    window.removeEventListener('pointercancel', this.onUp)
    this.synth.dispose()
    this.renderer.dispose()
    canvas.remove()
  }
}

export type { Shape }
