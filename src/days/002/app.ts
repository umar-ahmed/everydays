import * as THREE from 'three/webgpu'
import { PopSynth } from './audio.ts'
import { FOV, TABLE_GAP, fitCamera } from './camera.ts'
import { FilmLook } from './materials.ts'
import { MAX_RENDER_VERTS, RenderSurface, renderIndex } from './render.ts'
import { LatticeGenerator, type Shape, type Tiling } from './generators/lattice.ts'
import { buildSheetFromMap, layoutFor, type Sheet } from './sheet.ts'
import { BubbleSim } from './sim.ts'
import { StudioPipeline, studioEnvironment } from './studio.ts'

export type SheetParams = { shape: Shape; tiling: Tiling; size: number; spacing: number; cols: number; pinholeChance: number }

export type Params = SheetParams & {
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
  /** Crumple: amount, scale, crease sharpness, film-vs-dome balance, seed. */
  crumple: number
  crumpleScale: number
  crumpleSharpness: number
  crumpleBalance: number
  crumpleSeed: number
  /** Studio: strip-box and rim/kicker intensities, rotation of the whole rig (degrees). */
  keyStrip: number
  rimLight: number
  lightRotation: number
  background: 'black' | 'dark gray'
  /** Camera: exposure, depth of field 0..1, focus on the pointer, bloom, grain, vignette. */
  exposure: number
  dof: number
  focusFollows: boolean
  bloom: number
  grain: number
  vignette: number
  volume: number
  pitch: number
  reducedMotion: boolean
  simulation: 'running' | 'paused' | 'rest pose'
  view: 'shaded' | 'normals' | 'wireframe' | 'map'
  /** Render vertices per sim segment. */
  detail: 1 | 2 | 4
  texture: 'none' | 'grid' | 'dots'
  textureScale: number
  textureDepth: number
  textureSharpness: number
  tint: 'clear' | 'pink' | 'blue' | 'custom'
  tintColor: string
  density: number
}

/** Transmittance of one film pass for each tint preset (pink anti-static and blue PE). */
const TINTS = { clear: '#ffffff', pink: '#ffb0d6', blue: '#a8d0ff' } as const

export type Stats = { popped: number; total: number }

export async function webgpuAvailable() {
  if (!('gpu' in navigator) || !navigator.gpu) return false
  try {
    return (await navigator.gpu.requestAdapter()) !== null
  } catch {
    return false
  }
}

const sameSheet = (a: SheetParams, b: SheetParams) =>
  a.shape === b.shape &&
  a.tiling === b.tiling &&
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
  private sheet!: Sheet
  private params: Params
  private generator = new LatticeGenerator(this.sim.mapRaw)
  private builds = 0
  private key = new THREE.DirectionalLight(0xffffff, 1)
  private rim = new THREE.DirectionalLight(0xdfe9ff, 1)
  private fill = new THREE.DirectionalLight(0xfff3e6, 0.35)
  private ring: THREE.Mesh
  private table: THREE.Mesh | null = null
  private post: StudioPipeline | null = null
  private envTarget: THREE.RenderTarget | null = null
  private envKey = ''
  private envTimer = 0
  private focus = 20
  private mesh: THREE.Mesh | null = null
  private surface = new RenderSurface(this.sim)
  private look = new FilmLook(this.sim, this.surface)

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
    this.renderer = new THREE.WebGPURenderer({ antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    // AgX rolls the hard strip-light highlights off instead of clipping them flat.
    this.renderer.toneMapping = THREE.AgXToneMapping
    this.renderer.toneMappingExposure = 1
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
    scene.background = new THREE.Color(0, 0, 0)
    this.post = new StudioPipeline(this.renderer, scene, this.camera)

    // Subdivided so no single triangle reaches far behind the camera and gets clipped by the
    // near plane, which costs the clipped triangle depth interpolation precision.
    const table = new THREE.Mesh(new THREE.PlaneGeometry(600, 600, 60, 60).rotateX(-Math.PI / 2), this.look.table)
    table.position.y = -TABLE_GAP
    scene.add(table)
    this.table = table

    this.key.position.set(-7, 14, 9)
    this.rim.position.set(5, 4, -14)
    this.fill.position.set(10, 3, 6)
    scene.add(this.key, this.rim, this.fill)

    // Positions come from the render surface's storage buffer; this only sizes the vertex range.
    this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_RENDER_VERTS * 3), 3))
    const mesh = new THREE.Mesh(this.geometry, this.look.film)
    mesh.frustumCulled = false
    mesh.renderOrder = 1
    scene.add(mesh, this.ring)
    this.mesh = mesh

    await this.rebuild(this.params)
    if (this.disposed) return

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
    // The old sheet keeps running until the new map has been generated and read back.
    if (rebuild && this.sheet) void this.rebuild(params)
    if (this.sheet) this.apply(params)
  }

  /** Generate the bubble map on the GPU, read it back once, and build the sheet from it. */
  private async rebuild(params: Params) {
    const build = ++this.builds
    const layout = layoutFor(params)
    this.generator.configure(layout, params)
    this.renderer.compute(this.generator.compute)
    const buf = await this.renderer.getArrayBufferAsync(this.sim.mapRaw.value, null, 0, layout.mapW * layout.mapH * 16)
    // A newer dial change superseded this build while it was on the GPU.
    if (build !== this.builds || this.disposed) return
    const sheet = buildSheetFromMap({ layout, raw: new Float32Array(buf), pinholeChance: params.pinholeChance })
    this.sheet = sheet
    this.loadSheet(sheet)
    this.apply(this.params)
  }

  reset() {
    this.loadSheet(this.sheet)
    this.apply(this.params)
  }

  /** Re-rolled for every sheet (and New sheet), so each one crumples differently. */
  private sheetSeed = Math.random() * 1000

  private loadSheet(sheet: Sheet) {
    this.sheetSeed = Math.random() * 1000
    this.generation++
    this.sim.load(sheet, this.params.fill)
    const { look } = this
    look.lens.value = sheet.domeHeight
    look.contour.value = sheet.size * 0.06
    // Domes focus the key light onto the table on the side away from it.
    const away = new THREE.Vector2(-this.key.position.x, -this.key.position.z).normalize()
    look.spot.value.copy(away).multiplyScalar(sheet.domeHeight * 0.55)
    this.configureSurface()
    this.seen = new Uint8Array(sheet.count)
    // Rebuilds run inside React's synchronous flush of a dial change; reporting from there nests
    // a state update per slider step and trips React's update-depth limit under rapid input.
    const total = sheet.count
    queueMicrotask(() => this.onStats({ popped: 0, total }))
  }

  private detail = 0

  private configureSurface() {
    const detail = this.params.detail
    this.detail = detail
    const { RW, RH } = this.surface.configure(this.sheet, detail)
    const index = renderIndex(RW, RH)
    this.geometry.setIndex(new THREE.BufferAttribute(index, 1))
    this.geometry.setDrawRange(0, index.length)
  }

  private apply(p: Params) {
    if (p.detail !== this.detail) this.configureSurface()
    this.sim.setParams(p, this.sheet)
    const { look } = this
    look.thickness.value = p.thickness
    look.iridescence.value = p.iridescence
    look.haze.value = p.haze
    look.wrinkle.value = p.wrinkle
    look.wrinkleScale.value = p.wrinkleScale
    look.sharpness.value = p.sharpness
    // The table and its caustics only exist in the dark-gray comparison mode.
    const field = p.background === 'black'
    if (this.table) this.table.visible = !field
    this.scene.background = new THREE.Color(field ? 0x000000 : 0x1d1e21)
    look.caustics.value = field ? 0 : p.keyStrip / 3
    look.texture.value = ['none', 'grid', 'dots'].indexOf(p.texture)
    look.textureScale.value = p.textureScale
    look.textureDepth.value = p.textureDepth
    look.textureSharpness.value = p.textureSharpness
    look.bubbleSize.value = this.sheet.size
    const tint = new THREE.Color()
    try {
      tint.setStyle(p.tint === 'custom' ? p.tintColor : TINTS[p.tint])
    } catch {
      tint.set(0xffffff)
    }
    // The picked (sRGB) colour reads as the per-pass transmittance, so a light pink stays light.
    tint.convertLinearToSRGB()
    look.tint.value.set(tint.r, tint.g, tint.b)
    look.density.value = p.density
    look.crumple.value = p.crumple
    look.crumpleScale.value = p.crumpleScale
    look.crumpleSharpness.value = p.crumpleSharpness
    look.crumpleBalance.value = p.crumpleBalance
    look.crumpleSeed.value = p.crumpleSeed * 7.31 + this.sheetSeed
    // The strip boxes and kickers live in the environment; the punctual lights add the sharpest glints.
    // The env strips do the lighting; a point key on glossy film only makes hot dots.
    this.key.intensity = 0.05 * p.keyStrip
    this.fill.intensity = 0
    this.rim.intensity = p.rimLight
    const turn = THREE.MathUtils.degToRad(p.lightRotation)
    this.scene.environmentRotation.set(0, turn, 0)
    this.key.position.set(-7, 14, 9).applyAxisAngle(new THREE.Vector3(0, 1, 0), turn)
    this.rim.position.set(5, 4, -14).applyAxisAngle(new THREE.Vector3(0, 1, 0), turn)
    this.renderer.toneMappingExposure = p.exposure
    this.post?.configure({ dof: p.dof, bloom: p.bloom, grain: p.reducedMotion ? 0 : p.grain, vignette: p.vignette })
    this.scheduleEnvironment(p)
    this.synth.volume = p.volume
    this.synth.pitch = p.pitch
    look.restPose.value = p.simulation === 'rest pose' ? 1 : 0
    this.surface.restPose.value = look.restPose.value
    if (this.mesh) {
      const views = { shaded: look.film, normals: look.normals, wireframe: look.wireframe, map: look.mapView }
      this.mesh.material = views[p.view]
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

  /** Rebake the studio environment when its intensities change, debounced for slider drags. */
  private scheduleEnvironment(p: Params) {
    const key = `${p.keyStrip}/${p.rimLight}`
    if (key === this.envKey) return
    this.envKey = key
    window.clearTimeout(this.envTimer)
    const bake = () => {
      if (this.disposed) return
      const old = this.envTarget
      this.envTarget = studioEnvironment(this.renderer, { key: p.keyStrip, rim: p.rimLight })
      this.scene.environment = this.envTarget.texture
      old?.dispose()
    }
    if (!this.envTarget) bake()
    else this.envTimer = window.setTimeout(bake, 120)
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
    this.surface.update(this.renderer)
    // Focus where you look: the pointer's spot on the sheet, else the sheet's middle.
    const focusAt = params.focusFollows && onSheet ? this.hit : new THREE.Vector3(0, 0, 0)
    const forward = this.camera.getWorldDirection(new THREE.Vector3())
    const want = focusAt.clone().sub(this.camera.position).dot(forward)
    this.focus += (want - this.focus) * (params.reducedMotion ? 1 : 1 - Math.exp(-dt * 8))
    // Debug views show raw data, so they skip depth of field, bloom, grain and the vignette.
    if (this.post && params.view === 'shaded') {
      this.post.focus.value = this.focus
      // Depth of field in world units: about a third of the sheet's depth stays sharp.
      this.post.focalLength.value = sheet.depth * 0.35
      this.post.render()
    } else {
      this.renderer.render(this.scene, this.camera)
    }
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
        const total = sheet.count
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
          const [x, , , diameter] = sheet.centers.subarray(b * 4, b * 4 + 4)
          this.synth.play({ pressure: pops[b * 4 + 1], size: diameter, pan: x / (sheet.width / 2) })
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
    window.clearTimeout(this.envTimer)
    this.envTarget?.dispose()
    this.post?.dispose()
    this.renderer.dispose()
    canvas.remove()
  }
}

export type { Shape, Tiling }
