import { type CameraFit, type Pose, type SheetBox, type View, rayHit, zoomRange } from './camera.ts'

const DEG = Math.PI / 180
const MIN_ELEVATION = 12 * DEG
const MAX_ELEVATION = 85 * DEG
const MAX_AZIMUTH = 70 * DEG

/** Offsets from the auto-fit framing: pan (cm), log zoom (log of dist / fit dist), orbit (radians). */
type Channels = { x: number; z: number; zoom: number; az: number; el: number }
const KEYS = ['x', 'z', 'zoom', 'az', 'el'] as const
const zero = (): Channels => ({ x: 0, z: 0, zoom: 0, az: 0, el: 0 })

type Pointer = { x: number; y: number; downX: number; downY: number; downT: number }
type Pinch = { mid: [number, number]; dist: number; angle: number }

/**
 * View mode: pan, zoom toward the cursor or pinch, and a gentle orbit, all as targets that a
 * critically damped spring follows. New input retargets the spring without resetting its
 * velocity, so a gesture can interrupt a reset (or another gesture) without a jump.
 */
export class ViewController {
  reducedMotion = false
  private target = zero()
  private value = zero()
  private velocity = zero()
  /** Idle sway amount: fades out once the camera has been moved, back in after a reset. */
  private swayAmount = 1
  /** Reset springs back more slowly than direct manipulation follows the fingers. */
  private soft = false

  private pointers = new Map<number, Pointer>()
  private gesture: 'pan' | 'orbit' | 'pinch' | null = null
  private grab: [number, number, number] | null = null
  private pinch: Pinch | null = null
  private lastTap = { t: -1e9, x: 0, y: 0 }

  private fit: CameraFit | null = null
  private box: SheetBox = { width: 1, depth: 1, top: 0 }
  private view: View = { width: 1, height: 1 }
  private size = 1
  private domeHeight = 0

  get moved() {
    return KEYS.some((k) => Math.abs(this.target[k]) > 1e-4)
  }

  get dragging() {
    return this.gesture !== null
  }

  /** Called every frame with the current framing; keeps the targets inside the limits. */
  setContext(fit: CameraFit, box: SheetBox, view: View, size: number, domeHeight: number) {
    this.fit = fit
    this.box = box
    this.view = view
    this.size = size
    this.domeHeight = domeHeight
    this.clamp(this.target)
  }

  /** Back to the auto-fit framing (springs unless `instant` or reduced motion). */
  reset(instant = false) {
    this.target = zero()
    this.soft = true
    if (instant) {
      this.value = zero()
      this.velocity = zero()
    }
  }

  /** Drop any gesture in progress (mode switch, pointer lost). */
  cancel() {
    this.pointers.clear()
    this.gesture = null
    this.grab = null
    this.pinch = null
  }

  /** Steps the springs; returns the pose to render and how much idle sway to add. */
  update(dt: number): { pose: Pose; sway: number } {
    const omega = this.soft ? 7 : 16
    if (this.reducedMotion) {
      this.value = { ...this.target }
      this.velocity = zero()
    } else {
      const steps = Math.max(1, Math.ceil(dt / (1 / 240)))
      const h = dt / steps
      for (let s = 0; s < steps; s++) {
        for (const k of KEYS) {
          const a = omega * omega * (this.target[k] - this.value[k]) - 2 * omega * this.velocity[k]
          this.velocity[k] += a * h
          this.value[k] += this.velocity[k] * h
        }
      }
    }
    const want = this.moved ? 0 : 1
    this.swayAmount += (want - this.swayAmount) * (1 - Math.exp(-dt * 3))
    return { pose: this.pose(this.value), sway: this.swayAmount }
  }

  /** Current zoom as dist / fit dist (for depth of field). */
  get zoom() {
    return Math.exp(this.value.zoom)
  }

  pose(c: Channels = this.target): Pose {
    const fit = this.fit!
    return {
      target: [fit.target[0] + c.x, fit.target[1], fit.target[2] + c.z],
      dist: fit.dist * Math.exp(c.zoom),
      elevation: fit.elevation + c.el,
      azimuth: fit.azimuth + c.az,
    }
  }

  down(id: number, x: number, y: number, orbit: boolean, time: number) {
    if (!this.fit) return
    this.soft = false
    this.pointers.set(id, { x, y, downX: x, downY: y, downT: time })
    if (this.pointers.size === 1) {
      if (time - this.lastTap.t < 320 && Math.hypot(x - this.lastTap.x, y - this.lastTap.y) < 28) {
        this.lastTap.t = -1e9
        this.reset()
        this.gesture = null
        return
      }
      this.gesture = orbit ? 'orbit' : 'pan'
      this.grab = this.hit(this.ndc(x, y))
    } else if (this.pointers.size === 2) {
      this.gesture = 'pinch'
      this.pinch = this.measure()
    }
  }

  move(id: number, x: number, y: number) {
    const p = this.pointers.get(id)
    if (!p || !this.fit) return
    const dx = x - p.x
    const dy = y - p.y
    p.x = x
    p.y = y
    if (this.gesture === 'pan' && this.grab) {
      const hit = this.hit(this.ndc(x, y))
      if (hit) {
        this.target.x += this.grab[0] - hit[0]
        this.target.z += this.grab[2] - hit[2]
      }
    } else if (this.gesture === 'orbit') {
      this.target.az -= dx * 0.005
      this.target.el += dy * 0.004
    } else if (this.gesture === 'pinch' && this.pinch) {
      const now = this.measure()
      this.zoomAt(this.pinch.mid, this.pinch.dist / Math.max(now.dist, 1))
      // Then carry the point under the old midpoint to the new one (two-finger pan).
      const from = this.hit(this.pinch.mid)
      const to = this.hit(now.mid)
      if (from && to) {
        this.target.x += from[0] - to[0]
        this.target.z += from[2] - to[2]
      }
      let twist = now.angle - this.pinch.angle
      if (twist > Math.PI) twist -= 2 * Math.PI
      if (twist < -Math.PI) twist += 2 * Math.PI
      this.target.az += twist
      this.pinch = now
    }
    this.clamp(this.target)
  }

  up(id: number, time: number) {
    const p = this.pointers.get(id)
    if (!p) return
    this.pointers.delete(id)
    const still = Math.hypot(p.x - p.downX, p.y - p.downY) < 8 && time - p.downT < 260
    if (still && this.pointers.size === 0) this.lastTap = { t: time, x: p.x, y: p.y }
    if (this.pointers.size === 1) {
      // Lifting one finger of a pinch continues as a one-finger pan from where the other is.
      const [rest] = this.pointers.values()
      this.gesture = 'pan'
      this.grab = this.hit(this.ndc(rest.x, rest.y))
      this.pinch = null
    } else if (this.pointers.size === 0) {
      this.cancel()
    }
  }

  /** Wheel or trackpad pinch (ctrlKey) zooms toward the cursor. */
  wheel(x: number, y: number, deltaY: number, pinch: boolean) {
    if (!this.fit) return
    this.soft = false
    this.zoomAt(this.ndc(x, y), Math.exp(deltaY * (pinch ? 0.01 : 0.0015)))
    this.clamp(this.target)
  }

  private zoomAt(ndc: [number, number], factor: number) {
    const before = this.hit(ndc)
    this.target.zoom += Math.log(factor)
    this.clamp(this.target)
    const after = this.hit(ndc)
    if (before && after) {
      this.target.x += before[0] - after[0]
      this.target.z += before[2] - after[2]
    }
  }

  private clamp(c: Channels) {
    const fit = this.fit
    if (!fit) return
    c.el = Math.min(Math.max(c.el, MIN_ELEVATION - fit.elevation), MAX_ELEVATION - fit.elevation)
    c.az = Math.min(Math.max(c.az, -MAX_AZIMUTH), MAX_AZIMUTH)
    const range = zoomRange(fit, this.size, this.domeHeight, this.view)
    // Never closer than the dome tops allow at the current tilt.
    const clear = (this.domeHeight * 1.6 + this.size * 0.15) / Math.sin(fit.elevation + c.el)
    const min = Math.min(Math.max(range.min, clear), range.max)
    c.zoom = Math.min(Math.max(c.zoom, Math.log(min / fit.dist)), Math.log(range.max / fit.dist))
    c.x = Math.min(Math.max(c.x, -this.box.width / 2), this.box.width / 2)
    c.z = Math.min(Math.max(c.z, -this.box.depth / 2), this.box.depth / 2)
  }

  private ndc(x: number, y: number): [number, number] {
    return [(x / this.view.width) * 2 - 1, 1 - (y / this.view.height) * 2]
  }

  private hit(ndc: [number, number]) {
    return rayHit(this.pose(this.target), this.fit!.offset, this.view, ndc, this.domeHeight * 0.5)
  }

  private measure(): Pinch {
    const [a, b] = [...this.pointers.values()]
    return {
      mid: this.ndc((a.x + b.x) / 2, (a.y + b.y) / 2),
      dist: Math.hypot(b.x - a.x, b.y - a.y),
      angle: Math.atan2(b.y - a.y, b.x - a.x),
    }
  }
}
