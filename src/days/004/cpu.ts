import type { Display, Tracer } from './gpu.ts'
import {
  LAMBDA_MAX,
  LAMBDA_MIN,
  compile,
  spectralRgb,
  tracePath,
  viewFor,
  type Compiled,
  type Optics,
  type Scene,
} from './optics.ts'

/** Must match the GPU tracer so both look alike. */
const BRIGHTNESS = 0.05
/** Milliseconds of tracing per frame; the rest is left for the page. */
const BUDGET = 9

const neutral = (c: [number, number, number]) => {
  const start = 0.76
  const x = Math.min(c[0], c[1], c[2])
  const offset = x < 0.08 ? x - 6.25 * x * x : 0.04
  let [r, g, b] = [c[0] - offset, c[1] - offset, c[2] - offset]
  const peak = Math.max(r, g, b)
  if (peak < start) return [r, g, b]
  const d = 1 - start
  const top = 1 - (d * d) / (peak + d - start)
  ;[r, g, b] = [(r * top) / peak, (g * top) / peak, (b * top) / peak]
  const k = 1 - 1 / (0.15 * (peak - top) + 1)
  return [r + (top - r) * k, g + (top - g) * k, b + (top - b) * k]
}

const encode = (v: number) => {
  const c = Math.min(Math.max(v, 0), 1)
  return 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055)
}

/**
 * The same light tracer on the CPU, drawn into a canvas at reduced resolution, for browsers
 * without WebGPU. Slower and noisier, but it is the same physics.
 */
export class CpuTracer implements Tracer {
  readonly maxPaths = 4000
  paths = 0
  paused = false
  private canvas = document.createElement('canvas')
  private ctx: CanvasRenderingContext2D
  private image: ImageData | null = null
  private acc = new Float32Array(0)
  private width = 1
  private height = 1
  private compiled: Compiled | null = null
  private scene: Scene | null = null
  private optics: Optics | null = null
  private exposure = 1
  private raf = 0
  private resize: ResizeObserver
  private container: HTMLElement
  private onStats: (paths: number) => void

  constructor(container: HTMLElement, onStats: (paths: number) => void) {
    this.container = container
    this.onStats = onStats
    this.ctx = this.canvas.getContext('2d')!
    Object.assign(this.canvas.style, { display: 'block', width: '100%', height: '100%' })
    container.appendChild(this.canvas)
    this.resize = new ResizeObserver(() => this.fit())
    this.resize.observe(container)
    this.fit()
    this.raf = requestAnimationFrame(this.frame)
  }

  private fit() {
    this.width = Math.max(1, Math.round(this.container.clientWidth / 2))
    this.height = Math.max(1, Math.round(this.container.clientHeight / 2))
    this.canvas.width = this.width
    this.canvas.height = this.height
    this.image = this.ctx.createImageData(this.width, this.height)
    this.acc = new Float32Array(this.width * this.height * 3)
    this.paths = 0
  }

  setScene(scene: Scene, optics: Optics) {
    this.scene = scene
    this.optics = optics
    this.compiled = compile(scene.shapes)
    this.acc.fill(0)
    this.paths = 0
  }

  setDisplay(d: Display) {
    this.exposure = d.exposure
  }

  private frame = () => {
    this.raf = requestAnimationFrame(this.frame)
    const { scene, optics, compiled: c, width: W, height: H, acc } = this
    if (!scene || !optics || !c || this.paused) return
    const view = viewFor(W / H)
    const sx = W / (2 * view.hw)
    const sy = H / (2 * view.hh)
    const until = performance.now() + BUDGET
    let traced = 0
    while (traced < this.maxPaths && (traced & 63 || performance.now() < until)) {
      const lambda = LAMBDA_MIN + Math.random() * (LAMBDA_MAX - LAMBDA_MIN)
      const rgb = spectralRgb(lambda)
      for (const s of tracePath(c, view, scene, optics, lambda, Math.random)) {
        const x0 = (s.x0 + view.hw) * sx
        const y0 = (s.y0 + view.hh) * sy
        const dx = (s.x1 + view.hw) * sx - x0
        const dy = (s.y1 + view.hh) * sy - y0
        const steps = Math.ceil(Math.max(Math.abs(dx), Math.abs(dy)))
        if (!steps) continue
        const w = (Math.hypot(dx, dy) / steps) * s.throughput
        for (let k = 0; k < steps; k++) {
          const px = Math.floor(x0 + (dx * (k + Math.random())) / steps)
          const py = Math.floor(y0 + (dy * (k + Math.random())) / steps)
          if (px < 0 || py < 0 || px >= W || py >= H) continue
          const i = (py * W + px) * 3
          acc[i] += rgb[0] * w
          acc[i + 1] += rgb[1] * w
          acc[i + 2] += rgb[2] * w
        }
      }
      traced++
    }
    this.paths += traced
    const scale = (this.exposure * optics.power * BRIGHTNESS * H) / Math.max(1, this.paths)
    const data = this.image!.data
    for (let p = 0, i = 0; p < W * H; p++, i += 3) {
      const t = neutral([
        Math.max(acc[i] * scale, 0) + 0.004,
        Math.max(acc[i + 1] * scale, 0) + 0.0045,
        Math.max(acc[i + 2] * scale, 0) + 0.008,
      ])
      data[p * 4] = encode(t[0])
      data[p * 4 + 1] = encode(t[1])
      data[p * 4 + 2] = encode(t[2])
      data[p * 4 + 3] = 255
    }
    this.ctx.putImageData(this.image!, 0, 0)
    this.onStats(this.paths)
  }

  dispose() {
    cancelAnimationFrame(this.raf)
    this.resize.disconnect()
    this.canvas.remove()
  }
}
