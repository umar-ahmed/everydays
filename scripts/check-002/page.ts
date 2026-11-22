// Runs inside headless Chromium (WebGPU on SwiftShader): the real lattice generator fills the
// bubble map, the sheet is built from its readback, and every invariant is checked.
import { instancedArray } from 'three/tsl'
import * as THREE from 'three/webgpu'
import { TABLE_GAP, depthStep, fitCamera } from '../../src/days/002/camera.ts'
import { LatticeGenerator, SHAPES, TILINGS } from '../../src/days/002/generators/lattice.ts'
import { DETAILS, MAX_DETAIL, MAX_RENDER_VERTS, renderSize } from '../../src/days/002/render.ts'
import { MAX_BUBBLES, MAX_MAP, MAX_QUADS, MAX_VERTS, buildSheetFromMap, layoutFor } from '../../src/days/002/sheet.ts'

const GRID = [4, 11, 20]
const SIZE = [0.6, 1.35, 2.2]
const SPACING = [0.1, 0.86, 1]
const ASPECTS = [0.46, 1, 1.6, 2.4]
const DEPTH_MARGIN = 8

type Result = { configs: number; failures: string[]; summary: Record<string, string | number> }

async function run(): Promise<Result> {
  const renderer = new THREE.WebGPURenderer()
  await renderer.init()
  const mapRaw = instancedArray(MAX_MAP, 'vec4')
  const generator = new LatticeGenerator(mapRaw)
  const failures: string[] = []
  const counts: Record<string, number[]> = {}
  // Device limits the render surface has to fit (WebGPU defaults are 128 MiB bindings, 256 MiB buffers).
  const limits = (renderer.backend as unknown as { device: GPUDevice }).device.limits
  const storageBytes = MAX_RENDER_VERTS * 16
  if (storageBytes > limits.maxStorageBufferBindingSize) {
    failures.push(`render storage buffers ${storageBytes} B > maxStorageBufferBindingSize ${limits.maxStorageBufferBindingSize}`)
  }
  let worstRender = { verts: 0, indexBytes: 0 }
  let configs = 0
  let worstDepth = 0
  let minGapRatio = Infinity

  for (const tiling of TILINGS) {
    for (const shape of SHAPES) {
      for (const cols of GRID) {
        for (const size of SIZE) {
          for (const spacing of SPACING) {
            configs++
            const label = `${tiling}/${shape} grid=${cols} size=${size} spacing=${spacing}`
            const fail = (msg: string) => failures.push(`${label}: ${msg}`)
            const layout = layoutFor({ size, spacing, cols })
            const texels = layout.mapW * layout.mapH
            if (texels > MAX_MAP) fail(`map ${texels} texels > capacity ${MAX_MAP}`)
            generator.configure(layout, { tiling, shape })
            renderer.compute(generator.compute)
            const raw = new Float32Array(await renderer.getArrayBufferAsync(mapRaw.value, null, 0, texels * 16))
            const sheet = buildSheetFromMap({ layout, raw, pinholeChance: 0.06 })
            const verts = sheet.W * sheet.H
            ;(counts[`${tiling} grid=${cols}`] ??= []).push(sheet.count)

            // Capacity.
            if (verts > MAX_VERTS) fail(`${verts} vertices > ${MAX_VERTS}`)
            if (sheet.count < 1) fail('no bubbles')
            if (sheet.count > MAX_BUBBLES) fail(`${sheet.count} bubbles > ${MAX_BUBBLES}`)
            if (sheet.quadList.length > MAX_QUADS) fail(`${sheet.quadList.length} listed quads > ${MAX_QUADS}`)
            let maxIndex = 0
            for (const i of sheet.index) maxIndex = Math.max(maxIndex, i)
            if (maxIndex >= verts) fail(`index ${maxIndex} >= ${verts}`)

            // Dense ids, no overlaps, whole bubbles only.
            const covered = new Uint8Array(sheet.count)
            let maxCover = 0
            let edgeTouch = 0
            for (let t = 0; t < texels; t++) {
              const id = sheet.map[t * 4]
              const sd = sheet.map[t * 4 + 1]
              maxCover = Math.max(maxCover, sheet.map[t * 4 + 3])
              if (id < -1 || id >= sheet.count || id !== Math.floor(id)) fail(`texel ${t} has id ${id}`)
              if (id >= 0 && sd < 0) {
                covered[id] = 1
                const tx = t % layout.mapW
                const ty = Math.floor(t / layout.mapW)
                if (tx === 0 || ty === 0 || tx === layout.mapW - 1 || ty === layout.mapH - 1) edgeTouch++
              }
            }
            const missing = covered.reduce((n, c) => n + (c ? 0 : 1), 0)
            if (missing) fail(`${missing} ids have no footprint (ids not dense)`)
            if (maxCover > 1) fail(`footprints overlap (coverage ${maxCover})`)
            if (edgeTouch) fail(`${edgeTouch} footprint texels touch the sheet edge (cut-off bubble)`)
            if (tiling === 'square' && sheet.count !== layout.cols * layout.rows) {
              fail(`square tiling has ${sheet.count} bubbles, expected ${layout.cols * layout.rows}`)
            }

            // Volumes and quad lists.
            for (let b = 0; b < sheet.count; b++) {
              if (!(sheet.bubbles[b * 4 + 3] > 0)) fail(`bubble ${b} rest volume ${sheet.bubbles[b * 4 + 3]}`)
              if (!(sheet.quadRange[b * 2 + 1] > 0)) fail(`bubble ${b} owns no quads`)
            }

            // Spacing semantic: neighbouring centers are one pitch apart on every tiling.
            let nearest = Infinity
            for (let a = 0; a < sheet.count; a++) {
              for (let b = a + 1; b < sheet.count; b++) {
                const d = Math.hypot(sheet.centers[a * 4] - sheet.centers[b * 4], sheet.centers[a * 4 + 1] - sheet.centers[b * 4 + 1])
                nearest = Math.min(nearest, d)
              }
            }
            // Centers are footprint-texel centroids, so each carries up to half a texel of quantisation.
            if (sheet.count > 1) {
              const error = Math.abs(nearest - layout.pitch) / layout.texel
              minGapRatio = Math.min(minGapRatio, nearest / layout.pitch)
              if (error > 1.5) fail(`nearest neighbours ${nearest.toFixed(3)} cm apart, pitch ${layout.pitch.toFixed(3)}`)
            }

            // Render surface at every detail level, including the 4x maximum.
            for (const detail of DETAILS) {
              const { RW, RH } = renderSize(sheet.W, sheet.H, detail)
              const rVerts = RW * RH
              const indexBytes = (RW - 1) * (RH - 1) * 6 * 4
              if (rVerts > MAX_RENDER_VERTS) fail(`${detail}x render surface ${rVerts} vertices > ${MAX_RENDER_VERTS}`)
              if (indexBytes > limits.maxBufferSize) fail(`${detail}x index buffer ${indexBytes} B > maxBufferSize`)
              if (detail === MAX_DETAIL && rVerts > worstRender.verts) worstRender = { verts: rVerts, indexBytes }
            }

            // Rest normals.
            let badNormals = 0
            for (let v = 0; v < verts; v++) {
              const n = sheet.restNormal.subarray(v * 4, v * 4 + 3)
              const len = Math.hypot(n[0], n[1], n[2])
              if (!Number.isFinite(len) || Math.abs(len - 1) > 1e-5 || n[1] < 0) badNormals++
            }
            if (badNormals) fail(`${badNormals} rest normals NaN, non-unit or facing down`)

            // Depth precision: the film must stay resolvable above the table everywhere in view.
            for (const aspect of ASPECTS) {
              const fit = fitCamera(sheet.width, sheet.depth, aspect)
              const cam = [0, Math.sin(fit.elevation) * fit.dist, Math.cos(fit.elevation) * fit.dist]
              const fwd = [fit.target[0] - cam[0], fit.target[1] - cam[1], fit.target[2] - cam[2]]
              const fl = Math.hypot(fwd[0], fwd[1], fwd[2])
              let zMin = Infinity
              let zMax = 0
              for (const cx of [-1, 1]) {
                for (const cz of [-1, 1]) {
                  const p = [(cx * sheet.width) / 2 - cam[0], -cam[1], (cz * sheet.depth) / 2 - cam[2]]
                  const z = (p[0] * fwd[0] + p[1] * fwd[1] + p[2] * fwd[2]) / fl
                  zMin = Math.min(zMin, z)
                  zMax = Math.max(zMax, z)
                }
              }
              if (zMin <= fit.near) fail(`aspect ${aspect}: sheet inside the near plane`)
              const ratio = (depthStep(zMax * 1.1, fit.near, fit.far) * DEPTH_MARGIN) / TABLE_GAP
              worstDepth = Math.max(worstDepth, ratio)
              if (ratio >= 1) fail(`aspect ${aspect}: depth buffer too coarse for the film/table gap`)
            }
          }
        }
      }
    }
  }
  renderer.dispose()
  const range = (k: string) => `${Math.min(...counts[k])}-${Math.max(...counts[k])}`
  return {
    configs,
    failures,
    summary: {
      'bubbles square grid=20': range('square grid=20'),
      'bubbles triangular grid=20': range('triangular grid=20'),
      'bubbles hexagonal grid=20': range('hexagonal grid=20'),
      'nearest-neighbour / pitch (min)': minGapRatio.toFixed(4),
      'tightest depth margin': `${(DEPTH_MARGIN / worstDepth).toFixed(0)}x`,
      'render 4x max': `${worstRender.verts} / ${MAX_RENDER_VERTS} vertices, index ${(worstRender.indexBytes / 2 ** 20).toFixed(1)} MiB, storage ${(storageBytes / 2 ** 20).toFixed(1)} MiB per buffer`,
    },
  }
}

const w = window as unknown as { __check?: Result | { error: string } }
run().then(
  (r) => (w.__check = r),
  (e: unknown) => (w.__check = { error: String((e as Error)?.stack ?? e) }),
)
