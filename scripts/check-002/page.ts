// Runs inside headless Chromium (WebGPU on SwiftShader): the real lattice generator fills the
// bubble map, the sheet is built from its readback, and every invariant is checked.
import { instancedArray } from 'three/tsl'
import * as THREE from 'three/webgpu'
import {
  ELEVATION,
  ELEVATIONS,
  MIN_GAP,
  type Pose,
  type View,
  clipPlanes,
  depthStep,
  fitView,
  rayHit,
  safeRect,
  sheetAspect,
  sheetBounds,
  viewDepth,
  zoomRange,
} from '../../src/days/002/camera.ts'
import { LatticeGenerator, SHAPES, TILINGS, latticeFor } from '../../src/days/002/generators/lattice.ts'
import { DETAILS, MAX_DETAIL, MAX_RENDER_VERTS, backIndex, renderSize } from '../../src/days/002/render.ts'
import {
  FULL_BUDGET,
  MAX_BUBBLES,
  MAX_MAP,
  MAX_QUADS,
  MAX_VERTS,
  PHONE_BUDGET,
  buildSheetFromMap,
  layoutFor,
} from '../../src/days/002/sheet.ts'

/** iPhone portrait and landscape, iPad portrait, 16:9 desktop, ultrawide; phones get the phone budget. */
const VIEWPORTS = [
  { name: 'iphone-portrait', width: 390, height: 844, budget: PHONE_BUDGET },
  { name: 'iphone-landscape', width: 844, height: 390, budget: PHONE_BUDGET },
  { name: 'ipad', width: 820, height: 1180, budget: FULL_BUDGET },
  { name: 'desktop-16x9', width: 1920, height: 1080, budget: FULL_BUDGET },
  { name: 'ultrawide', width: 3440, height: 1440, budget: FULL_BUDGET },
]
const ACROSS = [3, 8, 16]
const SIZE = [0.6, 1.35, 2.2]
const SPACING = [0.1, 0.86, 1]
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
  let worstBack = 0
  let configs = 0
  let worstDepth = 0
  let minGapRatio = Infinity
  let minFill = Infinity
  let poses = 0

  for (const vp of VIEWPORTS) {
    const view: View = { width: vp.width, height: vp.height }
    const aspect = sheetAspect(view)
    const portrait = vp.height > vp.width
    if (portrait ? aspect >= 1 : aspect <= 1) failures.push(`${vp.name}: sheet aspect ${aspect.toFixed(2)} doesn't follow the screen`)
    for (const tiling of TILINGS) {
      const lattice = latticeFor(tiling)
      for (const shape of SHAPES) {
        for (const across of ACROSS) {
          for (const size of SIZE) {
            for (const spacing of SPACING) {
              configs++
              const label = `${vp.name} ${tiling}/${shape} across=${across} size=${size} spacing=${spacing}`
              const fail = (msg: string) => failures.push(`${label}: ${msg}`)
              const layout = layoutFor({ size, spacing, across, aspect, lattice, budget: vp.budget })
              const texels = layout.mapW * layout.mapH
              if (texels > MAX_MAP) fail(`map ${texels} texels > capacity ${MAX_MAP}`)
              generator.configure(layout, { tiling, shape })
              renderer.compute(generator.compute)
              const raw = new Float32Array(await renderer.getArrayBufferAsync(mapRaw.value, null, 0, texels * 16))
              const sheet = buildSheetFromMap({ layout, raw, pinholeChance: 0.06 })
              const verts = sheet.W * sheet.H
              ;(counts[vp.name] ??= []).push(sheet.count)

              // Capacity and the per-device budget.
              if (verts > MAX_VERTS) fail(`${verts} vertices > ${MAX_VERTS}`)
              if (verts > vp.budget.verts) fail(`${verts} vertices > budget ${vp.budget.verts}`)
              if (sheet.count < 1) fail('no bubbles')
              if (sheet.count > MAX_BUBBLES) fail(`${sheet.count} bubbles > ${MAX_BUBBLES}`)
              if (sheet.count > vp.budget.bubbles) fail(`${sheet.count} bubbles > budget ${vp.budget.bubbles}`)
              if (sheet.count !== layout.sites) fail(`${sheet.count} bubbles in the map, layout planned ${layout.sites}`)
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
                // The back film shares the render vertex range; its index covers footprints + seal band.
                if (detail === MAX_DETAIL && across === 16 && shape === 'circle') {
                  const back = backIndex(sheet, detail)
                  let maxBack = 0
                  for (const v of back) maxBack = Math.max(maxBack, v)
                  if (back.length === 0) fail('back film index is empty')
                  if (maxBack >= rVerts) fail(`back film index ${maxBack} >= ${rVerts} render vertices`)
                  if (back.length * 4 > limits.maxBufferSize) fail('back film index exceeds maxBufferSize')
                  worstBack = Math.max(worstBack, back.length * 4)
                }
              }

              // Rest normals.
              let badNormals = 0
              for (let v = 0; v < verts; v++) {
                const n = sheet.restNormal.subarray(v * 4, v * 4 + 3)
                const len = Math.hypot(n[0], n[1], n[2])
                if (!Number.isFinite(len) || Math.abs(len - 1) > 1e-5 || n[1] < 0) badNormals++
              }
              if (badNormals) fail(`${badNormals} rest normals NaN, non-unit or facing down`)

              // Framing and depth precision only depend on the rectangle, not the shape.
              if (shape !== 'circle') continue
              const box = { width: sheet.width, depth: sheet.depth, top: sheet.domeHeight }
              const rect = safeRect(view)
              for (const elevation of [ELEVATION, ...ELEVATIONS]) {
                const fit = fitView(box, view, elevation)
                const b = sheetBounds(fit, fit.offset, box, view)
                if (!b) {
                  fail(`elevation ${elevation}: fit view has the sheet behind the camera`)
                  continue
                }
                const eps = 1e-3
                if (b.x0 < rect.x0 - eps || b.x1 > rect.x1 + eps || b.y0 < rect.y0 - eps || b.y1 > rect.y1 + eps) {
                  fail(`elevation ${elevation}: sheet spills outside the safe area`)
                }
                if (elevation === ELEVATION) {
                  // Fills the safe area in both directions (whole bubbles leave a little slack).
                  const fill = Math.min((b.x1 - b.x0) / (rect.x1 - rect.x0), (b.y1 - b.y0) / (rect.y1 - rect.y0))
                  minFill = Math.min(minFill, fill)
                  if (fill < 0.8) fail(`sheet fills only ${(fill * 100).toFixed(0)}% of the safe area`)
                }
                // The fit framing, then View-mode extremes: macro zoom panned to the center and the
                // corners, orbited and tilted to the limits.
                const range = zoomRange(fit, sheet.size, sheet.domeHeight, view)
                const candidates: Pose[] = [fit]
                for (const [px, pz] of [[0, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]]) {
                  for (const az of [0, -70, 70]) {
                    for (const el of [elevation, 12, 85]) {
                      const e = (el * Math.PI) / 180
                      const clear = (sheet.domeHeight * 1.6 + sheet.size * 0.15) / Math.sin(e)
                      for (const dist of [Math.min(Math.max(range.min, clear), range.max), range.max]) {
                        candidates.push({
                          target: [(px * box.width) / 2, 0, (pz * box.depth) / 2],
                          dist,
                          elevation: e,
                          azimuth: (az * Math.PI) / 180,
                        })
                      }
                    }
                  }
                }
                for (const pose of candidates) {
                  poses++
                  const { near, far } = clipPlanes(pose, fit.offset, box, view, fit.far)
                  const tag = `elevation ${elevation} pose ${pose.target.map((v) => v.toFixed(1))}/${pose.dist.toFixed(1)}/${((pose.elevation * 180) / Math.PI).toFixed(0)}/${((pose.azimuth * 180) / Math.PI).toFixed(0)}`
                  // Everything visible on the sheet (sampled through the frustum) sits between the planes.
                  let zMin = Infinity
                  let zMax = 0
                  for (let i = 0; i <= 8; i++) {
                    for (let j = 0; j <= 8; j++) {
                      const ndc: [number, number] = [i / 4 - 1, j / 4 - 1]
                      for (const y of [0, sheet.domeHeight]) {
                        const p = rayHit(pose, fit.offset, view, ndc, y)
                        if (!p || Math.abs(p[0]) > box.width / 2 || Math.abs(p[2]) > box.depth / 2) continue
                        const z = viewDepth(pose, p)
                        zMin = Math.min(zMin, z)
                        zMax = Math.max(zMax, z)
                      }
                    }
                  }
                  if (!Number.isFinite(zMin)) continue
                  if (zMin <= near) fail(`${tag}: sheet inside the near plane (${zMin.toFixed(3)} <= ${near.toFixed(3)})`)
                  if (zMax >= far) fail(`${tag}: sheet beyond the far plane`)
                  // The tightest separations: cap/back film at the seal band, and back film/table.
                  const ratio = (depthStep(zMax * 1.1, near, far) * DEPTH_MARGIN) / MIN_GAP
                  worstDepth = Math.max(worstDepth, ratio)
                  if (ratio >= 1) fail(`${tag}: depth buffer too coarse for the ${MIN_GAP} cm layer gap`)
                }
              }
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
      ...Object.fromEntries(
        VIEWPORTS.map((vp) => [
          `${vp.name} ${vp.width}x${vp.height}`,
          `sheet aspect ${sheetAspect(vp).toFixed(2)}, bubbles ${range(vp.name)}`,
        ]),
      ),
      'min fill of the safe area': `${(minFill * 100).toFixed(0)}%`,
      'nearest-neighbour / pitch (min)': minGapRatio.toFixed(4),
      'tightest depth margin': `${(DEPTH_MARGIN / worstDepth).toFixed(0)}x over ${poses} fit and zoomed poses`,
      'render 4x max': `${worstRender.verts} / ${MAX_RENDER_VERTS} vertices, index ${(worstRender.indexBytes / 2 ** 20).toFixed(1)} MiB, storage ${(storageBytes / 2 ** 20).toFixed(1)} MiB per buffer (6 buffers incl. back film), back-film index up to ${(worstBack / 2 ** 20).toFixed(1)} MiB`,
    },
  }
}

const w = window as unknown as { __check?: Result | { error: string } }
run().then(
  (r) => (w.__check = r),
  (e: unknown) => (w.__check = { error: String((e as Error)?.stack ?? e) }),
)
