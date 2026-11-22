// Day 002 invariants across the dial ranges: buffer capacity, rest normals, and enough depth
// precision to keep the film in front of the table. Run with `npm run check`.
import { TABLE_GAP, depthStep, fitCamera } from '../src/days/002/camera.ts'
import { MAX_BUBBLES, MAX_VERTS, buildSheet } from '../src/days/002/sheet.ts'
import { SHAPES } from '../src/days/002/shapes.ts'

const GRID = [4, 11, 18, 20]
const SIZE = [0.6, 1.1, 1.35, 2.2]
const SPACING = [0.1, 0.28, 0.86, 1]
const ASPECTS = [0.46, 1, 1.6, 2.4]
const DEPTH_MARGIN = 8

const failures: string[] = []
let configs = 0
let worstDepth = { ratio: 0, label: '' }

for (const shape of SHAPES) {
  for (const cols of GRID) {
    for (const size of SIZE) {
      for (const spacing of SPACING) {
        configs++
        const label = `${shape} grid=${cols} size=${size} spacing=${spacing}`
        const sheet = buildSheet({ shape, size, spacing, cols, pinholeChance: 0.06 })
        const verts = sheet.W * sheet.H
        const bubbles = sheet.cols * sheet.rows
        if (verts > MAX_VERTS) failures.push(`${label}: ${verts} vertices > capacity ${MAX_VERTS}`)
        if (bubbles > MAX_BUBBLES) failures.push(`${label}: ${bubbles} bubbles > capacity ${MAX_BUBBLES}`)
        if (sheet.rest.length !== verts * 4 || sheet.restNormal.length !== verts * 4 || sheet.meta.length !== verts * 4) {
          failures.push(`${label}: per-vertex arrays do not match ${verts} vertices`)
        }
        if (sheet.index.length !== (sheet.W - 1) * (sheet.H - 1) * 6) failures.push(`${label}: index count ${sheet.index.length}`)
        let maxIndex = 0
        for (const i of sheet.index) maxIndex = Math.max(maxIndex, i)
        if (maxIndex >= verts) failures.push(`${label}: index ${maxIndex} >= ${verts} vertices`)

        let bad = 0
        for (let v = 0; v < verts; v++) {
          const [x, y, z] = [sheet.restNormal[v * 4], sheet.restNormal[v * 4 + 1], sheet.restNormal[v * 4 + 2]]
          const len = Math.hypot(x, y, z)
          if (!Number.isFinite(len) || Math.abs(len - 1) > 1e-5 || y < 0) bad++
        }
        if (bad) failures.push(`${label}: ${bad} rest normals are NaN, not unit length, or face down`)

        for (const aspect of ASPECTS) {
          const fit = fitCamera(sheet.width, sheet.depth, aspect)
          // View depth of the sheet's corners; sway moves the camera by a few degrees at most.
          const cam = [0, Math.sin(fit.elevation) * fit.dist, Math.cos(fit.elevation) * fit.dist]
          const fwd = [fit.target[0] - cam[0], fit.target[1] - cam[1], fit.target[2] - cam[2]]
          const fl = Math.hypot(...fwd)
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
          if (zMin <= fit.near) failures.push(`${label} aspect=${aspect}: sheet corner at ${zMin.toFixed(1)} cm is inside the near plane ${fit.near.toFixed(1)}`)
          const ratio = (depthStep(zMax * 1.1, fit.near, fit.far) * DEPTH_MARGIN) / TABLE_GAP
          if (ratio > worstDepth.ratio) worstDepth = { ratio, label: `${label} aspect=${aspect}` }
          if (ratio >= 1) {
            failures.push(`${label} aspect=${aspect}: depth step ${depthStep(zMax * 1.1, fit.near, fit.far).toExponential(2)} cm is too coarse for the ${TABLE_GAP} cm film/table gap`)
          }
        }
      }
    }
  }
}

console.log(`checked ${configs} sheets x ${ASPECTS.length} aspects; tightest depth margin ${(DEPTH_MARGIN / worstDepth.ratio).toFixed(0)}x (${worstDepth.label})`)
if (failures.length) {
  console.error(failures.slice(0, 20).join('\n'))
  console.error(`${failures.length} failures`)
  process.exit(1)
}
