import {
  log,
  sqrt,
  cameraPosition,
  Fn,
  abs,
  clamp,
  cos,
  cross,
  dot,
  exp,
  faceDirection,
  float,
  floor,
  fract,
  fwidth,
  int,
  length,
  max,
  mix,
  mx_noise_float,
  normalize,
  oneMinus,
  positionWorld,
  pow,
  select,
  sin,
  smoothstep,
  transformNormalToView,
  uniform,
  varying,
  vec2,
  vec3,
  vertexIndex,
} from 'three/tsl'
import * as THREE from 'three/webgpu'
import type { Node } from 'three/webgpu'
import type { RenderSurface } from './render.ts'
import type { BubbleSim } from './sim.ts'


export class FilmLook {
  readonly thickness = uniform(420)
  readonly iridescence = uniform(0.35)
  readonly haze = uniform(0.08)
  readonly wrinkle = uniform(0.45)
  readonly wrinkleScale = uniform(1)
  readonly sharpness = uniform(0.4)
  /** Refraction depth of a full dome, cm: each dome reads as a small lens. */
  readonly lens = uniform(0.4)

  /** Embossed film texture: 0 none, 1 grid, 2 dots. Period is size / (10 * scale). */
  readonly texture = uniform(0, 'int')
  readonly textureScale = uniform(1)
  readonly textureDepth = uniform(0.8)
  readonly textureSharpness = uniform(0.5)
  readonly bubbleSize = uniform(1.1)
  /** Beer-Lambert tint: transmittance colour for one film pass, raised to the Density power. */
  readonly tint = uniform(new THREE.Vector3(1, 1, 1))
  readonly density = uniform(1)

  /** Offset of each dome's focused spot on the table, away from the key light (cm). */
  readonly spot = uniform(new THREE.Vector2(0.2, 0.15))
  readonly caustics = uniform(1)
  /** Contour spacing for the map view (cm). */
  readonly contour = uniform(0.1)

  readonly film: THREE.MeshPhysicalNodeMaterial
  readonly table: THREE.MeshStandardNodeMaterial
  /** Debug views: world normals as colour, and the triangle wireframe. */
  readonly normals: THREE.MeshBasicNodeMaterial
  readonly wireframe: THREE.MeshBasicNodeMaterial
  readonly mapView: THREE.MeshBasicNodeMaterial

  /** 1 renders the sheet builder's rest geometry with no sim output (Debug > Simulation). */
  readonly restPose = uniform(0)

  private surface: RenderSurface

  constructor(sim: BubbleSim, surface: RenderSurface) {
    this.surface = surface
    this.film = this.makeFilm(sim)
    this.table = this.makeTable(sim)
    const position = this.position(sim)
    this.normals = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide })
    this.normals.positionNode = position
    this.normals.colorNode = normalize(varying(this.normal(sim).xyz)).mul(0.5).add(0.5)
    this.wireframe = new THREE.MeshBasicNodeMaterial({ color: 0xd8dce2, wireframe: true })
    this.wireframe.positionNode = position
    this.mapView = this.makeMapView(sim)
  }

  private live() {
    return this.restPose.lessThan(0.5)
  }

  // The render surface already folds rest pose in (no sim displacement), so every view reads it.
  private position(_sim: BubbleSim) {
    return this.surface.pos.element(vertexIndex).xyz
  }

  /** xyz normal, w compression (0 in rest pose). */
  private normal(_sim: BubbleSim) {
    return this.surface.normal.element(vertexIndex)
  }

  /**
   * Clear PE film (IOR ~1.5): physical transmission so the backdrop refracts through each
   * dome, haze from surface roughness rather than the bulk, Belcour-Barla thin-film
   * iridescence (three's built-in model) kept subtle by default, plus a procedural crease
   * layer for the wrinkles finer than the sim mesh.
   */
  private makeFilm(sim: BubbleSim) {
    const film = new THREE.MeshPhysicalNodeMaterial({
      color: 0xffffff,
      metalness: 0,
      ior: 1.5,
      iridescenceIOR: 1.5,
      transmission: 1,
      side: THREE.DoubleSide,
      specularIntensity: 1,
      attenuationColor: new THREE.Color(0xeef1f4),
      attenuationDistance: 3,
    })
    film.positionNode = this.position(sim)
    const n = varying(this.normal(sim))
    const info = this.surface.info.element(vertexIndex)
    const dome = varying(this.surface.rest.element(vertexIndex).w)
    const inside = varying(info.y)
    const restXZ = varying(this.surface.rest.element(vertexIndex).xz)
    const bubble = int(max(info.x, 0))
    const popped = varying(select(this.live().and(info.x.greaterThanEqual(0)), sim.pops.element(bubble).x, float(0)))

    // Creases: anisotropic ridged noise whose direction drifts across the film.
    const sharp = mix(float(1.5), float(9), this.sharpness)
    // Fade creases where they get finer than a pixel: the fine octave first, then the coarse one.
    const footprint = fwidth(restXZ.x).add(fwidth(restXZ.y)).mul(this.wrinkleScale)
    const resolved = smoothstep(0.06, 0.02, footprint)
    const fine = smoothstep(0.025, 0.008, footprint)
    const crease = Fn(([p]: [Node<'vec2'>]) => {
      const theta = mx_noise_float(vec3(p.mul(0.35), 1.7)).mul(Math.PI)
      const c = cos(theta)
      const s = sin(theta)
      const q = vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c))).mul(this.wrinkleScale)
      const r1 = oneMinus(abs(mx_noise_float(vec3(q.x.mul(1.1), q.y.mul(5), 0))))
      const r2 = oneMinus(abs(mx_noise_float(vec3(q.x.mul(2.4), q.y.mul(11), 3.3))))
      return pow(r1, sharp).mul(0.7).add(pow(r2, sharp).mul(fine.mul(0.3)))
    })
    // Taut domes stay smooth (and their steep walls would smear rest-space noise into streaks);
    // the laminated flat film, collapsed domes and squeezed areas crinkle.
    const mask = clamp(
      oneMinus(inside).add(popped.mul(inside).mul(1.4)).add(smoothstep(0.03, 0.2, n.w).mul(1.5)),
      0,
      2.5,
    )
    const eps = float(0.004).div(this.wrinkleScale)
    const h0 = crease(restXZ)
    const hu = crease(restXZ.add(vec2(eps, 0)))
    const hv = crease(restXZ.add(vec2(0, eps)))
    const amp = this.wrinkle.mul(mask).mul(resolved).mul(0.006)
    const wrinkleGrad = vec2(hu.sub(h0), hv.sub(h0)).div(eps).mul(amp)

    // Embossed texture in dome-local coordinates (rest position minus the bubble's map center),
    // so it sits on each bubble and stretches with the film. Analytic gradients, faded by the
    // pixel footprint so the pattern never aliases.
    const center = varying(select(info.x.greaterThanEqual(0), sim.centers.element(bubble).xy, vec2(0)))
    const local = restXZ.sub(center)
    const freq = this.textureScale.mul(10).div(this.bubbleSize)
    const cycles = fwidth(local.x).add(fwidth(local.y)).mul(freq)
    const k = mix(float(2), float(24), this.textureSharpness)
    const texFade = smoothstep(0.45, 0.12, cycles.mul(this.textureSharpness.mul(1.5).add(1)))
    const texMask = mix(float(0.25), float(1), inside).mul(texFade).mul(this.textureDepth)
    // Crosshatch ridges: r(t) = (0.5 + 0.5 cos 2 pi f t)^k along each axis.
    const ridge = (t: Node<'float'>) => {
      const c = cos(t.mul(freq).mul(2 * Math.PI)).mul(0.5).add(0.5)
      const slope = pow(max(c, 1e-4), k.sub(1)).mul(k).mul(sin(t.mul(freq).mul(2 * Math.PI)).mul(-Math.PI)).mul(freq)
      return { h: pow(c, k), slope }
    }
    const rx = ridge(local.x)
    const ry = ridge(local.y)
    const gridGrad = vec2(rx.slope, ry.slope).mul(texMask.mul(0.03).div(sqrt(k)))
    const gridHeight = rx.h.add(ry.h).mul(texMask)
    // Dimples on offset rows: h = -exp(-k r^2), r measured in cells.
    const row = floor(local.y.mul(freq))
    const shift = row.sub(floor(row.mul(0.5)).mul(2)).mul(0.5)
    const d = vec2(fract(local.x.mul(freq).add(shift)), fract(local.y.mul(freq))).sub(0.5).div(0.32)
    const e = exp(dot(d, d).mul(k.mul(-0.5)))
    const dotsGrad = d.mul(e).mul(k).mul(freq).div(0.32).mul(texMask.mul(0.035).div(sqrt(k)))
    const dotsHeight = e.mul(texMask)
    const isGrid = this.texture.equal(1)
    const isDots = this.texture.equal(2)
    const grad = wrinkleGrad.add(select(isGrid, gridGrad, select(isDots, dotsGrad, vec2(0))))
    const texHeight = select(isGrid, gridHeight, select(isDots, dotsHeight, float(0)))

    const N = normalize(n.xyz)
    const T = normalize(vec3(1, 0, 0).sub(N.mul(N.x)))
    const B = cross(N, T)
    const bent = normalize(N.sub(T.mul(grad.x)).sub(B.mul(grad.y)))
    film.normalNode = transformNormalToView(bent).normalize().mul(faceDirection)
    // Emboss ridges glint: a touch smoother than the surrounding haze.
    film.roughnessNode = max(
      this.haze.add(h0.mul(this.wrinkle).mul(mask).mul(mix(float(0.04), float(0.12), resolved))).sub(texHeight.mul(0.06)),
      0.02,
    )
    // Beer-Lambert: transmittance exp(-sigma L), with the path length L growing at grazing angles;
    // overlapping layers multiply on their own. Clear film has sigma = 0.
    const sigma = log(max(this.tint, vec3(0.01))).negate().mul(this.density)
    const view = normalize(cameraPosition.sub(positionWorld))
    const path = float(1).div(max(abs(dot(bent, view)), 0.15))
    const transmittance = exp(sigma.mul(path).negate())
    film.colorNode = transmittance
    // A thin film reflects off its back surface too, and that light crosses the film twice: so
    // half the specular carries the tint squared. On black this is where the colour shows.
    film.specularColorNode = mix(vec3(1), transmittance.mul(transmittance), 0.5)
    // Clear PE scatters almost nothing diffusely: on black, any diffuse share turns the film gray.
    film.transmissionNode = float(1)
    film.thicknessNode = dome.mul(this.lens).mul(oneMinus(popped.mul(0.8)))
    film.iridescenceNode = this.iridescence
    // Thermoforming stretches the domes thinner than the flat base; noise breaks up the bands.
    const wobble = mx_noise_float(vec3(restXZ.mul(0.9), 0.3)).mul(0.18).add(1)
    film.iridescenceThicknessNode = this.thickness.mul(float(1).sub(dome.mul(0.45))).mul(wobble)
    return film
  }

  /** Studio sweep with a faint focused spot under each intact dome (a cheap fake caustic). */
  private makeTable(sim: BubbleSim) {
    const table = new THREE.MeshStandardNodeMaterial({ roughness: 0.92, metalness: 0 })
    const p = positionWorld.xz
    // Linear values: a lit centre around 18% gray falling to a near-black backdrop.
    const sweep = mix(vec3(0.016, 0.0165, 0.018), vec3(0.004, 0.0042, 0.0048), smoothstep(3, 22, length(p)))
    const { onSheet, id } = mapAt(sim, p)
    const bubble = onSheet.and(id.greaterThanEqual(0))
    const b = select(bubble, id, int(0))
    const center = sim.centers.element(b)
    const intact = select(this.live(), oneMinus(sim.pops.element(b).x), float(1))
    const f = p.sub(center.xy).sub(this.spot)
    const focus: Node<'float'> = exp(dot(f, f).div(center.z.mul(center.z).mul(0.06)).negate())
    const glow = select(bubble, focus.mul(intact).mul(this.caustics).mul(0.035), float(0))
    // The film absorbs a little, so the sheet's footprint sits slightly darker.
    const shade = select(onSheet, float(0.9), float(1))
    table.colorNode = sweep.mul(shade).add(vec3(1, 0.98, 0.95).mul(glow))
    table.envMapIntensity = 0.3
    return table
  }

  /** Debug view of the map: a colour per bubble id, distance contours, white rims, red overlaps. */
  private makeMapView(sim: BubbleSim) {
    const view = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide })
    view.positionNode = this.position(sim)
    const { texel, id } = mapAt(sim, varying(this.surface.rest.element(vertexIndex).xz))
    const sd = texel.y
    const k = float(id)
    const hue = vec3(fract(sin(k.mul(12.9898)).mul(43758.5)), fract(sin(k.mul(78.233)).mul(12543.1)), fract(sin(k.mul(39.425)).mul(24634.6)))
    const tint = mix(vec3(0.25), hue.mul(0.75).add(0.2), select(sd.lessThan(0), float(1), float(0.35)))
    const band = smoothstep(0, 0.12, fract(sd.div(this.contour)))
    const rim = oneMinus(smoothstep(0, 1.5, abs(sd).div(sim.texel)))
    const base = select(id.lessThan(0), vec3(0.16), tint.mul(mix(float(0.7), float(1), band)))
    view.colorNode = select(texel.w.greaterThan(1.5), vec3(1, 0.1, 0.1), mix(base, vec3(1), rim))
    return view
  }
}

/** Nearest map texel for a sheet-space xz position. */
function mapAt(sim: BubbleSim, xz: Node<'vec2'>) {
  const t = xz.add(sim.half).div(sim.texel)
  const tx = int(floor(t.x.add(0.5)))
  const ty = int(floor(t.y.add(0.5)))
  const onSheet = tx.greaterThanEqual(0).and(ty.greaterThanEqual(0)).and(tx.lessThan(sim.mapW)).and(ty.lessThan(sim.mapH))
  const texel = sim.map.element(select(onSheet, ty.mul(sim.mapW).add(tx), int(0)))
  return { texel, onSheet, id: int(select(onSheet, texel.x, float(-1))) }
}
