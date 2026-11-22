import {
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
import type { BubbleSim } from './sim.ts'


export class FilmLook {
  readonly thickness = uniform(420)
  readonly iridescence = uniform(0.35)
  readonly haze = uniform(0.16)
  readonly wrinkle = uniform(0.45)
  readonly wrinkleScale = uniform(1)
  readonly sharpness = uniform(0.4)
  /** Refraction depth of a full dome, cm: each dome reads as a small lens. */
  readonly lens = uniform(0.4)

  // Sheet layout for the table's caustic lookup.
  readonly cell = uniform(1.4)
  readonly cols = uniform(1, 'int')
  readonly rows = uniform(1, 'int')
  readonly half = uniform(new THREE.Vector2(1, 1))
  readonly spot = uniform(new THREE.Vector2(0.2, 0.15))
  readonly domeR = uniform(0.5)
  readonly caustics = uniform(1)

  readonly film: THREE.MeshPhysicalNodeMaterial
  readonly table: THREE.MeshStandardNodeMaterial
  /** Debug views: world normals as colour, and the triangle wireframe. */
  readonly normals: THREE.MeshBasicNodeMaterial
  readonly wireframe: THREE.MeshBasicNodeMaterial

  /** 1 renders the sheet builder's rest geometry with no sim output (Debug > Simulation). */
  readonly restPose = uniform(0)

  constructor(sim: BubbleSim) {
    this.film = this.makeFilm(sim)
    this.table = this.makeTable(sim)
    const position = this.position(sim)
    this.normals = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide })
    this.normals.positionNode = position
    this.normals.colorNode = normalize(varying(this.normal(sim).xyz)).mul(0.5).add(0.5)
    this.wireframe = new THREE.MeshBasicNodeMaterial({ color: 0xd8dce2, wireframe: true })
    this.wireframe.positionNode = position
  }

  private live() {
    return this.restPose.lessThan(0.5)
  }

  private position(sim: BubbleSim) {
    return select(this.live(), sim.x.element(vertexIndex).xyz, sim.rest.element(vertexIndex).xyz)
  }

  /** xyz normal, w compression; rest normals carry w = 0. */
  private normal(sim: BubbleSim) {
    return select(this.live(), sim.normal.element(vertexIndex), sim.restNormal.element(vertexIndex))
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
    const dome = varying(sim.rest.element(vertexIndex).w)
    const inside = varying(sim.meta.element(vertexIndex).y)
    const restXZ = varying(sim.rest.element(vertexIndex).xz)
    const popped = varying(select(this.live(), sim.pops.element(int(sim.meta.element(vertexIndex).x)).x, float(0)))

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
    const N = normalize(n.xyz)
    const T = normalize(vec3(1, 0, 0).sub(N.mul(N.x)))
    const B = cross(N, T)
    const bent = normalize(N.sub(T.mul(hu.sub(h0).div(eps).mul(amp))).sub(B.mul(hv.sub(h0).div(eps).mul(amp))))
    film.normalNode = transformNormalToView(bent).normalize().mul(faceDirection)
    film.roughnessNode = this.haze.add(h0.mul(this.wrinkle).mul(mask).mul(mix(float(0.04), float(0.12), resolved)))
    // Clear PE scatters almost nothing diffusely; the laminated flat film is only a touch milkier.
    film.transmissionNode = mix(float(0.965), float(1), dome)
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
    const local = p.add(this.half).div(this.cell)
    const id = floor(local)
    const inside = id.x
      .greaterThanEqual(0)
      .and(id.y.greaterThanEqual(0))
      .and(id.x.lessThan(float(this.cols)))
      .and(id.y.lessThan(float(this.rows)))
    const b = select(inside, int(id.y).mul(this.cols).add(int(id.x)), int(0))
    const intact = select(this.live(), oneMinus(sim.pops.element(b).x), float(1))
    const f = fract(local).sub(0.5).mul(this.cell).sub(this.spot)
    const focus: Node<'float'> = exp(dot(f, f).div(this.domeR.mul(this.domeR).mul(0.06)).negate())
    const glow = select(inside, focus.mul(intact).mul(this.caustics).mul(0.035), float(0))
    // The film absorbs a little, so the sheet's footprint sits slightly darker.
    const shade = select(inside, float(0.9), float(1))
    table.colorNode = sweep.mul(shade).add(vec3(1, 0.98, 0.95).mul(glow))
    table.envMapIntensity = 0.3
    return table
  }
}
