import { bloom } from 'three/addons/tsl/display/BloomNode.js'
import { dof } from 'three/addons/tsl/display/DepthOfFieldNode.js'
import { film } from 'three/addons/tsl/display/FilmNode.js'
import { float, length, mix, pass, screenUV, smoothstep, uniform, vec4 } from 'three/tsl'
import * as THREE from 'three/webgpu'
import type { Node } from 'three/webgpu'

export type StudioLights = { key: number; rim: number }

/**
 * A dark-field product studio baked into an environment map: emissive panels on pure black,
 * so clear plastic reflects clean white bars and its rims and grazing faces glow while faces
 * toward the camera stay dark. A large overhead softbox, two vertical strip boxes, two long
 * horizontal strips behind the set and low edge kickers for the rims.
 */
export function studioEnvironment(renderer: THREE.WebGPURenderer, { key, rim }: StudioLights) {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0, 0, 0)
  const panel = (w: number, h: number, intensity: number, x: number, y: number, z: number) => {
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(1, 1, 1).multiplyScalar(intensity), side: THREE.DoubleSide }),
    )
    m.position.set(x, y, z)
    m.lookAt(0, 0, 0)
    scene.add(m)
  }
  // Layout rule: keep the flat film's mirror direction (about 56 degrees up, behind the set,
  // as seen from the camera) black, so the flat film stays dark and only creases, dome slopes
  // and silhouettes pick up the lights.
  // Overhead softbox straight up, leaning toward the camera: dome tops.
  panel(8, 4, key * 0.7, 0, 9, 3)
  // Vertical strip boxes left and right: long bars down the sides of every dome.
  panel(0.8, 8, key, -8, 2.5, 1)
  panel(0.8, 8, key * 0.8, 8, 2.5, -1)
  // Long, low horizontal strips behind the set: bars along the far slopes.
  panel(12, 0.5, key * 0.9, 0, 3, -8)
  panel(12, 0.3, key * 0.5, 0, 1.2, -8.5)
  // Edge kickers just above and below the horizon, all the way round: grazing rays off dome
  // rims, folds and silhouettes land here, which is what makes clear plastic glow at its edges.
  for (let k = 0; k < 8; k++) {
    const a = (k / 8) * Math.PI * 2 + 0.2
    panel(2.8, 0.3, rim, Math.cos(a) * 9, 0.5, Math.sin(a) * 9)
    panel(3.6, 0.6, rim * 0.8, Math.cos(a + 0.4) * 9, -3, Math.sin(a + 0.4) * 9)
  }
  const pmrem = new THREE.PMREMGenerator(renderer)
  const target = pmrem.fromScene(scene, 0.02)
  pmrem.dispose()
  scene.traverse((o) => {
    if (o instanceof THREE.Mesh) {
      o.geometry.dispose()
      ;(o.material as THREE.Material).dispose()
    }
  })
  return target
}

export type CameraLook = { dof: number; bloom: number; grain: number; vignette: number }

/**
 * Camera feel: a thin depth of field focused where you look, a little halation on the brightest
 * speculars, a faint vignette and very fine grain, then AgX tone mapping in the output transform.
 * DOF and bloom drop out of the graph entirely at 0.
 */
export class StudioPipeline {
  readonly focus = uniform(20)
  readonly focalLength = uniform(6)
  readonly bokeh = uniform(1)
  readonly bloomStrength = uniform(0.35)
  readonly grain = uniform(0.05)
  readonly vignette = uniform(0.35)
  private pipeline: THREE.RenderPipeline
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private shape = ''

  constructor(renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    this.pipeline = new THREE.RenderPipeline(renderer)
    this.scene = scene
    this.camera = camera
  }

  configure(look: CameraLook) {
    this.bokeh.value = look.dof * 3
    this.bloomStrength.value = look.bloom
    this.grain.value = look.grain
    this.vignette.value = look.vignette
    const shape = `${look.dof > 0}-${look.bloom > 0}`
    if (shape === this.shape) return
    this.shape = shape
    const scenePass = pass(this.scene, this.camera)
    let color: Node<'vec4'> = scenePass.getTextureNode('output')
    if (look.dof > 0) color = dof(color, scenePass.getViewZNode(), this.focus, this.focalLength, this.bokeh) as unknown as Node<'vec4'>
    if (look.bloom > 0) color = color.add(bloom(color, this.bloomStrength, 0.25, 0.9))
    const edge = smoothstep(0.85, 0.25, length(screenUV.sub(0.5)).mul(1.35))
    const vignetted = vec4(color.rgb.mul(mix(float(1), edge, this.vignette)), 1)
    this.pipeline.outputNode = film(vignetted, this.grain)
    this.pipeline.needsUpdate = true
  }

  render() {
    this.pipeline.render()
  }

  dispose() {
    this.pipeline.dispose()
  }
}
