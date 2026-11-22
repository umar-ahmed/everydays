// `npm run check`: serves scripts/check-002/ with Vite and runs it in headless Chromium, where
// WebGPU runs on SwiftShader (no GPU needed), so the checks exercise the real map generator.
import { chromium } from 'playwright'
import { createServer } from 'vite'

const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' })
await server.listen()
const { port } = server.httpServer.address()
const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader'],
})
let code = 1
try {
  const page = await browser.newPage()
  page.on('pageerror', (e) => console.error('page error:', e.message))
  const started = Date.now()
  await page.goto(`http://127.0.0.1:${port}/scripts/check-002/`)
  await page.waitForFunction(() => window.__check, null, { timeout: 15 * 60 * 1000, polling: 1000 })
  const result = await page.evaluate(() => window.__check)
  if (result.error) {
    console.error(result.error)
  } else {
    const seconds = ((Date.now() - started) / 1000).toFixed(0)
    console.log(`checked ${result.configs} map-driven sheets (3 tilings x 6 shapes x grid/size/spacing) in ${seconds}s`)
    for (const [k, v] of Object.entries(result.summary)) console.log(`  ${k}: ${v}`)
    if (result.failures.length) {
      console.error(result.failures.slice(0, 25).join('\n'))
      console.error(`${result.failures.length} failures`)
    } else {
      code = 0
    }
  }
} finally {
  await browser.close()
  await server.close()
}
process.exit(code)
