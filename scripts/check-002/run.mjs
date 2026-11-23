// `vp run check:gpu`: serves scripts/check-002/ with Vite and runs it in headless Chromium, where
// WebGPU runs on SwiftShader (no GPU needed), so the checks exercise the real map generator and
// the real cloth sim. ONLY=sheets or ONLY=stability runs one of the two pages.
import { chromium } from 'playwright'
import { createServer } from 'vite-plus'

const only = process.env.ONLY
const server = await createServer({
  server: { port: 0, host: '127.0.0.1', watch: null, hmr: false },
  logLevel: 'error',
})
await server.listen()
const { port } = server.httpServer.address()
const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader'],
})
const timeout = 30 * 60 * 1000
let failed = false
try {
  if (only !== 'stability') {
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.error('page error:', e.message))
    const started = Date.now()
    await page.goto(`http://127.0.0.1:${port}/scripts/check-002/`)
    await page.waitForFunction(() => window.__check, null, { timeout, polling: 1000 })
    const result = await page.evaluate(() => window.__check)
    if (result.error) {
      console.error(result.error)
      failed = true
    } else {
      const seconds = ((Date.now() - started) / 1000).toFixed(0)
      console.log(
        `checked ${result.configs} map-driven sheets (5 viewports x 3 tilings x 6 shapes x bubbles-across/margin/size/spacing) in ${seconds}s`,
      )
      for (const [k, v] of Object.entries(result.summary)) console.log(`  ${k}: ${v}`)
      if (result.failures.length) {
        console.error(result.failures.slice(0, 25).join('\n'))
        console.error(`${result.failures.length} failures`)
        failed = true
      }
    }
    await page.close()
  }

  if (only !== 'sheets') {
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.error('page error:', e.message))
    const started = Date.now()
    const query = process.env.QUERY ? `?${process.env.QUERY}` : ''
    await page.goto(`http://127.0.0.1:${port}/scripts/check-002/stability.html${query}`)
    await page.waitForFunction(() => window.__stability, null, { timeout, polling: 1000 })
    const results = await page.evaluate(() => window.__stability)
    if (results.error) {
      console.error(results.error)
      failed = true
    } else {
      const seconds = ((Date.now() - started) / 1000).toFixed(0)
      console.log(`popped one bubble and let go in ${results.length} pop-stability scenarios in ${seconds}s`)
      for (const r of results) {
        const last = r.samples[r.samples.length - 1]
        const peak = Math.max(...r.samples.map((s) => s.energy))
        const pops = Math.max(...r.samples.map((s) => s.popped))
        console.log(
          `  ${r.failures.length ? 'FAIL' : 'ok  '} ${r.name}: ${r.bubbles} bubbles, ${pops} popped, energy ${peak.toExponential(1)} -> ${last.energy.toExponential(1)} in ${last.t}s, height ${last.height.toFixed(2)} cm, max in-plane drift ${last.spread.toFixed(2)} cm`,
        )
        for (const f of r.failures) console.error(`       ${f}`)
        if (r.failures.length) {
          failed = true
          for (const h of r.hot ?? []) console.error(`       hot: ${h}`)
        }
      }
    }
    await page.close()
  }
} finally {
  await browser.close()
  await server.close()
}
process.exit(failed ? 1 : 0)
