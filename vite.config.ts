import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type Plugin } from 'vite'
import { days, site } from './src/days.ts'

const root = import.meta.dirname

function resolveSiteUrl(env: Record<string, string>): string {
  const explicit = env.SITE_URL || env.VITE_SITE_URL
  if (explicit) return explicit.replace(/\/+$/, '')
  const vercelHost = env.VERCEL_PROJECT_PRODUCTION_URL || env.VERCEL_URL
  if (vercelHost) return `https://${vercelHost}`
  return 'http://localhost:43118'
}

// The site URL's path doubles as Vite's `base`, so a sub-path deploy
// (e.g. https://user.github.io/everydays) needs only SITE_URL.
const basePath = (siteUrl: string) => new URL(siteUrl).pathname.replace(/\/*$/, '/')

const escapeHtml = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

function openGraph(siteUrl: string, base: string): Plugin {
  return {
    name: 'everydays-open-graph',
    transformIndexHtml(html, ctx) {
      const pagePath = ctx.path.startsWith(base) ? `/${ctx.path.slice(base.length)}` : ctx.path
      const slug = pagePath.match(/^\/(\d{3})\//)?.[1]
      const day = slug ? days.find((d) => d.slug === slug) : undefined
      const path = day ? `/${day.slug}/` : '/'
      const title = day ? `day ${day.slug} / ${day.title} · ${site.name}` : site.name
      const description = day ? day.concept : site.description
      const meta: Record<string, string> = {
        'og:type': 'website',
        'og:site_name': site.name,
        'og:title': title,
        'og:description': description,
        'og:url': `${siteUrl}${path}`,
        'og:image': `${siteUrl}${path}og.png`,
        'og:image:width': '1200',
        'og:image:height': '630',
        'twitter:card': 'summary_large_image',
        'twitter:title': title,
        'twitter:description': description,
        'twitter:image': `${siteUrl}${path}og.png`,
      }
      const tags = [
        `<title>${escapeHtml(title)}</title>`,
        `<meta name="description" content="${escapeHtml(description)}" />`,
        `<link rel="canonical" href="${siteUrl}${path}" />`,
        ...Object.entries(meta).map(([key, value]) => {
          const attr = key.startsWith('twitter:') ? 'name' : 'property'
          return `<meta ${attr}="${key}" content="${escapeHtml(value)}" />`
        }),
      ]
      return html.replace('<!-- head:meta -->', tags.join('\n    '))
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = { ...process.env, ...loadEnv(mode, root, '') } as Record<string, string>
  const siteUrl = resolveSiteUrl(env)
  const base = basePath(siteUrl)

  const input: Record<string, string> = { index: resolve(root, 'index.html') }
  for (const day of days) {
    const page = resolve(root, day.slug, 'index.html')
    if (!existsSync(page)) throw new Error(`days.ts lists ${day.slug} but ${day.slug}/index.html is missing`)
    input[day.slug] = page
  }

  return {
    base,
    plugins: [react(), tailwindcss(), openGraph(siteUrl, base)],
    // three's WebGPU build alone is ~800 kB; only day 002 loads it.
    build: { rollupOptions: { input }, chunkSizeWarningLimit: 1200 },
    server: { port: 43117, host: true },
    preview: { port: 43118, host: true },
  }
})
