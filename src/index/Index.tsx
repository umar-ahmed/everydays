import { motion, useReducedMotion } from 'motion/react'
import { days, site } from '../days.ts'

const nextSlug = String(days.length + 1).padStart(3, '0')

export function Index() {
  const reduced = useReducedMotion()
  const sorted = [...days].reverse()

  return (
    <main className="mx-auto flex min-h-full max-w-3xl flex-col px-6 py-16 sm:px-10 sm:py-24">
      <header className="mb-14 sm:mb-20">
        <p className="font-mono text-xs tracking-[0.2em] text-fuchsia-300/70 uppercase">
          {days.length} {days.length === 1 ? 'day' : 'days'} so far
        </p>
        <h1 className="mt-3 text-5xl font-semibold tracking-tight sm:text-7xl">
          every<span className="text-fuchsia-400">days</span>
        </h1>
        <p className="mt-4 max-w-md text-base text-white/60 sm:text-lg">{site.description}</p>
      </header>

      <ol className="flex flex-col border-t border-white/10">
        {sorted.map((day, i) => (
          <motion.li
            key={day.slug}
            initial={reduced ? { opacity: 0 } : { opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', visualDuration: 0.4, bounce: 0.15, delay: i * 0.05 }}
            className="border-b border-white/10"
          >
            <a
              href={`/${day.slug}/`}
              className="group flex items-baseline gap-5 py-6 outline-none focus-visible:bg-white/5 sm:gap-8"
            >
              <span className="font-mono text-sm text-fuchsia-300/80 tabular-nums">{day.slug}</span>
              <span className="flex min-w-0 flex-1 flex-col gap-1">
                <span className="text-xl font-medium transition-colors group-hover:text-fuchsia-300 sm:text-2xl">
                  {day.title}
                </span>
                <span className="text-sm text-white/50 sm:text-base">{day.concept}</span>
              </span>
              <span
                aria-hidden
                className="text-white/30 transition-transform group-hover:translate-x-1 group-hover:text-fuchsia-300"
              >
                →
              </span>
            </a>
          </motion.li>
        ))}
        <li className="flex items-baseline gap-5 py-6 text-white/25 sm:gap-8">
          <span className="font-mono text-sm tabular-nums">{nextSlug}</span>
          <span className="text-xl sm:text-2xl">tomorrow</span>
        </li>
      </ol>

      <footer className="mt-auto pt-16 font-mono text-xs text-white/30">
        one toy a day · ~30 minutes each
      </footer>
    </main>
  )
}
