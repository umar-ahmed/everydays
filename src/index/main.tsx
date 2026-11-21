import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../styles.css'
import { Index } from './Index.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Index />
  </StrictMode>,
)
