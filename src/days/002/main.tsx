import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'dialkit/styles.css'
import '../../styles.css'
import { BubbleWrap } from './BubbleWrap.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BubbleWrap />
  </StrictMode>,
)
