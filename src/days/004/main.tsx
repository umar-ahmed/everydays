import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'dialkit/styles.css'
import '../../styles.css'
import { Light } from './Light.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Light />
  </StrictMode>,
)
