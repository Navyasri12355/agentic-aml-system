import React, { useEffect } from 'react'
import GlobalShell from './components/layout/GlobalShell'
import Investigate from './pages/Investigate'
import useTheme from './hooks/useTheme'

function App() {
  const { theme } = useTheme();

  // Ensure the initial theme is applied to the document
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, []);

  return (
    <GlobalShell>
      <Investigate />
    </GlobalShell>
  )
}

export default App