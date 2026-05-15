import React, { useEffect, useState } from 'react'
import GlobalShell from './components/layout/GlobalShell'
import Investigate from './pages/Investigate'
import Research from './pages/Research'
import useTheme from './hooks/useTheme'

function App() {
  const { theme } = useTheme();
  const [currentPage, setCurrentPage] = useState('investigation');

  // Ensure the initial theme is applied to the document
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  return (
    <GlobalShell currentPage={currentPage} setCurrentPage={setCurrentPage}>
      {currentPage === 'investigation' ? <Investigate /> : <Research />}
    </GlobalShell>
  )
}

export default App