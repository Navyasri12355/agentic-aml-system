import React, { useState } from 'react';
import { Search, FlaskConical } from 'lucide-react';

export default function Sidebar({ currentPage, setCurrentPage }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div 
      className={`fixed left-4 top-1/2 -translate-y-1/2 z-40 transition-all duration-300 ease-in-out flex flex-col ${isExpanded ? 'w-64' : 'w-[60px]'}`}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
    >
      {/* Glassmorphic Background Panel */}
      <div className="absolute inset-0 bg-[#FFB6C1]/50 dark:bg-[#FF69B4]/20 backdrop-blur-xl rounded-2xl border border-white/50 dark:border-white/10 shadow-[0_8px_32px_0_rgba(0,0,0,0.15)]"></div>
      
      {/* Content */}
      <div className="relative z-10 flex flex-col gap-4 py-6 px-2.5">
        
        {/* Nav Item: Investigation */}
        <button 
          onClick={() => setCurrentPage('investigation')}
          className={`flex items-center gap-4 px-3 py-3 rounded-xl transition-colors whitespace-nowrap overflow-hidden
            ${currentPage === 'investigation' ? 'bg-black/10 dark:bg-white/20 shadow-sm' : 'hover:bg-black/5 dark:hover:bg-white/10'}`}
        >
          <Search className="w-5 h-5 shrink-0 text-black dark:text-white" />
          <span className={`font-heading font-bold text-sm text-black dark:text-white transition-opacity duration-300 ${isExpanded ? 'opacity-100' : 'opacity-0'}`}>
            Investigation
          </span>
        </button>

        {/* Nav Item: Research */}
        <button 
          onClick={() => setCurrentPage('research')}
          className={`flex items-center gap-4 px-3 py-3 rounded-xl transition-colors whitespace-nowrap overflow-hidden
            ${currentPage === 'research' ? 'bg-black/10 dark:bg-white/20 shadow-sm' : 'hover:bg-black/5 dark:hover:bg-white/10'}`}
        >
          <FlaskConical className="w-5 h-5 shrink-0 text-black dark:text-white" />
          <span className={`font-heading font-bold text-sm text-black dark:text-white transition-opacity duration-300 ${isExpanded ? 'opacity-100' : 'opacity-0'}`}>
            Research Behind Amlo
          </span>
        </button>

      </div>
    </div>
  );
}
