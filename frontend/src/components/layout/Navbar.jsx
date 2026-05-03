import React from 'react';
import { Sun, Moon, ShieldAlert, ShieldCheck } from 'lucide-react';
import useTheme from '../../hooks/useTheme';

export default function Navbar() {
  const { theme, toggleTheme } = useTheme();

  return (
    <nav className="fixed top-0 w-full z-50 glass-nav">
      <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center justify-between">
        
        {/* Left: Branding & Logos */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-brand-dark dark:text-brand-light">
            <ShieldAlert className="w-6 h-6 text-brand-ochre" />
            <div className="h-6 w-px bg-black/20 dark:bg-white/20"></div>
            <ShieldCheck className="w-6 h-6 text-brand-sage" />
          </div>
          <span className="font-heading font-black tracking-[0.2em] text-xl uppercase text-brand-dark dark:text-brand-light">
            amlo
          </span>
        </div>

        {/* Center: Search/Action Bar Placeholder */}
        <div className="hidden md:flex flex-1 max-w-xl mx-8">
          <div className="w-full glass-panel rounded-full h-10 px-4 flex items-center text-sm text-black/50 dark:text-white/50 border-black/10 dark:border-white/10">
            <span className="font-body">Search account ID or transaction hash...</span>
          </div>
        </div>

        {/* Right: Profile & Theme Toggle */}
        <div className="flex items-center gap-6">
          <button 
            onClick={toggleTheme}
            className="p-2 rounded-full hover:bg-black/5 dark:hover:bg-white/5 transition-colors text-brand-dark dark:text-brand-light"
          >
            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          
          <div className="flex items-center gap-3">
            <div className="text-right hidden sm:block">
              <div className="font-heading font-bold text-sm text-brand-dark dark:text-brand-light">Jane Smith</div>
              <div className="font-body text-xs text-black/60 dark:text-white/60">Senior AML Analyst</div>
            </div>
            <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-brand-sky to-brand-sage p-[2px]">
              <div className="w-full h-full bg-white dark:bg-brand-dark rounded-full flex items-center justify-center font-heading font-bold text-sm">
                JS
              </div>
            </div>
          </div>
        </div>

      </div>
    </nav>
  );
}
