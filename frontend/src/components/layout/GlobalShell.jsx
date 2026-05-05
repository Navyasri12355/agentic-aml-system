import React from 'react';
import Navbar from './Navbar';

export default function GlobalShell({ children }) {
  return (
    <div className="min-h-screen w-full relative overflow-hidden transition-colors duration-500">
      {/* Dynamic Background Gradients */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-brand-sky/20 dark:bg-brand-sky/10 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-brand-sage/20 dark:bg-brand-sage/10 blur-[120px]" />
        <div className="absolute top-[40%] left-[60%] w-[30%] h-[30%] rounded-full bg-brand-ochre/10 dark:bg-brand-ochre/5 blur-[120px]" />
      </div>

      {/* Noise Overlay */}
      <div 
        className="fixed inset-0 pointer-events-none z-0 opacity-40 dark:opacity-20 mix-blend-overlay"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
        }}
      />

      <Navbar />

      {/* Main Content Area */}
      <main className="relative z-10 pt-24 px-6 max-w-screen-2xl mx-auto min-h-screen flex flex-col">
        {children}
      </main>
    </div>
  );
}
