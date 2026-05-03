import React from 'react';
import { motion } from 'framer-motion';

export default function RiskGauge({ score, tier }) {
  // Map score (0-1) to an angle between -90 and 90 degrees
  const angle = (score * 180) - 90;
  
  let colorClass = "text-brand-sage";
  let bgClass = "bg-brand-sage/20 border-brand-sage/30";
  
  if (tier === "HIGH") {
    colorClass = "text-brand-ochre";
    bgClass = "bg-brand-ochre/20 border-brand-ochre/30";
  } else if (tier === "MEDIUM") {
    colorClass = "text-brand-sky";
    bgClass = "bg-brand-sky/20 border-brand-sky/30";
  }

  return (
    <div className="glass-panel p-8 rounded-3xl flex flex-col items-center justify-center relative overflow-hidden">
      <h3 className="font-heading text-lg font-bold mb-6 text-black/80 dark:text-white/80">Overall Risk Score</h3>
      
      <div className="relative w-48 h-24 overflow-hidden mb-4">
        {/* Arc Background */}
        <div className="absolute top-0 left-0 w-48 h-48 rounded-full border-[12px] border-black/5 dark:border-white/5" />
        
        {/* Needle */}
        <motion.div
          className="absolute bottom-0 left-24 w-1 h-24 origin-bottom bg-black dark:bg-white rounded-full z-10"
          initial={{ rotate: -90 }}
          animate={{ rotate: angle }}
          transition={{ type: "spring", stiffness: 50, damping: 15, delay: 0.2 }}
        />
        
        {/* Center Pivot */}
        <div className="absolute bottom-[-6px] left-[calc(50%-6px)] w-3 h-3 rounded-full bg-black dark:bg-white z-20" />
      </div>

      <div className="flex flex-col items-center">
        <span className="font-heading text-5xl font-black tracking-tighter">
          {(score * 100).toFixed(1)}
        </span>
        <div className={`mt-3 px-4 py-1 rounded-full text-xs font-bold tracking-widest uppercase border ${colorClass} ${bgClass}`}>
          {tier} RISK
        </div>
      </div>
    </div>
  );
}
