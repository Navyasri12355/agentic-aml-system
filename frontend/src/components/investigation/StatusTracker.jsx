import React from 'react';
import { motion } from 'framer-motion';
import { Activity, Network, FileSearch, ShieldAlert, Cpu } from 'lucide-react';

const steps = [
  { id: 'detection', label: 'Isolation Forest', icon: <Activity className="w-4 h-4" /> },
  { id: 'graph', label: 'Network Extraction', icon: <Network className="w-4 h-4" /> },
  { id: 'features', label: 'Feature Engineering', icon: <Cpu className="w-4 h-4" /> },
  { id: 'risk', label: 'Risk Scoring', icon: <ShieldAlert className="w-4 h-4" /> },
  { id: 'explanation', label: 'LLM Narrative', icon: <FileSearch className="w-4 h-4" /> },
];

export default function StatusTracker() {
  return (
    <div className="w-full max-w-4xl mx-auto my-8">
      <div className="glass-panel rounded-2xl p-6 flex justify-between items-center relative overflow-hidden">
        
        {/* Pulsing background glow to indicate "thinking" */}
        <motion.div 
          className="absolute inset-0 bg-brand-sky/10 dark:bg-brand-sky/5"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />

        {steps.map((step, index) => (
          <div key={step.id} className="relative z-10 flex flex-col items-center gap-3">
            <motion.div 
              className="w-12 h-12 rounded-full glass-panel flex items-center justify-center text-brand-sky border-brand-sky/30 shadow-[0_0_15px_rgba(96,165,250,0.4)]"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: index * 0.2 }}
            >
              {step.icon}
            </motion.div>
            <span className="font-heading text-xs font-bold uppercase tracking-wider text-black/60 dark:text-white/60">
              {step.label}
            </span>
            
            {/* Connecting line */}
            {index < steps.length - 1 && (
              <div className="absolute top-6 left-12 w-[calc(100vw/5-6rem)] md:w-32 h-[2px] bg-brand-sky/20 -z-10" />
            )}
          </div>
        ))}
      </div>
      <p className="text-center mt-4 font-body text-sm text-black/50 dark:text-white/50 animate-pulse">
        LangGraph Orchestration is actively investigating... This may take up to 20 seconds.
      </p>
    </div>
  );
}
