import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FileText } from 'lucide-react';

export default function NarrativeBox({ narrative }) {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    if (!narrative) {
      setDisplayedText('No narrative generated for this risk tier.');
      setIsTyping(false);
      return;
    }

    setDisplayedText('');
    setIsTyping(true);
    let index = 0;
    
    // Typewriter effect
    const intervalId = setInterval(() => {
      setDisplayedText((prev) => prev + narrative.charAt(index));
      index++;
      if (index === narrative.length) {
        clearInterval(intervalId);
        setIsTyping(false);
      }
    }, 10); // Adjust speed here

    return () => clearInterval(intervalId);
  }, [narrative]);

  return (
    <div className="glass-panel p-8 rounded-3xl flex flex-col h-full relative">
      <div className="flex items-center gap-3 mb-6 border-b border-black/5 dark:border-white/5 pb-4">
        <div className="p-2 rounded-full bg-brand-sky/20 text-brand-sky">
          <FileText className="w-5 h-5" />
        </div>
        <h3 className="font-heading text-lg font-bold">Suspicious Activity Report</h3>
      </div>
      
      <div className="flex-1 overflow-y-auto pr-4 custom-scrollbar">
        <div className="font-body text-sm leading-relaxed whitespace-pre-wrap text-black/80 dark:text-white/80">
          {displayedText}
          {isTyping && (
            <motion.span 
              className="inline-block w-2 h-4 ml-1 bg-brand-sky"
              animate={{ opacity: [1, 0] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
