import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FileText, Maximize2, Download } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import jsPDF from 'jspdf';

export default function NarrativeBox({ narrative, onExpand }) {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    if (!narrative) {
      setDisplayedText('No narrative generated for this risk tier.');
      setIsTyping(false);
      return;
    }

    // Removing the artificial typewriter effect entirely to fix "typos" where 
    // raw markdown syntax (like **) was displayed to the user before rendering into bold HTML.
    setDisplayedText(narrative);
    setIsTyping(false);
  }, [narrative]);

  const handleDownload = () => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.width;

    // --- ADDING A LOGO TO THE PDF ---
    // To add a logo image, place 'logo.png' in your frontend/public directory.
    // Uncomment the lines below and adjust the x/y/width/height as needed.
    //
    const img = new Image();
    img.src = '/logos/darklogo.png';
    doc.addImage(img, 'PNG', 14, 12, 10, 10);

    // Header
    doc.setFontSize(16);
    doc.setFont("helvetica", "bold");
    // If you uncomment the logo above, change '14' to '28' below so "AMLO" shifts right.
    doc.text("AMLO", 28, 20);

    doc.setFontSize(12);
    doc.setFont("helvetica", "normal");
    doc.text("Jane Smith", pageWidth - 14, 20, { align: 'right' });

    // Line separator
    doc.setDrawColor(200);
    doc.line(14, 25, pageWidth - 14, 25);

    // Body Text
    doc.setFontSize(11);

    // Strip markdown symbols (* and #) from the PDF text so they don't print raw
    const strippedNarrative = narrative ? narrative.replace(/\*/g, '').replace(/#/g, '') : "No narrative generated.";

    const splitText = doc.splitTextToSize(strippedNarrative, pageWidth - 28);
    doc.text(splitText, 14, 35);

    // Save
    doc.save("SAR_Report.pdf");
  };

  return (
    <div className="glass-panel p-8 rounded-3xl flex flex-col h-full relative">
      <div className="flex items-center justify-between mb-6 border-b border-black/5 dark:border-white/5 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-full bg-brand-sky/20 text-brand-sky">
            <FileText className="w-5 h-5" />
          </div>
          <h3 className="font-heading text-lg font-bold">Suspicious Activity Report</h3>
        </div>

        <div className="flex items-center gap-2">
          {narrative && (
            <button
              onClick={handleDownload}
              className="p-2 rounded-full hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-black/60 dark:text-white/60"
              title="Download PDF"
            >
              <Download className="w-4 h-4" />
            </button>
          )}
          {onExpand && narrative && (
            <button
              onClick={() => onExpand(narrative)}
              className="p-2 rounded-full hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-black/60 dark:text-white/60"
              title="Expand Report"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto pr-4 custom-scrollbar relative">
        <div className="font-body text-sm leading-relaxed text-black/80 dark:text-white/80">
          <ReactMarkdown
            components={{
              strong: ({ node, ...props }) => <span className="font-bold text-brand-dark dark:text-brand-light" {...props} />,
              h1: ({ node, ...props }) => <h1 className="text-xl font-bold mt-4 mb-2 text-brand-dark dark:text-brand-light" {...props} />,
              h2: ({ node, ...props }) => <h2 className="text-lg font-bold mt-3 mb-1 text-brand-dark dark:text-brand-light" {...props} />,
              p: ({ node, ...props }) => <p className="mb-3 whitespace-pre-wrap" {...props} />,
              ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-3" {...props} />,
              ol: ({ node, ...props }) => <ol className="list-decimal pl-5 mb-3" {...props} />,
            }}
          >
            {displayedText}
          </ReactMarkdown>
          {isTyping && (
            <motion.span
              className="inline-block w-2 h-4 ml-1 bg-brand-sky align-middle"
              animate={{ opacity: [1, 0] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
