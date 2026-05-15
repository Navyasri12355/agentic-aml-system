import React from 'react';
import { X, FileText, AudioLines } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function DocumentViewer({ isOpen, onClose, docData }) {
  if (!isOpen || !docData) return null;

  return (
    <>
      {/* Backdrop overlay */}
      <div 
        className="fixed inset-0 bg-black/40 dark:bg-black/60 backdrop-blur-sm z-50 transition-opacity duration-300"
        onClick={onClose}
      />
      
      {/* Sidebar Panel */}
      <div className={`fixed top-0 right-0 h-full w-[400px] md:w-[600px] bg-white/90 dark:bg-[#1A1A1D]/90 backdrop-blur-2xl border-l border-white/20 shadow-2xl z-50 flex flex-col transition-transform duration-500 ease-out transform ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
        
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-black/10 dark:border-white/10">
          <div className="flex items-center gap-3 text-brand-dark dark:text-brand-light">
            {docData.type === 'audio' ? <AudioLines className="w-5 h-5 text-[#F13E93]" /> : <FileText className="w-5 h-5 text-[#F13E93]" />}
            <h2 className="font-heading font-bold text-lg">{docData.title}</h2>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded-full hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-black/60 dark:text-white/60"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
          {docData.type === 'audio' ? (
            <div className="glass-panel p-6 rounded-3xl flex flex-col items-center justify-center h-48 border border-[#F13E93]/20">
              <AudioLines className="w-12 h-12 text-[#F13E93] mb-4 opacity-50" />
              <p className="font-body text-sm text-black/60 dark:text-white/60 text-center mb-4">Interview Audio Recording</p>
              <audio controls className="w-full max-w-sm rounded-full">
                <source src={docData.url} type="audio/mp4" />
                <source src={docData.url} type="audio/mpeg" />
                Your browser does not support the audio element.
              </audio>
            </div>
          ) : docData.type === 'markdown' ? (
            <div className="glass-panel p-8 rounded-3xl overflow-y-auto border border-[#F13E93]/20 flex flex-col font-body text-black/80 dark:text-white/80">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown>{docData.content || ''}</ReactMarkdown>
              </div>
            </div>
          ) : (
            <div className="glass-panel rounded-3xl overflow-hidden h-full min-h-[700px] border border-[#F13E93]/20 flex flex-col">
              <iframe 
                src={docData.url} 
                title="Document Viewer"
                className="w-full h-full flex-1 border-none"
              />
            </div>
          )}
        </div>
      </div>
    </>
  );
}
