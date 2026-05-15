import React, { useState } from 'react';
import { 
  Building2, Mic, Map, Database, GitGraph, 
  Workflow, FileText, CheckCircle 
} from 'lucide-react';
import DocumentViewer from '../components/layout/DocumentViewer';

const TIMELINE_ENTRIES = [
  {
    id: 1,
    title: "Interview with Bank Official",
    description: "We spoke to a Bank Official at Kotak Bank, RV College of Engineering regarding how their bank currently deals with traditional money laundering through bank transactions",
    image: "../timeline/meeting.jpeg",
    icon: Building2
  },
  {
    id: 2,
    title: "Audio recording and Transcript of Interview",
    description: "Listen to the raw interview audio or read the full PDF transcript.",
    hasDocuments: true,
    audioUrl: "../timeline/Voice.mp3",
    transcriptUrl: "../timeline/Transcript.pdf",
    icon: Mic
  },
  {
    id: 3,
    title: "Implementation Plan",
    description: "Interviews inspired us to create a comprehensive implementation plan, with 5 phases of work.",
    // image: "/timeline/image.png",
    icon: Map
  },
  {
    id: 4,
    title: "Phase 1 work: Data Foundation & Detection Agent",
    description: "Built data ingestion and cleaning pipeline, trained an Isolation Forest model to flag anomalous transactions with scoring and baseline evaluation.",
    image: "../timeline/phase1.jpeg",
    icon: Database
  },
  {
    id: 5,
    title: "Phase 2 work: Graph Construction & Investigation Agent",
    description: "Constructed directed transaction subgraphs per flagged account, extracted graph features, detected laundering patterns, and computed weighted risk scores with tier assignment.",
    image: "../timeline/phase2.jpeg",
    icon: GitGraph
  },
  {
    id: 6,
    title: "Phase 3 work: LangGraph Orchestration",
    description: "Wired all agents into a LangGraph state machine with shared typed state and conditional routing based on risk tier.",
    image: "../timeline/phase3.jpeg",
    icon: Workflow
  },
  {
    id: 7,
    title: "Phase 4 & 5 work: Explanation Agent & SAR Report Generation, Frontend",
    description: "Integrated Groq LLM to dynamically generate structured, tone-adjusted Suspicious Activity Reports from graph evidence and risk data. Built a React + Vite investigator UI with FastAPI backend, and ran full evaluation metrics against a held-out test set.",
    image: "../timeline/phase4.jpeg",
    icon: FileText
  },
];

export default function Research() {
  const [viewerState, setViewerState] = useState({ isOpen: false, docData: null });

  const openDocument = (type, url, title) => {
    setViewerState({
      isOpen: true,
      docData: { type, url, title }
    });
  };

  return (
    <div className="w-full flex flex-col gap-6 pb-24 animate-fade-in">
      
      {/* Header Section */}
      <div className="flex flex-col gap-2 pt-6">
        <h1 className="font-heading font-black text-4xl text-brand-dark dark:text-brand-light">
          Research Behind Amlo
        </h1>
        <p className="font-body text-black/60 dark:text-white/60 max-w-2xl">
          Understanding the methodology, models, and architectural decisions powering our Anti-Money Laundering system.
        </p>
      </div>

      {/* Initial Concept Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
        <div className="glass-panel rounded-3xl p-8 hover:scale-[1.02] transition-transform duration-300">
          <h2 className="font-heading font-bold text-xl mb-4 text-brand-dark dark:text-brand-light">Graph Intelligence</h2>
          <p className="font-body text-sm text-black/70 dark:text-white/70 leading-relaxed">
            Amlo uses an advanced Directed Graph (DiGraph) representation of financial transactions. By modeling accounts as nodes and transactions as edges, we employ Breadth-First Search (BFS) expansion with time-window filtering to build highly localized subgraphs. This allows us to instantly detect cyclical money flows and funneling behaviors typical of structuring.
          </p>
        </div>

        <div className="glass-panel rounded-3xl p-8 hover:scale-[1.02] transition-transform duration-300">
          <h2 className="font-heading font-bold text-xl mb-4 text-brand-dark dark:text-brand-light">Hybrid Anomaly Detection</h2>
          <p className="font-body text-sm text-black/70 dark:text-white/70 leading-relaxed">
            Our detection pipeline uses a hybrid approach: an Isolation Forest model flags statistical anomalies in transaction behavior, while a Random Forest Classifier cross-verifies against known laundering topologies. This significantly reduces false positive rates compared to traditional rule-based threshold systems.
          </p>
        </div>
      </div>

      {/* Vertical Timeline */}
      <div className="relative mt-20 max-w-5xl mx-auto w-full">
        {/* Center Line */}
        <div className="absolute left-1/2 transform -translate-x-1/2 h-full w-1 bg-gradient-to-b from-[#F13E93]/50 via-[#FAFFCB]/50 to-[#F13E93]/50 dark:from-[#A64D79]/50 dark:via-[#6A1E55]/50 dark:to-[#A64D79]/50 rounded-full" />

        <div className="flex flex-col gap-12 relative z-10">
          {TIMELINE_ENTRIES.map((entry, index) => {
            const isLeft = index % 2 === 0;
            const Icon = entry.icon;

            return (
              <div key={entry.id} className={`flex items-center w-full relative ${isLeft ? 'flex-row' : 'flex-row-reverse'}`}>
                
                {/* Horizontal Connector Line */}
                <div 
                  className={`absolute top-1/2 transform -translate-y-1/2 h-0.5 bg-[#F13E93]/40 dark:bg-[#A64D79]/60 
                  ${isLeft ? 'right-1/2 w-12' : 'left-1/2 w-12'} -z-10`} 
                />

                {/* Timeline Card */}
                <div className={`w-1/2 flex ${isLeft ? 'justify-end pr-12' : 'justify-start pl-12'}`}>
                  <div className="group bg-white/40 dark:bg-[#3B1C32]/40 backdrop-blur-md rounded-[3rem] p-6 border border-white/40 dark:border-white/10 shadow-lg hover:shadow-2xl hover:bg-white/60 dark:hover:bg-[#6A1E55]/60 hover:scale-105 transition-all duration-300 w-full max-w-md flex flex-col items-center text-center">
                    
                    <h3 className="font-heading font-bold text-lg text-brand-dark dark:text-brand-light mb-2">
                      {entry.title}
                    </h3>
                    <p className="font-body text-sm text-black/70 dark:text-white/70 mb-4">
                      {entry.description}
                    </p>

                    {/* Image / Documents */}
                    {entry.image && (
                      <div className="w-32 h-32 rounded-full overflow-hidden border-4 border-white/50 dark:border-[#6A1E55]/50 shadow-inner bg-black/5 dark:bg-white/5 flex items-center justify-center">
                        <img src={entry.image} alt={entry.title} className="w-full h-full object-cover" />
                      </div>
                    )}

                    {entry.hasDocuments && (
                      <div className="flex gap-4 mt-2">
                        <button 
                          onClick={() => openDocument('audio', entry.audioUrl, "Interview Audio")}
                          className="flex flex-col items-center gap-1 p-3 rounded-full hover:bg-[#F13E93]/20 dark:hover:bg-[#A64D79]/30 transition-colors text-brand-dark dark:text-brand-light"
                        >
                          <Mic className="w-6 h-6 text-[#F13E93]" />
                          <span className="text-[10px] font-bold">Audio</span>
                        </button>
                        <button 
                          onClick={() => openDocument('pdf', entry.transcriptUrl, "Interview Transcript")}
                          className="flex flex-col items-center gap-1 p-3 rounded-full hover:bg-[#F13E93]/20 dark:hover:bg-[#A64D79]/30 transition-colors text-brand-dark dark:text-brand-light"
                        >
                          <FileText className="w-6 h-6 text-[#F13E93]" />
                          <span className="text-[10px] font-bold">Transcript</span>
                        </button>
                      </div>
                    )}

                  </div>
                </div>

                {/* Center Node */}
                <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-14 h-14 rounded-full bg-white/30 dark:bg-[#3B1C32]/50 backdrop-blur-xl border-2 border-white dark:border-[#A64D79] shadow-[0_0_20px_rgba(241,62,147,0.3)] hover:scale-125 transition-transform duration-300 group">
                  <Icon className="w-6 h-6 text-[#F13E93] dark:text-brand-light drop-shadow-md group-hover:text-black dark:group-hover:text-white transition-colors" />
                </div>

                {/* Empty Spacer for the other side */}
                <div className="w-1/2"></div>
              </div>
            );
          })}
        </div>
      </div>

      <DocumentViewer 
        isOpen={viewerState.isOpen} 
        onClose={() => setViewerState({ isOpen: false, docData: null })}
        docData={viewerState.docData}
      />
    </div>
  );
}
