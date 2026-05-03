import React from 'react';
import useInvestigationStore from '../store/useInvestigationStore';
import StatusTracker from '../components/investigation/StatusTracker';
import RiskGauge from '../components/investigation/RiskGauge';
import NetworkGraph from '../components/investigation/NetworkGraph';
import NarrativeBox from '../components/investigation/NarrativeBox';
import { Search, UploadCloud } from 'lucide-react';

export default function Investigate() {
  const { 
    accountId, setAccountId, 
    selectedFile, setSelectedFile, 
    isInvestigating, startInvestigation, 
    error, result 
  } = useInvestigationStore();

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    startInvestigation();
  };

  return (
    <div className="w-full flex-1 flex flex-col gap-8 pb-12">
      
      {/* Header & Form */}
      <div className="flex flex-col items-center mt-12 mb-8">
        <h1 className="text-4xl font-heading font-black tracking-tight mb-2">New Investigation</h1>
        <p className="text-black/50 dark:text-white/50 font-body mb-8">Initiate an agentic AML sweep on a specific account.</p>
        
        <form onSubmit={handleSubmit} className="w-full max-w-2xl glass-panel p-2 rounded-full flex items-center gap-4">
          <div className="flex-1 flex items-center pl-6">
            <Search className="w-5 h-5 text-black/30 dark:text-white/30 mr-3" />
            <input 
              type="text" 
              placeholder="Enter Account ID (e.g., 800737690)" 
              required
              value={accountId}
              onChange={(e) => setAccountId(e.target.value)}
              className="w-full bg-transparent border-none outline-none font-body text-brand-dark dark:text-brand-light placeholder-black/30 dark:placeholder-white/30"
              disabled={isInvestigating}
            />
          </div>
          
          <div className="flex items-center gap-2 pr-2">
            <label className="cursor-pointer p-3 rounded-full hover:bg-black/5 dark:hover:bg-white/5 transition-colors relative group">
              <UploadCloud className={`w-5 h-5 ${selectedFile ? 'text-brand-sky' : 'text-black/50 dark:text-white/50'}`} />
              <input type="file" accept=".csv" className="hidden" onChange={handleFileChange} disabled={isInvestigating} />
              
              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity pointer-events-none">
                {selectedFile ? selectedFile.name : "Optional CSV override"}
              </div>
            </label>
            
            <button 
              type="submit" 
              disabled={isInvestigating || !accountId}
              className="bg-brand-dark dark:bg-brand-light text-brand-light dark:text-brand-dark px-6 py-3 rounded-full font-heading font-bold text-sm tracking-wide disabled:opacity-50 transition-all hover:scale-105 active:scale-95"
            >
              RUN SWEEP
            </button>
          </div>
        </form>
        {error && <p className="text-red-500 mt-4 text-sm font-bold bg-red-500/10 px-4 py-2 rounded-lg border border-red-500/20">{error}</p>}
      </div>

      {/* Loading State */}
      {isInvestigating && <StatusTracker />}

      {/* Results State */}
      {!isInvestigating && result && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in slide-in-from-bottom-8 duration-700">
          {/* Left Column: Risk Gauge & Patterns */}
          <div className="lg:col-span-1 flex flex-col gap-6">
            <RiskGauge score={result.risk_score} tier={result.risk_tier} />
            
            <div className="glass-panel p-6 rounded-3xl flex-1">
              <h3 className="font-heading font-bold mb-4">Detected Patterns</h3>
              <div className="flex flex-wrap gap-2">
                {result.detected_patterns && result.detected_patterns.length > 0 ? (
                  result.detected_patterns.map((pattern) => (
                    <span key={pattern} className="px-3 py-1 bg-brand-ochre/10 border border-brand-ochre/20 text-brand-ochre text-xs font-bold rounded-full">
                      {pattern}
                    </span>
                  ))
                ) : (
                  <span className="text-black/40 dark:text-white/40 text-sm">No suspicious patterns detected.</span>
                )}
              </div>
            </div>
          </div>

          {/* Right Column: Graph & Narrative */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            <NetworkGraph subgraph={result.subgraph} targetAccountId={accountId} />
            <div className="h-[300px]">
              <NarrativeBox narrative={result.sar_narrative || result.exit_summary} />
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
