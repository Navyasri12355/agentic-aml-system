import React from 'react';

export default function Research() {
  return (
    <div className="w-full flex flex-col gap-6 pb-12 animate-fade-in pl-16">
      <div className="flex flex-col gap-2">
        <h1 className="font-heading font-black text-4xl text-brand-dark dark:text-brand-light">
          Research Behind Amlo
        </h1>
        <p className="font-body text-black/60 dark:text-white/60 max-w-2xl">
          Understanding the methodology, models, and architectural decisions powering our Anti-Money Laundering system.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
        <div className="glass-panel rounded-3xl p-8">
          <h2 className="font-heading font-bold text-xl mb-4 text-brand-dark dark:text-brand-light">Graph Intelligence</h2>
          <p className="font-body text-sm text-black/70 dark:text-white/70 leading-relaxed">
            Amlo uses an advanced Directed Graph (DiGraph) representation of financial transactions. By modeling accounts as nodes and transactions as edges, we employ Breadth-First Search (BFS) expansion with time-window filtering to build highly localized subgraphs. This allows us to instantly detect cyclical money flows and funneling behaviors typical of structuring.
          </p>
        </div>

        <div className="glass-panel rounded-3xl p-8">
          <h2 className="font-heading font-bold text-xl mb-4 text-brand-dark dark:text-brand-light">Hybrid Anomaly Detection</h2>
          <p className="font-body text-sm text-black/70 dark:text-white/70 leading-relaxed">
            Our detection pipeline uses a hybrid approach: an Isolation Forest model flags statistical anomalies in transaction behavior, while a Random Forest Classifier cross-verifies against known laundering topologies. This significantly reduces false positive rates compared to traditional rule-based threshold systems.
          </p>
        </div>
      </div>
    </div>
  );
}
