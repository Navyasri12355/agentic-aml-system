import React, { useRef, useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

export default function TransactionGraph({ subgraph, targetAccount }) {
  const fgRef = useRef();
  const containerRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [dimensions, setDimensions] = useState({ width: 800, height: 384 });

  useEffect(() => {
    if (containerRef.current) {
      setDimensions({
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight
      });
      
      const handleResize = () => {
        if (containerRef.current) {
          setDimensions({
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight
          });
        }
      };
      
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }
  }, []);

  useEffect(() => {
    if (!subgraph) return;
    
    // nx.node_link_data uses 'links', but we might receive 'edges'
    const edges = subgraph.links || subgraph.edges || [];
    const nodes = subgraph.nodes || [];
    
    setGraphData({
      nodes: nodes.map(n => ({ ...n, id: n.id || n.name || n.label })),
      links: edges.map(e => ({ ...e, source: e.source, target: e.target, amount: e.amount || 0 }))
    });
  }, [subgraph]);

  return (
    <div ref={containerRef} className="w-full h-96 border border-gray-700 rounded overflow-hidden bg-gray-900 mt-4 relative">
      <div className="absolute top-2 left-2 z-10 text-xs text-gray-300 bg-gray-950/80 p-3 rounded border border-gray-700 shadow-lg pointer-events-none">
        <h3 className="font-bold mb-2 border-b border-gray-600 pb-1">Graph Legend</h3>
        <div className="flex items-center gap-2 mb-1">
          <span className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]"></span> 
          <span>Subject Account ({targetAccount})</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]"></span> 
          <span>Connected Account</span>
        </div>
        <p className="mt-2 text-gray-500 italic">Drag nodes to move. Scroll to zoom.</p>
      </div>
      
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeId="id"
        nodeLabel="id"
        nodeColor={node => String(node.id) === String(targetAccount) ? '#ef4444' : '#3b82f6'}
        nodeRelSize={6}
        linkColor={() => 'rgba(156, 163, 175, 0.4)'}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkWidth={link => Math.min(Math.max(link.amount / 1000, 1), 5)}
        linkCurvature={0.2}
        linkLabel={link => `Amount: $${link.amount?.toLocaleString() || 0}`}
        d3VelocityDecay={0.3}
        onEngineStop={() => fgRef.current.zoomToFit(400, 50)}
      />
    </div>
  );
}
