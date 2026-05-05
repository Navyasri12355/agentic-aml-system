import React, { useEffect, useState } from 'react';
import ReactFlow, { Background, Controls, MarkerType, useNodesState, useEdgesState, Handle, Position } from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css';

// A custom minimalist node with a hover tooltip for Risk Score
const GlassNode = ({ data }) => {
  const isTarget = data.isTarget;
  return (
    <div className={`group relative px-4 py-2 shadow-lg rounded-lg backdrop-blur-md border ${isTarget ? 'bg-brand-ochre/20 border-brand-ochre/50 text-brand-ochre' : 'bg-white/60 dark:bg-black/60 border-black/10 dark:border-white/10 text-brand-dark dark:text-brand-light'}`}>
      {/* React Flow requires Handles on custom nodes so edges can connect */}
      <Handle type="target" position={Position.Top} className="opacity-0 w-full h-full absolute top-0 left-0 border-none bg-transparent" />

      <div className="font-heading font-bold text-xs">{data.label}</div>

      {/* Tooltip (visible on hover) */}
      <div className="absolute -top-10 left-1/2 -translate-x-1/2 hidden group-hover:block w-max bg-brand-dark/90 text-white text-[10px] px-2 py-1 rounded shadow-xl z-50 pointer-events-none">
        Risk Score: {data.risk_score !== undefined ? data.risk_score.toFixed(3) : 'N/A'}
      </div>

      <Handle type="source" position={Position.Bottom} className="opacity-0 w-full h-full absolute top-0 left-0 border-none bg-transparent" />
    </div>
  );
};

const nodeTypes = { glass: GlassNode };

// Dagre layout engine helper
const getLayoutedElements = (nodes, edges, direction = 'TB') => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  
  const nodeWidth = 120;
  const nodeHeight = 50;

  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = direction === 'TB' ? Position.Top : Position.Left;
    node.sourcePosition = direction === 'TB' ? Position.Bottom : Position.Right;

    // We are shifting the dagre node position (anchor=center center) to the top left
    // so it matches the React Flow node anchor point (top left).
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };
    return node;
  });

  return { nodes, edges };
};

export default function NetworkGraph({ subgraph, targetAccountId }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [isAutoLayout, setIsAutoLayout] = useState(false);

  useEffect(() => {
    if (!subgraph || !subgraph.nodes) {
      setNodes([]);
      setEdges([]);
      return;
    }

    const numNodes = subgraph.nodes.length;
    const radius = Math.min(250, numNodes * 20);

    // 1. Create base nodes
    const initialNodes = subgraph.nodes.map((node, i) => {
      const isTarget = String(node.id) === String(targetAccountId);
      
      let x = 0;
      let y = 0;
      
      // If auto-layout is disabled, use the original circular math
      if (!isAutoLayout) {
        const angle = (i / numNodes) * 2 * Math.PI;
        x = isTarget ? 0 : Math.cos(angle) * radius;
        y = isTarget ? 0 : Math.sin(angle) * radius;
      }

      return {
        id: String(node.id),
        type: 'glass',
        position: { x: x + 300, y: y + 200 }, // Initial offset
        data: { label: String(node.id), isTarget, risk_score: node.risk_score }
      };
    });

    const nodeIds = new Set(initialNodes.map(n => n.id));

    // 2. Create and filter edges
    const initialEdges = (subgraph.edges || [])
      .filter(edge => nodeIds.has(String(edge.source)) && nodeIds.has(String(edge.target)))
      .map((edge, i) => ({
        id: `e${i}-${edge.source}-${edge.target}`,
        source: String(edge.source),
        target: String(edge.target),
        type: 'smoothstep',
        animated: true,
        label: `Tx: ${edge.transaction_id || 'N/A'} | $${edge.amount || 0}`,
        labelStyle: { fill: '#fff', fontSize: 9, fontWeight: 600 },
        labelBgStyle: { fill: 'rgba(10,10,10,0.8)', padding: 4, borderRadius: 4 },
        style: { stroke: 'rgba(245, 158, 11, 0.6)', strokeWidth: 2 },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: 'rgba(245, 158, 11, 0.6)',
        },
      }));

    if (isAutoLayout) {
      // 3. Apply Dagre auto-layout
      const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
        initialNodes,
        initialEdges,
        'TB' // Top to Bottom hierarchical layout
      );
      setNodes(layoutedNodes);
      setEdges(layoutedEdges);
    } else {
      setNodes(initialNodes);
      setEdges(initialEdges);
    }
  }, [subgraph, targetAccountId, setNodes, setEdges, isAutoLayout]);

  return (
    <div className="glass-panel rounded-3xl w-full h-[500px] overflow-hidden relative">
      <div className="absolute top-4 left-6 z-10 flex flex-col items-start gap-2">
        <div>
          <h3 className="font-heading font-bold text-lg">Transaction Subgraph</h3>
          <p className="text-xs opacity-50">Tracing laundering patterns up to 2 hops.</p>
        </div>
        <button 
          onClick={() => setIsAutoLayout(!isAutoLayout)}
          className={`px-3 py-1 text-xs font-bold rounded-full transition-colors ${isAutoLayout ? 'bg-brand-ochre text-black' : 'bg-black/10 dark:bg-white/10 text-brand-dark dark:text-brand-light'}`}
        >
          {isAutoLayout ? 'Hierarchical Layout (Auto)' : 'Circular Layout (Free)'}
        </button>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        className="bg-transparent"
      >
        <Background color="#ccc" gap={16} size={1} />
        <Controls className="glass-panel border-none fill-brand-dark dark:fill-brand-light" />
      </ReactFlow>
    </div>
  );
}
