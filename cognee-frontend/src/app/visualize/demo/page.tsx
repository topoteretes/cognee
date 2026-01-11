"use client";

import { useState } from "react";
import GraphVisualization from "@/ui/elements/GraphVisualization";
import { Edge, Node } from "@/ui/rendering/graph/types";

// Rich mock dataset representing an AI/ML knowledge graph
const mockNodes: Node[] = [
  // Core AI Concepts
  { id: "ai", label: "Artificial Intelligence", type: "Concept" },
  { id: "ml", label: "Machine Learning", type: "Concept" },
  { id: "dl", label: "Deep Learning", type: "Concept" },
  { id: "nlp", label: "Natural Language Processing", type: "Concept" },
  { id: "cv", label: "Computer Vision", type: "Concept" },
  { id: "rl", label: "Reinforcement Learning", type: "Concept" },

  // ML Algorithms
  { id: "supervised", label: "Supervised Learning", type: "Algorithm" },
  { id: "unsupervised", label: "Unsupervised Learning", type: "Algorithm" },
  { id: "svm", label: "Support Vector Machine", type: "Algorithm" },
  { id: "decision-tree", label: "Decision Tree", type: "Algorithm" },
  { id: "random-forest", label: "Random Forest", type: "Algorithm" },
  { id: "kmeans", label: "K-Means Clustering", type: "Algorithm" },
  { id: "pca", label: "Principal Component Analysis", type: "Algorithm" },

  // Deep Learning Architectures
  { id: "neural-net", label: "Neural Network", type: "Architecture" },
  { id: "cnn", label: "Convolutional Neural Network", type: "Architecture" },
  { id: "rnn", label: "Recurrent Neural Network", type: "Architecture" },
  { id: "lstm", label: "Long Short-Term Memory", type: "Architecture" },
  { id: "transformer", label: "Transformer", type: "Architecture" },
  { id: "gnn", label: "Graph Neural Network", type: "Architecture" },
  { id: "gan", label: "Generative Adversarial Network", type: "Architecture" },
  { id: "vae", label: "Variational Autoencoder", type: "Architecture" },

  // NLP Technologies
  { id: "bert", label: "BERT", type: "Technology" },
  { id: "gpt", label: "GPT", type: "Technology" },
  { id: "word2vec", label: "Word2Vec", type: "Technology" },
  { id: "attention", label: "Attention Mechanism", type: "Technology" },
  { id: "tokenization", label: "Tokenization", type: "Technology" },

  // CV Technologies
  { id: "resnet", label: "ResNet", type: "Technology" },
  { id: "yolo", label: "YOLO", type: "Technology" },
  { id: "segmentation", label: "Image Segmentation", type: "Technology" },
  { id: "detection", label: "Object Detection", type: "Technology" },

  // RL Components
  { id: "q-learning", label: "Q-Learning", type: "Algorithm" },
  { id: "dqn", label: "Deep Q-Network", type: "Architecture" },
  { id: "policy-gradient", label: "Policy Gradient", type: "Algorithm" },
  { id: "actor-critic", label: "Actor-Critic", type: "Architecture" },

  // Applications
  { id: "chatbot", label: "Chatbot", type: "Application" },
  { id: "recommendation", label: "Recommendation System", type: "Application" },
  { id: "autonomous", label: "Autonomous Vehicles", type: "Application" },
  { id: "medical-imaging", label: "Medical Imaging", type: "Application" },
  { id: "fraud-detection", label: "Fraud Detection", type: "Application" },

  // Data & Training
  { id: "dataset", label: "Training Dataset", type: "Data" },
  { id: "feature", label: "Feature Engineering", type: "Data" },
  { id: "augmentation", label: "Data Augmentation", type: "Data" },
  { id: "normalization", label: "Normalization", type: "Data" },

  // Optimization
  { id: "gradient-descent", label: "Gradient Descent", type: "Optimization" },
  { id: "adam", label: "Adam Optimizer", type: "Optimization" },
  { id: "backprop", label: "Backpropagation", type: "Optimization" },
  { id: "regularization", label: "Regularization", type: "Optimization" },
  { id: "dropout", label: "Dropout", type: "Optimization" },
];

const mockEdges: Edge[] = [
  // Core relationships
  { id: "e1", source: "ml", target: "ai", label: "is subfield of" },
  { id: "e2", source: "dl", target: "ml", label: "is subfield of" },
  { id: "e3", source: "nlp", target: "ai", label: "is subfield of" },
  { id: "e4", source: "cv", target: "ai", label: "is subfield of" },
  { id: "e5", source: "rl", target: "ml", label: "is subfield of" },

  // ML paradigms
  { id: "e6", source: "supervised", target: "ml", label: "is paradigm of" },
  { id: "e7", source: "unsupervised", target: "ml", label: "is paradigm of" },

  // ML algorithms
  { id: "e8", source: "svm", target: "supervised", label: "implements" },
  { id: "e9", source: "decision-tree", target: "supervised", label: "implements" },
  { id: "e10", source: "random-forest", target: "decision-tree", label: "ensemble of" },
  { id: "e11", source: "kmeans", target: "unsupervised", label: "implements" },
  { id: "e12", source: "pca", target: "unsupervised", label: "implements" },

  // Deep Learning
  { id: "e13", source: "neural-net", target: "dl", label: "foundation of" },
  { id: "e14", source: "cnn", target: "neural-net", label: "type of" },
  { id: "e15", source: "rnn", target: "neural-net", label: "type of" },
  { id: "e16", source: "lstm", target: "rnn", label: "variant of" },
  { id: "e17", source: "transformer", target: "neural-net", label: "type of" },
  { id: "e18", source: "gnn", target: "neural-net", label: "type of" },
  { id: "e19", source: "gan", target: "neural-net", label: "type of" },
  { id: "e20", source: "vae", target: "neural-net", label: "type of" },

  // CV architectures
  { id: "e21", source: "cnn", target: "cv", label: "used in" },
  { id: "e22", source: "resnet", target: "cnn", label: "implementation of" },
  { id: "e23", source: "yolo", target: "detection", label: "implements" },
  { id: "e24", source: "detection", target: "cv", label: "task in" },
  { id: "e25", source: "segmentation", target: "cv", label: "task in" },

  // NLP connections
  { id: "e26", source: "transformer", target: "nlp", label: "used in" },
  { id: "e27", source: "bert", target: "transformer", label: "based on" },
  { id: "e28", source: "gpt", target: "transformer", label: "based on" },
  { id: "e29", source: "attention", target: "transformer", label: "key component of" },
  { id: "e30", source: "word2vec", target: "nlp", label: "technique in" },
  { id: "e31", source: "tokenization", target: "nlp", label: "preprocessing for" },

  // RL connections
  { id: "e32", source: "q-learning", target: "rl", label: "algorithm in" },
  { id: "e33", source: "dqn", target: "q-learning", label: "deep version of" },
  { id: "e34", source: "policy-gradient", target: "rl", label: "algorithm in" },
  { id: "e35", source: "actor-critic", target: "policy-gradient", label: "combines" },

  // Applications
  { id: "e36", source: "chatbot", target: "nlp", label: "application of" },
  { id: "e37", source: "chatbot", target: "gpt", label: "powered by" },
  { id: "e38", source: "recommendation", target: "ml", label: "application of" },
  { id: "e39", source: "autonomous", target: "rl", label: "application of" },
  { id: "e40", source: "autonomous", target: "cv", label: "application of" },
  { id: "e41", source: "medical-imaging", target: "cv", label: "application of" },
  { id: "e42", source: "medical-imaging", target: "cnn", label: "uses" },
  { id: "e43", source: "fraud-detection", target: "ml", label: "application of" },

  // Data & Training
  { id: "e44", source: "dataset", target: "supervised", label: "required for" },
  { id: "e45", source: "feature", target: "ml", label: "preprocessing for" },
  { id: "e46", source: "augmentation", target: "dataset", label: "expands" },
  { id: "e47", source: "normalization", target: "feature", label: "step in" },

  // Optimization
  { id: "e48", source: "backprop", target: "neural-net", label: "trains" },
  { id: "e49", source: "gradient-descent", target: "backprop", label: "uses" },
  { id: "e50", source: "adam", target: "gradient-descent", label: "variant of" },
  { id: "e51", source: "regularization", target: "neural-net", label: "improves" },
  { id: "e52", source: "dropout", target: "regularization", label: "technique for" },

  // Cross-connections
  { id: "e53", source: "attention", target: "cv", label: "also used in" },
  { id: "e54", source: "gan", target: "augmentation", label: "generates" },
  { id: "e55", source: "transformer", target: "cv", label: "adapted to" },
  { id: "e56", source: "gnn", target: "recommendation", label: "powers" },
];

export default function VisualizationDemoPage() {
  const [showLegend, setShowLegend] = useState(true);
  const [showStats, setShowStats] = useState(true);

  const nodeTypes = Array.from(new Set(mockNodes.map(n => n.type)));
  const typeColors: Record<string, string> = {
    "Concept": "#5C10F4",
    "Algorithm": "#A550FF",
    "Architecture": "#0DFF00",
    "Technology": "#00D9FF",
    "Application": "#FF6B35",
    "Data": "#F7B801",
    "Optimization": "#FF1E56",
  };

  return (
    <div className="flex min-h-screen bg-black text-white">
      {/* Main Visualization */}
      <div className="flex-1 relative">
        <GraphVisualization
          nodes={mockNodes}
          edges={mockEdges}
          config={{
            fontSize: 12,
          }}
        />

        {/* Header Overlay */}
        <div className="absolute top-0 left-0 right-0 p-6 bg-gradient-to-b from-black/80 to-transparent pointer-events-none">
          <h1 className="text-3xl font-bold mb-2">AI/ML Knowledge Graph</h1>
          <p className="text-gray-400">
            Interactive visualization of artificial intelligence concepts and relationships
          </p>
        </div>

        {/* Controls */}
        <div className="absolute top-6 right-6 flex gap-2">
          <button
            onClick={() => setShowLegend(!showLegend)}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-sm transition-colors"
          >
            {showLegend ? "Hide" : "Show"} Legend
          </button>
          <button
            onClick={() => setShowStats(!showStats)}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-sm transition-colors"
          >
            {showStats ? "Hide" : "Show"} Stats
          </button>
        </div>

        {/* Instructions */}
        <div className="absolute bottom-6 left-6 bg-black/70 backdrop-blur-md p-4 rounded-lg max-w-md pointer-events-none">
          <h3 className="font-semibold mb-2">💡 How to Explore</h3>
          <ul className="text-sm text-gray-300 space-y-1">
            <li>• <strong>Hover</strong> over nodes to see labels</li>
            <li>• <strong>Zoom in</strong> (scroll) to see connections</li>
            <li>• <strong>Click & drag</strong> to pan around</li>
            <li>• <strong>Click</strong> on nodes to select them</li>
          </ul>
        </div>
      </div>

      {/* Legend Panel */}
      {showLegend && (
        <div className="w-80 bg-gray-900/95 backdrop-blur-md p-6 border-l border-gray-800 overflow-y-auto">
          <h2 className="text-xl font-bold mb-4">Node Types</h2>
          <div className="space-y-3">
            {nodeTypes.map((type) => (
              <div key={type} className="flex items-center gap-3">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: typeColors[type] || "#F4F4F4" }}
                />
                <div>
                  <div className="font-medium">{type}</div>
                  <div className="text-xs text-gray-400">
                    {mockNodes.filter(n => n.type === type).length} nodes
                  </div>
                </div>
              </div>
            ))}
          </div>

          {showStats && (
            <div className="mt-8 pt-6 border-t border-gray-800">
              <h2 className="text-xl font-bold mb-4">Statistics</h2>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Nodes:</span>
                  <span className="font-semibold">{mockNodes.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Edges:</span>
                  <span className="font-semibold">{mockEdges.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Avg. Connections:</span>
                  <span className="font-semibold">
                    {(mockEdges.length * 2 / mockNodes.length).toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Node Types:</span>
                  <span className="font-semibold">{nodeTypes.length}</span>
                </div>
              </div>
            </div>
          )}

          <div className="mt-8 pt-6 border-t border-gray-800">
            <h3 className="font-semibold mb-2">About This Graph</h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              This knowledge graph represents the interconnected landscape of
              artificial intelligence, machine learning, and deep learning. Nodes
              represent concepts, algorithms, architectures, and applications,
              while edges show their relationships.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
