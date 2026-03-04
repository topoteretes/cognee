import type { Node, Edge } from "@/ui/rendering/graph/types";

// Seeded PRNG (Linear Congruential Generator) for deterministic output
function createRng(seed: number) {
  let state = seed;
  return {
    next(): number {
      state = (state * 1664525 + 1013904223) & 0xffffffff;
      return (state >>> 0) / 0xffffffff;
    },
    nextInt(max: number): number {
      return Math.floor(this.next() * max);
    },
    pick<T>(arr: T[]): T {
      return arr[this.nextInt(arr.length)];
    },
    shuffle<T>(arr: T[]): T[] {
      const result = [...arr];
      for (let i = result.length - 1; i > 0; i--) {
        const j = this.nextInt(i + 1);
        [result[i], result[j]] = [result[j], result[i]];
      }
      return result;
    },
  };
}

interface TypeSpec {
  type: string;
  ratio: number;
  labels: string[];
}

const TYPE_SPECS: TypeSpec[] = [
  {
    type: "Domain",
    ratio: 0.02,
    labels: [
      "Computer Science", "Physics", "Mathematics", "Biology", "Chemistry",
      "Engineering", "Medicine", "Economics", "Psychology", "Philosophy",
      "Linguistics", "Sociology", "Astronomy", "Geology", "Ecology",
    ],
  },
  {
    type: "Field",
    ratio: 0.05,
    labels: [
      "Machine Learning", "Quantum Mechanics", "Algebra", "Genetics", "Organic Chemistry",
      "Robotics", "Neuroscience", "Microeconomics", "Cognitive Science", "Ethics",
      "Computational Linguistics", "Urban Studies", "Astrophysics", "Geophysics", "Marine Biology",
      "Data Science", "Thermodynamics", "Topology", "Immunology", "Biochemistry",
      "Control Theory", "Cardiology", "Game Theory", "Developmental Psychology", "Logic",
    ],
  },
  {
    type: "Subfield",
    ratio: 0.10,
    labels: [
      "Deep Learning", "Particle Physics", "Graph Theory", "Genomics", "Catalysis",
      "Computer Vision", "Synaptic Plasticity", "Behavioral Economics", "Memory Studies", "Epistemology",
      "NLP", "Social Networks", "Cosmology", "Plate Tectonics", "Conservation",
      "Reinforcement Learning", "Optics", "Number Theory", "Proteomics", "Polymer Science",
      "Swarm Intelligence", "Oncology", "Auction Theory", "Clinical Psychology", "Modal Logic",
      "Speech Recognition", "Demography", "Exoplanets", "Volcanology", "Population Ecology",
      "Generative Models", "Fluid Dynamics", "Combinatorics", "CRISPR", "Electrochemistry",
    ],
  },
  {
    type: "Concept",
    ratio: 0.30,
    labels: [
      "Neural Network", "Entropy", "Eigenvalue", "DNA Replication", "Covalent Bond",
      "Backpropagation", "Wave Function", "Group Theory", "Transcription", "Acid-Base",
      "Gradient Descent", "Relativity", "Prime Number", "Mutation", "Oxidation",
      "Attention Mechanism", "Superposition", "Manifold", "Gene Expression", "Isomer",
      "Loss Function", "Entanglement", "Hilbert Space", "Mitosis", "Redox Reaction",
      "Transformer", "Uncertainty Principle", "Fourier Transform", "Meiosis", "pH Scale",
      "Embedding", "Photon", "Probability", "Enzyme", "Molecule",
      "Convolution", "Boson", "Integral", "Protein Folding", "Ionic Bond",
      "Regularization", "Fermion", "Derivative", "Antibody", "Catalyst",
      "Activation Function", "Quark", "Limit", "Receptor", "Solvent",
      "Dropout", "Neutrino", "Vector Space", "Membrane", "Crystal",
      "Batch Normalization", "Muon", "Tensor", "Chromosome", "Alloy",
    ],
  },
  {
    type: "Method",
    ratio: 0.20,
    labels: [
      "Stochastic Gradient Descent", "Monte Carlo", "Regression Analysis", "PCR",
      "Spectroscopy", "Random Forest", "Finite Element", "Bayesian Inference",
      "Western Blot", "Chromatography", "Support Vector Machine", "Perturbation Theory",
      "Principal Component Analysis", "Gel Electrophoresis", "Mass Spectrometry",
      "K-Means Clustering", "Variational Method", "Hypothesis Testing", "Flow Cytometry",
      "X-Ray Diffraction", "Cross Validation", "Renormalization", "ANOVA",
      "Immunoassay", "Titration", "Grid Search", "Lattice QCD", "T-Test",
      "ELISA", "NMR Spectroscopy", "Beam Search", "Path Integral", "Chi-Square",
      "Microscopy", "Calorimetry", "Ensemble Methods", "Numerical Integration",
      "Confidence Interval", "Cell Culture", "Distillation",
    ],
  },
  {
    type: "Theory",
    ratio: 0.10,
    labels: [
      "Information Theory", "General Relativity", "Category Theory", "Evolution Theory",
      "Molecular Orbital Theory", "Complexity Theory", "String Theory", "Set Theory",
      "Germ Theory", "Valence Bond Theory", "Computability Theory", "Quantum Field Theory",
      "Chaos Theory", "Cell Theory", "Kinetic Theory", "Game Theory", "Loop Quantum Gravity",
      "Probability Theory", "Central Dogma", "VSEPR Theory", "Automata Theory",
      "Standard Model", "Measure Theory", "Endosymbiotic Theory", "Crystal Field Theory",
    ],
  },
  {
    type: "Technology",
    ratio: 0.13,
    labels: [
      "TensorFlow", "Particle Accelerator", "MATLAB", "CRISPR-Cas9", "Mass Spectrometer",
      "PyTorch", "Telescope", "Mathematica", "DNA Sequencer", "Electron Microscope",
      "Kubernetes", "Interferometer", "R Language", "Bioreactor", "Spectrometer",
      "Docker", "Laser", "Julia", "Centrifuge", "Calorimeter",
      "Apache Spark", "Photon Detector", "SageMath", "PCR Machine", "Chromatograph",
      "CUDA", "Collider", "Jupyter", "Microarray", "Polarimeter",
      "MLflow", "Synchrotron", "LaTeX", "Flow Cytometer", "Viscometer",
      "Hugging Face", "Gravitational Wave Detector", "Wolfram Alpha", "Gene Gun", "Refractometer",
    ],
  },
  {
    type: "Application",
    ratio: 0.10,
    labels: [
      "Self-Driving Cars", "Solar Panels", "Cryptography", "Gene Therapy", "Drug Design",
      "Chatbot", "MRI Scanner", "Weather Forecasting", "Vaccine Development", "Battery Tech",
      "Recommendation System", "GPS Navigation", "Climate Modeling", "Cancer Treatment", "Fertilizer",
      "Fraud Detection", "Satellite Imaging", "Risk Analysis", "Prosthetics", "Water Purification",
      "Speech Assistant", "Fiber Optics", "Earthquake Prediction", "Diagnostics", "Plastic Recycling",
      "Image Recognition", "Nuclear Power", "Portfolio Optimization", "Tissue Engineering", "Desalination",
    ],
  },
];

interface GenerateResult {
  nodes: Node[];
  edges: Edge[];
  clusters: Map<string, string[]>;
}

export function generateOntologyGraph(
  mode: "simple" | "medium" | "complex"
): GenerateResult {
  const targetCount = mode === "simple" ? 500 : mode === "medium" ? 1000 : 1500;
  const seed = mode === "simple" ? 42 : mode === "medium" ? 137 : 271;
  const rng = createRng(seed);

  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const clusters = new Map<string, string[]>();

  // Allocate nodes per type
  const typeNodes = new Map<string, Node[]>();

  for (const spec of TYPE_SPECS) {
    const count = Math.max(1, Math.round(targetCount * spec.ratio));
    const typeNodeList: Node[] = [];

    for (let i = 0; i < count; i++) {
      const label = spec.labels[i % spec.labels.length];
      const suffix = i >= spec.labels.length ? ` ${Math.floor(i / spec.labels.length) + 1}` : "";
      const node: Node = {
        id: `${spec.type.toLowerCase()}_${i}`,
        label: `${label}${suffix}`,
        type: spec.type,
      };
      nodes.push(node);
      typeNodeList.push(node);
    }

    typeNodes.set(spec.type, typeNodeList);
    clusters.set(spec.type, typeNodeList.map((n) => n.id));
  }

  let edgeIndex = 0;

  function addEdge(source: string, target: string, label: string) {
    edges.push({
      id: `edge_${edgeIndex++}`,
      label,
      source,
      target,
    });
  }

  // Hierarchy edges: Domain -> Field -> Subfield -> Concept
  const hierarchy: [string, string, string][] = [
    ["Domain", "Field", "contains_field"],
    ["Field", "Subfield", "contains_subfield"],
    ["Subfield", "Concept", "defines_concept"],
  ];

  for (const [parentType, childType, edgeLabel] of hierarchy) {
    const parents = typeNodes.get(parentType) ?? [];
    const children = typeNodes.get(childType) ?? [];

    for (const child of children) {
      const parent = rng.pick(parents);
      addEdge(parent.id, child.id, edgeLabel);
    }
  }

  // Cross-type edges
  const crossLinks: [string, string, string][] = [
    ["Concept", "Method", "used_in"],
    ["Method", "Technology", "implemented_by"],
    ["Technology", "Application", "enables"],
    ["Theory", "Concept", "explains"],
    ["Theory", "Method", "motivates"],
    ["Application", "Domain", "applied_to"],
  ];

  for (const [fromType, toType, edgeLabel] of crossLinks) {
    const fromNodes = typeNodes.get(fromType) ?? [];
    const toNodes = typeNodes.get(toType) ?? [];
    const count = Math.max(1, Math.round(Math.min(fromNodes.length, toNodes.length) * 0.3));

    for (let i = 0; i < count; i++) {
      const from = rng.pick(fromNodes);
      const to = rng.pick(toNodes);
      if (from.id !== to.id) {
        addEdge(from.id, to.id, edgeLabel);
      }
    }
  }

  // Intra-type edges (same type lateral connections)
  for (const [type, typeNodeList] of typeNodes) {
    const count = Math.max(1, Math.round(typeNodeList.length * 0.15));

    for (let i = 0; i < count; i++) {
      const a = rng.pick(typeNodeList);
      const b = rng.pick(typeNodeList);
      if (a.id !== b.id) {
        addEdge(a.id, b.id, "related_to");
      }
    }
  }

  return { nodes, edges, clusters };
}
