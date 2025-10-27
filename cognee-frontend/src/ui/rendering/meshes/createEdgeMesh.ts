import * as three from "three";
import createEdgeMaterial from "../materials/createEdgeMaterial";
import { Edge } from "../graph/types";

export default function createEdgeMesh(
  edges: Edge[],
  nodePositionTexture: three.DataTexture,
  edgeIndices: Float32Array,
  initialCameraDistance: number
): three.LineSegments {
  const numberOfEdges = edges.length;

  const instGeom = new three.InstancedBufferGeometry();
  instGeom.setAttribute(
    "position",
    new three.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0]), 3)
  );
  // instGeom.index = baseGeom.index;
  instGeom.instanceCount = numberOfEdges;

  instGeom.setAttribute(
    "edgeIndices",
    new three.InstancedBufferAttribute(edgeIndices, 2)
  );

  const material = createEdgeMaterial(
    nodePositionTexture,
    initialCameraDistance
  );

  const edgeMesh = new three.LineSegments(instGeom, material);
  edgeMesh.frustumCulled = false;

  return edgeMesh;
}
