import {
  InstancedBufferAttribute,
  InstancedMesh,
  DataTexture,
  PlaneGeometry,
} from "three";
import { Node } from "../graph/types";
import createDensityAccumulatorMaterial from "../materials/createDensityAccumulatorMaterial";

export default function createDensityAccumulatorMesh(
  nodes: Node[],
  nodeColors: Float32Array,
  nodePositionsTexture: DataTexture,
  initialCameraDistance: number
) {
  const geometry = new PlaneGeometry(2, 2);

  const material = createDensityAccumulatorMaterial(
    nodePositionsTexture,
    initialCameraDistance
  );

  geometry.setAttribute(
    "nodeColor",
    new InstancedBufferAttribute(nodeColors, 3)
  );

  const mesh = new InstancedMesh(geometry, material, nodes.length);

  mesh.frustumCulled = false;

  return mesh;
}
