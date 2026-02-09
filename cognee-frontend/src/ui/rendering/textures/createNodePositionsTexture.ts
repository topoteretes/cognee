import * as three from "three";
import { Node } from "../graph/types";

export default function createNodePositionsTexture(
  nodes: Node[],
  nodePositionData: Float32Array
): three.DataTexture {
  const textureSize = Math.ceil(Math.sqrt(nodes.length));

  for (let i = 0; i < nodes.length; i++) {
    nodePositionData[i * 4 + 0] = 0.0;
    nodePositionData[i * 4 + 1] = 0.0;
    nodePositionData[i * 4 + 2] = 0.0;
    nodePositionData[i * 4 + 3] = 1.0;
  }

  const texture = new three.DataTexture(
    nodePositionData,
    textureSize,
    textureSize,
    three.RGBAFormat,
    three.FloatType
  );
  texture.needsUpdate = true;
  texture.minFilter = three.NearestFilter;
  texture.magFilter = three.NearestFilter;
  return texture;
}
