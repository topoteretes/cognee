import { Mesh, PlaneGeometry, WebGLRenderTarget } from "three";
import { createMetaballMaterial } from "../materials/createMetaballMaterial";

export default function createMetaballMesh(
  fieldRenderTarget: WebGLRenderTarget
) {
  const quadGeo = new PlaneGeometry(2, 2);

  const metaballMat = createMetaballMaterial(fieldRenderTarget.texture);

  const quad = new Mesh(quadGeo, metaballMat);
  quad.frustumCulled = false;

  return quad;
}
