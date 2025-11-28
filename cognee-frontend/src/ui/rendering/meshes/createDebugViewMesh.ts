import { Mesh, PlaneGeometry, WebGLRenderTarget } from "three";
import { createDebugViewMaterial } from "../materials/createDebugViewMaterial";

export default function createDebugViewMesh(renderTarget: WebGLRenderTarget) {
  const debugQuad = new Mesh(
    new PlaneGeometry(2, 2),
    createDebugViewMaterial(renderTarget.texture)
  );

  debugQuad.frustumCulled = false;

  return debugQuad;
}
