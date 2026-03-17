import { FloatType, LinearFilter, RGBAFormat, WebGLRenderTarget } from "three";

export default function createDensityRenderTarget(size = 512) {
  return new WebGLRenderTarget(size, size, {
    format: RGBAFormat,
    type: FloatType,
    minFilter: LinearFilter,
    magFilter: LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });
}
