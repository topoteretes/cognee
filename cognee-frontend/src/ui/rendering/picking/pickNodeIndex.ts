import {
  OrthographicCamera,
  Scene,
  WebGLRenderer,
  WebGLRenderTarget,
} from "three";

const pixelBuffer = new Uint8Array(4);

export default function pickNodeIndex(
  event: MouseEvent,
  renderer: WebGLRenderer,
  pickingScene: Scene,
  camera: OrthographicCamera,
  pickingRenderTarget: WebGLRenderTarget
) {
  const rect = renderer.domElement.getBoundingClientRect();
  // Convert from client coords to pixel coords in render target
  const x =
    ((event.clientX - rect.left) / rect.width) * pickingRenderTarget.width;
  const y =
    pickingRenderTarget.height -
    ((event.clientY - rect.top) / rect.height) * pickingRenderTarget.height;

  renderer.setRenderTarget(pickingRenderTarget);
  renderer.clear();
  renderer.render(pickingScene, camera);
  renderer.readRenderTargetPixels(
    pickingRenderTarget,
    Math.floor(x),
    Math.floor(y),
    1,
    1,
    pixelBuffer
  );
  renderer.setRenderTarget(null);

  const id = pixelBuffer[0] + pixelBuffer[1] * 256 + pixelBuffer[2] * 256 * 256;
  return id || -1;
}
