import { ShaderMaterial, Texture, Vector2 } from "three";

export function createBlurPassMaterial(
  texture: Texture,
  direction = new Vector2(1.0, 0.0)
) {
  return new ShaderMaterial({
    uniforms: {
      densityTex: { value: texture },
      direction: { value: direction }, // (1,0) = horizontal, (0,1) = vertical
      texSize: { value: new Vector2(512, 512) },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      uniform sampler2D densityTex;
      uniform vec2 direction;
      uniform vec2 texSize;
      varying vec2 vUv;

      void main() {
        vec2 texel = direction / texSize;
        float kernel[5];
        kernel[0] = 0.204164;
        kernel[1] = 0.304005;
        kernel[2] = 0.193783;
        kernel[3] = 0.072184;
        kernel[4] = 0.025864;

        vec4 sum = texture2D(densityTex, vUv) * kernel[0];
        for (int i = 1; i < 5; i++) {
          sum += texture2D(densityTex, vUv + texel * float(i)) * kernel[i];
          sum += texture2D(densityTex, vUv - texel * float(i)) * kernel[i];
        }

        gl_FragColor = sum;
      }
    `,
  });
}
