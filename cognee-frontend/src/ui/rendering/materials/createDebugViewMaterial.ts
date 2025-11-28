import { ShaderMaterial, Texture } from "three";

export function createDebugViewMaterial(fieldTexture: Texture) {
  return new ShaderMaterial({
    uniforms: {
      fieldTex: { value: fieldTexture },
    },
    vertexShader: `
      // void main() {
      //   gl_Position = vec4(position, 1.0);
      // }
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
    `,
    fragmentShader: `
      uniform sampler2D fieldTex;
      varying vec2 vUv;
      void main() {
        // gl_FragColor = texture2D(fieldTex, vUv);

        float field = texture2D(fieldTex, vUv).r;
        field = pow(field * 2.0, 0.5);  // optional tone mapping
        gl_FragColor = vec4(vec3(field), 1.0);
      }

      // precision highp float;
      // uniform sampler2D fieldTex;

      // void main() {
      //   vec2 uv = gl_FragCoord.xy / vec2(textureSize(fieldTex, 0));
      //   float field = texture2D(fieldTex, uv).r;
      //   // visualize the field as grayscale
      //   gl_FragColor = vec4(vec3(field), 1.0);
      // }
    `,
  });
}
