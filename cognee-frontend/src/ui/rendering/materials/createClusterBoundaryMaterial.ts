import * as three from "three";

export default function createClusterBoundaryMaterial(
  clusterColor: three.Color
): three.ShaderMaterial {
  const material = new three.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: three.DoubleSide,
    uniforms: {
      clusterColor: { value: clusterColor },
    },
    vertexShader: `
      varying vec2 vUv;

      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      uniform vec3 clusterColor;
      varying vec2 vUv;

      void main() {
        // Apple embedding atlas style: soft circular regions
        vec2 center = vec2(0.5, 0.5);
        float dist = length(vUv - center);

        // Soft radial gradient background
        float alpha = smoothstep(0.5, 0.25, dist) * 0.12; // More visible background

        // Prominent boundary ring (Apple style)
        float ring = smoothstep(0.49, 0.47, dist) - smoothstep(0.51, 0.49, dist);
        alpha += ring * 0.25; // More prominent border

        // Lighter, more vibrant colors for Apple aesthetic
        vec3 bgColor = clusterColor * 1.1;

        gl_FragColor = vec4(bgColor, alpha);
      }
    `,
  });

  return material;
}
