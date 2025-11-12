/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable React Compiler optimizations
  experimental: {
    reactCompiler: true,
  },

  // Optimize images
  images: {
    formats: ['image/webp', 'image/avif'],
    minimumCacheTTL: 60,
  },

  // Enable compression
  compress: true,

  // Production optimizations
  poweredByHeader: false,
  generateEtags: true,

  // Bundle analyzer (enable with ANALYZE=true npm run build)
  ...(process.env.ANALYZE === 'true' && {
    webpack: (config) => {
      const { BundleAnalyzerPlugin } = require('@next/bundle-analyzer')();
      config.plugins.push(new BundleAnalyzerPlugin());
      return config;
    },
  }),
};

export default nextConfig;
