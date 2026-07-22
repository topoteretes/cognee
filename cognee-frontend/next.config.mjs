/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  images: {
    remotePatterns: [{
      protocol: "https",
      hostname: "lh3.googleusercontent.com",
    }],
  },
};

export default nextConfig;
