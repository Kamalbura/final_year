import path from "node:path";

/** @type {import('next').NextConfig} */
const nextConfig = {
  outputFileTracingRoot: path.resolve(process.cwd(), "..", ".."),
};

export default nextConfig;
