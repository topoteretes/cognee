
export default function isCloudEnvironment() {
  return process.env.NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT?.toLowerCase() === "true";
}
