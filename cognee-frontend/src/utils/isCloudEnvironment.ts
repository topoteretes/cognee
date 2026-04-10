
// Default to cloud mode — local mode is opt-in via NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT=false
export default function isCloudEnvironment() {
  return process.env.NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT?.toLowerCase() !== "false";
}
