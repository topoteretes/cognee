"use client";

import { useEffect, useState } from "react";
import { stashOAuthOutcome } from "@/modules/integrations/oauthOutcome";
import type { SdkProvider } from "@/modules/integrations/sdkApi";
import SdkIntegrationCard from "./SdkIntegrationCard";

const PROVIDERS: SdkProvider[] = ["google_drive", "gmail", "github"];

export default function SdkIntegrationsSection() {
  const [connecting, setConnecting] = useState<SdkProvider | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const provider = PROVIDERS.find((key) => params.has(key));
    if (provider && window.opener && window.opener !== window) {
      stashOAuthOutcome(provider, params.get(provider));
      window.close();
    }
  }, []);

  return (
    <>
      {PROVIDERS.map((provider) => <SdkIntegrationCard key={provider} provider={provider} connecting={connecting} setConnecting={setConnecting} />)}
    </>
  );
}
