"use client";

import { useEffect, useState } from "react";
import { stashOAuthOutcome } from "@/modules/integrations/oauthOutcome";
import type { GoogleProvider } from "@/modules/integrations/googleApi";
import GoogleIntegrationCard from "./GoogleIntegrationCard";

const PROVIDERS: GoogleProvider[] = ["google_drive", "gmail"];

export default function GoogleIntegrationsSection() {
  const [connecting, setConnecting] = useState<GoogleProvider | null>(null);

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
      {PROVIDERS.map((provider) => <GoogleIntegrationCard key={provider} provider={provider} connecting={connecting} setConnecting={setConnecting} />)}
    </>
  );
}
