import { matchesQuery } from "@/app/(app)/memory-gap-analysis/search";

describe("matchesQuery", () => {
  it("matches plain substrings case-insensitively", () => {
    expect(matchesQuery("Which SOC 2 controls does the platform cover?", "soc")).toBe(true);
    expect(matchesQuery("Which SOC 2 controls does the platform cover?", "PLATFORM")).toBe(true);
  });

  it("unifies singular and plural both ways", () => {
    expect(matchesQuery("Is data encrypted for tenant databases?", "database")).toBe(true);
    expect(matchesQuery("How do I connect a brain?", "brains")).toBe(true);
    expect(matchesQuery("Company policies for retention", "policy")).toBe(true);
  });

  it("unifies common verb endings", () => {
    expect(matchesQuery("How do I connect a Notion workspace?", "connecting")).toBe(true);
    expect(matchesQuery("What shipped in the last three releases?", "ship")).toBe(true);
  });

  it("matches word prefixes while typing", () => {
    expect(matchesQuery("Does cognee support SCIM provisioning?", "provis")).toBe(true);
  });

  it("requires every query word to match", () => {
    expect(matchesQuery("How do I set up SAML single sign-on?", "saml sso")).toBe(false);
    expect(matchesQuery("How do I set up SAML single sign-on?", "saml sign")).toBe(true);
  });

  it("ignores punctuation and empty queries", () => {
    expect(matchesQuery("Can I enforce SSO for all members?", "sso,")).toBe(true);
    expect(matchesQuery("anything", "   ")).toBe(true);
  });

  it("rejects non-matching terms", () => {
    expect(matchesQuery("Is data encrypted at rest?", "billing")).toBe(false);
  });
});
