import { readdirSync, readFileSync, statSync } from "fs";
import { join } from "path";

// Guards the absence of two third-party marketing touchpoints inside the
// logged-in app (compliance items A3.3 and A3.4). Absence is exactly what a
// feature test cannot cover: nothing renders, nothing fires, and the page
// that would carry them looks identical either way. A scan of the source is
// what notices one coming back.

const SRC = join(process.cwd(), "src");

// The e2e suite blocks these hosts at the network layer, so its own pattern
// list names them on purpose and must not read as a reintroduction.
const EXEMPT = [join("src", "tests", "e2e"), join("src", "__tests__")];

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.(ts|tsx|js|jsx)$/.test(entry) ? [path] : [];
  });
}

const FILES = sourceFiles(SRC).filter(
  (path) => !EXEMPT.some((exempt) => path.includes(exempt)),
);

function filesMatching(pattern: RegExp): string[] {
  return FILES.filter((path) => pattern.test(readFileSync(path, "utf8"))).map((path) =>
    path.slice(process.cwd().length + 1),
  );
}

describe("third-party marketing touchpoints", () => {
  it("has no Google Ads conversion tag left in the app", () => {
    // Advertising measurement inside a B2B platform has no legal basis
    // without a second consent layer, and the notice does not mention it.
    // The call that used to be here never reached Google (nothing loads
    // gtag.js and CSP would refuse it), which is precisely why only a scan
    // like this one would notice it returning.
    expect(filesMatching(/\bgtag\b|\bdataLayer\b|AW-\d{6,}/)).toEqual([]);
  });

  it("has no Calendly link left in the app", () => {
    // The notice lists HubSpot Meetings as the only booking processor, so a
    // Calendly link would send the user's IP to a US processor it never
    // names.
    expect(filesMatching(/calendly\.com/i)).toEqual([]);
  });

  it("books calls through the HubSpot calendar the notice covers", () => {
    const navbarFooter = readFileSync(
      join(SRC, "ui", "layout", "Navbar", "NavbarFooter.tsx"),
      "utf8",
    );

    expect(navbarFooter).toContain("https://meetings-eu1.hubspot.com/");
  });
});
