import assert from "node:assert/strict";

const [origin, base = ""] = process.argv.slice(2);
const page = await fetch(`${origin}${base}/local-login`);
assert.equal(page.status, 200, "login page");
const html = await page.text();
const assets = [...html.matchAll(/(?:src|href)="([^"?#]+(?:\.js|\.css))[^"\s]*"/g)].map(m => m[1]);
assert.ok(assets.length, "page must reference compiled JS/CSS");
for (const path of assets) {
  assert.ok(path.startsWith(`${base}/_next/`), `asset escaped mount: ${path}`);
  const response = await fetch(new URL(path, origin));
  assert.equal(response.status, 200, path);
  if (path.endsWith(".css")) {
    const css = await response.text();
    for (const [, font] of css.matchAll(/url\(["']?([^\s)"']+\.woff2?)["']?\)/g)) {
      const url = new URL(font, response.url);
      assert.ok(url.pathname.startsWith(`${base}/`), `font escaped mount: ${url}`);
      assert.equal((await fetch(url)).status, 200, `font: ${url}`);
    }
  }
}
for (const path of ["/cognee-logo-black.svg", "/images/icons/avatar.svg", "/visuals/logos/openclaw.svg", "/videos/cognee-mascot-wave.mp4"]) {
  assert.equal((await fetch(`${origin}${base}${path}`)).status, 200, path);
}
if (base) {
  for (const path of ["/local-login", "/api/runtime-config", "/cognee-logo-black.svg"]) {
    assert.equal((await fetch(`${origin}${path}`, { redirect: "manual" })).status, 404, `unprefixed ${path}`);
  }
}
console.log(`UI HTTP/asset checks passed for ${base || "/"}`);
