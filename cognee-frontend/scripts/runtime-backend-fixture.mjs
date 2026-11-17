// Docker smoke-test backend. Port 8000 deliberately rejects requests so a
// hardcoded default cannot pass while runtime-config advertises port 9100.
import { createServer } from "node:http";

for (const port of [8000, 9100]) {
  createServer((request, response) => {
    if (port !== 9100) {
      response.writeHead(500).end("Wrong backend port");
      return;
    }
    const path = new URL(request.url, "http://fixture").pathname;
    if (path === "/ready") {
      response.end("ready");
      return;
    }
    // Any login succeeds, like a backend with the default password, so a proxy
    // that logs in on its own gets a token and the anonymous check sees 200.
    if (path === "/api/v1/auth/login" && request.method === "POST") {
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify({ access_token: "runtime-test-token", token_type: "bearer" }));
      return;
    }
    if (request.headers.authorization !== "Bearer runtime-test-token") {
      response.writeHead(401).end("Missing forwarded token");
      return;
    }
    if (path === "/api/v1/schema/inventory") {
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify({ runtime_backend: true }));
    } else if (["/api/v1/visualize", "/api/v1/schema/provenance"].includes(path)) {
      response.end("<html>runtime_backend</html>");
    } else {
      response.writeHead(404).end();
    }
  }).listen(port, "127.0.0.1");
}
