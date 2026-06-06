import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const { level, tag, message, url, method } = await request.json();
    const prefix = tag ? `[${tag}]` : "[CLIENT]";
    const detail = method && url ? `${method} ${url} — ` : "";

    if (level === "error") {
      console.error(`${prefix} ✗ ${detail}${message}`);
    } else if (level === "warn") {
      console.warn(`${prefix} ⚠ ${detail}${message}`);
    } else {
      console.log(`${prefix} ${detail}${message}`);
    }
  } catch {
    // Don't fail on malformed payloads
  }

  return NextResponse.json({ ok: true });
}
