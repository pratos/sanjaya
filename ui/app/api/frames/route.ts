import { readFile, stat } from "fs/promises";
import { join, extname } from "path";
import { NextResponse, type NextRequest } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");

const MIME: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
};

export async function GET(req: NextRequest) {
  const framePath = req.nextUrl.searchParams.get("path");
  if (!framePath) {
    return new NextResponse("Missing path", { status: 400 });
  }

  // Resolve: paths in manifests start with "data/..." so strip that prefix
  const cleaned = framePath.replace(/^data\//, "");
  const fullPath = join(DATA_DIR, cleaned);

  try {
    await stat(fullPath);
  } catch {
    return new NextResponse("Not found", { status: 404 });
  }

  const ext = extname(fullPath).toLowerCase();
  const contentType = MIME[ext] ?? "application/octet-stream";
  const buf = await readFile(fullPath);

  return new NextResponse(buf, {
    headers: {
      "Content-Type": contentType,
      "Content-Length": String(buf.length),
      "Cache-Control": "public, max-age=86400, immutable",
    },
  });
}
