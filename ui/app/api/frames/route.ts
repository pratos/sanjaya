import { readFile, stat } from "fs/promises";
import { join, extname } from "path";
import { NextResponse, type NextRequest } from "next/server";

const PROJECT_ROOT = join(process.cwd(), "..");
const DATA_DIR = join(PROJECT_ROOT, "data");

const MIME: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
};

async function resolveFramePath(framePath: string): Promise<string | null> {
  // Try data/ first (strips data/ prefix if present)
  const cleaned = framePath.replace(/^data\//, "");
  const dataPath = join(DATA_DIR, cleaned);
  try {
    await stat(dataPath);
    return dataPath;
  } catch {
    // noop
  }

  // Try project root (for sanjaya_artifacts/ paths)
  const rootPath = join(PROJECT_ROOT, framePath);
  try {
    await stat(rootPath);
    return rootPath;
  } catch {
    return null;
  }
}

export async function GET(req: NextRequest) {
  const framePath = req.nextUrl.searchParams.get("path");
  if (!framePath) {
    return new NextResponse("Missing path", { status: 400 });
  }

  const fullPath = await resolveFramePath(framePath);
  if (!fullPath) {
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
