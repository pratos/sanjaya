import { readFile, stat } from "fs/promises";
import { extname, join } from "path";
import { NextResponse, type NextRequest } from "next/server";

const PROJECT_ROOT = join(process.cwd(), "..");
const DATA_DIR = join(PROJECT_ROOT, "data");

const MIME: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".tiff": "image/tiff",
  ".tif": "image/tiff",
  ".bmp": "image/bmp",
  ".heic": "image/heic",
  ".heif": "image/heif",
  ".svg": "image/svg+xml",
  ".avif": "image/avif",
};

function normalizeInputPath(pathParam: string): string {
  const stripped = pathParam.startsWith("file://")
    ? pathParam.slice("file://".length)
    : pathParam;

  try {
    return decodeURIComponent(stripped);
  } catch {
    return stripped;
  }
}

async function existingFile(path: string): Promise<string | null> {
  try {
    const fileStat = await stat(path);
    return fileStat.isFile() ? path : null;
  } catch {
    return null;
  }
}

async function resolveFramePath(framePath: string): Promise<string | null> {
  const normalized = normalizeInputPath(framePath);

  // Absolute paths (used by image benchmark JSONs and uploads)
  if (normalized.startsWith("/")) {
    const absolute = await existingFile(normalized);
    if (absolute) return absolute;
  }

  // Relative to data/
  const cleaned = normalized.replace(/^data\//, "").replace(/^\/+/, "");

  const candidates = [
    join(DATA_DIR, cleaned),
    // Relative to project root (e.g. sanjaya_artifacts/...)
    join(PROJECT_ROOT, cleaned),
  ];

  for (const candidate of candidates) {
    const match = await existingFile(candidate);
    if (match) return match;
  }

  return null;
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
  const contentType = MIME[ext];
  if (!contentType) {
    return new NextResponse("Unsupported image type", { status: 415 });
  }

  const buf = await readFile(fullPath);

  return new NextResponse(buf, {
    headers: {
      "Content-Type": contentType,
      "Content-Length": String(buf.length),
      "Cache-Control": "public, max-age=86400, immutable",
    },
  });
}
