import { readdir, readFile } from "fs/promises";
import { join } from "path";
import { NextResponse, type NextRequest } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");

async function findManifest(runId: string): Promise<string | null> {
  // Scan data/*/artifacts/<runId>/manifest.json
  let topDirs;
  try {
    topDirs = await readdir(DATA_DIR, { withFileTypes: true });
  } catch {
    return null;
  }
  for (const d of topDirs) {
    if (!d.isDirectory()) continue;
    const manifestPath = join(DATA_DIR, d.name, "artifacts", runId, "manifest.json");
    try {
      await readFile(manifestPath);
      return manifestPath;
    } catch {
      // not here
    }
  }
  return null;
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ runId: string }> }
) {
  const { runId } = await params;
  const manifestPath = await findManifest(runId);
  if (!manifestPath) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  const raw = await readFile(manifestPath, "utf-8");
  const data = JSON.parse(raw);
  return NextResponse.json(data);
}
