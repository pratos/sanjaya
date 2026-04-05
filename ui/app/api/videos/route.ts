import { access, readdir } from "fs/promises";
import { join, relative, basename, dirname } from "path";
import { NextResponse } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");

export interface VideoEntry {
  path: string;
  hasTranscript: boolean;
}

async function fileExists(p: string): Promise<boolean> {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

async function findVideos(dir: string): Promise<VideoEntry[]> {
  const results: VideoEntry[] = [];
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return results;
  }
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "artifacts") continue;
      results.push(...(await findVideos(full)));
    } else if (entry.name.endsWith(".mp4")) {
      const stem = basename(entry.name, ".mp4");
      const metaDir = join(dirname(dirname(full)), "meta");
      const transcriptPath = join(metaDir, `${stem}_en.json`);
      results.push({
        path: relative(DATA_DIR, full),
        hasTranscript: await fileExists(transcriptPath),
      });
    }
  }
  return results;
}

export async function GET() {
  const videos = (await findVideos(DATA_DIR)).sort((a, b) =>
    a.path.localeCompare(b.path)
  );
  return NextResponse.json(videos);
}
