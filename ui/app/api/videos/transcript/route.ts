import { readFile } from "fs/promises";
import { join, basename, dirname } from "path";
import { NextResponse, type NextRequest } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");

export async function GET(req: NextRequest) {
  const videoRel = req.nextUrl.searchParams.get("path");
  if (!videoRel) {
    return NextResponse.json({ segments: [] }, { status: 400 });
  }

  const videoPath = join(DATA_DIR, videoRel);
  const stem = basename(videoPath, ".mp4");
  const metaDir = join(dirname(dirname(videoPath)), "meta");
  const transcriptPath = join(metaDir, `${stem}_en.json`);

  try {
    const raw = await readFile(transcriptPath, "utf-8");
    const data = JSON.parse(raw);
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ segments: [] }, { status: 404 });
  }
}
