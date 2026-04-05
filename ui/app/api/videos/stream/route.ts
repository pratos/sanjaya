import { stat, open } from "fs/promises";
import { join } from "path";
import { NextResponse, type NextRequest } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");

export async function GET(req: NextRequest) {
  const videoRel = req.nextUrl.searchParams.get("path");
  if (!videoRel) {
    return new NextResponse("Missing path", { status: 400 });
  }

  const videoPath = join(DATA_DIR, videoRel);

  let fileStat;
  try {
    fileStat = await stat(videoPath);
  } catch {
    return new NextResponse("Not found", { status: 404 });
  }

  const range = req.headers.get("range");
  const size = fileStat.size;

  if (range) {
    const [startStr, endStr] = range.replace("bytes=", "").split("-");
    const start = parseInt(startStr, 10);
    const end = endStr ? parseInt(endStr, 10) : Math.min(start + 5 * 1024 * 1024, size - 1);
    const chunkSize = end - start + 1;

    const fh = await open(videoPath, "r");
    const buf = Buffer.alloc(chunkSize);
    await fh.read(buf, 0, chunkSize, start);
    await fh.close();

    return new NextResponse(buf, {
      status: 206,
      headers: {
        "Content-Range": `bytes ${start}-${end}/${size}`,
        "Accept-Ranges": "bytes",
        "Content-Length": String(chunkSize),
        "Content-Type": "video/mp4",
      },
    });
  }

  const fh = await open(videoPath, "r");
  const buf = Buffer.alloc(size);
  await fh.read(buf, 0, size, 0);
  await fh.close();

  return new NextResponse(buf, {
    headers: {
      "Content-Length": String(size),
      "Content-Type": "video/mp4",
      "Accept-Ranges": "bytes",
    },
  });
}
