const VIDEO_ID_PATTERN = /^[A-Za-z0-9_-]{11}$/;

export function extractYouTubeVideoId(value) {
  const input = String(value || "").trim();
  if (!input) return "";
  if (VIDEO_ID_PATTERN.test(input)) return input;

  let url;
  try {
    const candidate = /^https?:\/\//i.test(input) ? input : `https://${input}`;
    url = new URL(candidate);
  } catch {
    return "";
  }

  const host = url.hostname.toLowerCase().replace(/^www\./, "").replace(/^m\./, "");
  let videoId = "";
  if (host === "youtu.be") {
    videoId = url.pathname.split("/").filter(Boolean)[0] || "";
  } else if (host === "youtube.com" || host === "youtube-nocookie.com") {
    if (url.pathname === "/watch") videoId = url.searchParams.get("v") || "";
    else {
      const parts = url.pathname.split("/").filter(Boolean);
      if (["shorts", "embed", "live"].includes(parts[0])) videoId = parts[1] || "";
    }
  }
  return VIDEO_ID_PATTERN.test(videoId) ? videoId : "";
}

export function findSharedYouTubeUrl(...values) {
  for (const value of values) {
    const text = String(value || "");
    const candidates = [text.trim(), ...text.match(/https?:\/\/[^\s<>"']+/gi) || []];
    for (const candidate of candidates) {
      const cleaned = candidate.replace(/[）、。,.!?]+$/g, "");
      const videoId = extractYouTubeVideoId(cleaned);
      if (videoId) return { videoId, url: `https://www.youtube.com/watch?v=${videoId}` };
    }
  }
  return null;
}
