// Keeps YouTube ads, pauses, and buffering out of content-time analysis.
export function createYouTubeContentSync({
  minStartTime = 0.2,
  requiredProgressTicks = 3,
  advanceEpsilon = 0.01,
} = {}) {
  let lastTime = null;
  let consecutiveProgressTicks = 0;
  let contentStarted = false;

  function observe({ currentTime, playerState, playingState = 1 }) {
    const time = Number(currentTime);
    const isPlaying = playerState === playingState;
    const hasTime = Number.isFinite(time) && time >= 0;
    const advanced = hasTime && lastTime !== null && time > lastTime + advanceEpsilon;

    if (!contentStarted) {
      if (isPlaying && time >= minStartTime && advanced) consecutiveProgressTicks += 1;
      else consecutiveProgressTicks = 0;
      lastTime = hasTime ? time : lastTime;
      if (consecutiveProgressTicks >= requiredProgressTicks) {
        contentStarted = true;
        return { waiting: false, startedNow: true, validForContent: true, pauseReason: "" };
      }
      return { waiting: true, startedNow: false, validForContent: false, pauseReason: "youtube_time_not_advancing" };
    }

    lastTime = hasTime ? time : lastTime;
    if (isPlaying && advanced) return { waiting: false, startedNow: false, validForContent: true, pauseReason: "" };
    return { waiting: false, startedNow: false, validForContent: false, pauseReason: "youtube_time_not_advancing" };
  }

  return {
    observe,
    get contentStarted() { return contentStarted; },
  };
}
