// Smooth the display between samples. This never changes analytics or events.
function displayPositions(frames, index, time, events) {
  const frame = frames[index], next = frames[index + 1];
  if (!frame?.calibrated) return [];
  const people = frame.display_players || frame.players;
  if (!next?.calibrated || next.time_s - frame.time_s > .32 + 1e-6 ||
      events.some(e => e.type === 'camera_cut' && e.time_s > frame.time_s && e.time_s <= next.time_s)) return people;
  const dt = next.time_s - frame.time_s;
  if (dt <= 0) return people;
  const alpha = Math.max(0, Math.min(1, (time - frame.time_s) / dt));
  return people.map(p => {
    const q = (next.display_players || next.players).find(q => q.identity_id === p.identity_id &&
      q.track_id === p.track_id && q.team_id === p.team_id);
    if (!q || Math.hypot(q.xy[0] - p.xy[0], q.xy[1] - p.xy[1]) > 12 * dt + .1) return p;
    return {...p, xy: p.xy.map((v, i) => v + alpha * (q.xy[i] - v))};
  });
}
if (typeof module !== 'undefined') module.exports = {displayPositions};
