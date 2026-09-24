const assert = require('node:assert/strict');
const {displayPositions} = require('../examples/soccer/components/tactical_player/replay.js');
const p = {identity_id: 1, track_id: 1, team_id: 0, xy: [10, 20]};
const frames = [
  {time_s: 0, dt: .08, calibrated: true, players: [p]},
  {time_s: .08, dt: .08, calibrated: true, players: [{...p, xy: [10.5, 20]}]}
];
assert.deepEqual(displayPositions(frames, 0, .04, [])[0].xy, [10.25, 20]);
assert.deepEqual(frames[0].players[0].xy, [10, 20]); // stats stay unchanged
assert.deepEqual(displayPositions(frames, 0, .04, [{type: 'camera_cut', time_s: .04}])[0].xy, [10, 20]);
frames[1].players[0].track_id = 2;
assert.deepEqual(displayPositions(frames, 0, .04, [])[0].xy, [10, 20]);
frames[0].calibrated = false;
assert.deepEqual(displayPositions(frames, 0, .04, []), []);
console.log('Replay interpolation, raw data, cuts, tracks and missing calibration: OK');
