const assert = require('node:assert/strict');
const {displayPositions, sourceClock, eventInWindow} = require('../examples/soccer/components/tactical_player/replay.js');
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
assert.equal(sourceClock(6436.26), '01:47:16');
assert.equal(sourceClock(120), '00:02:00');
const crossing = {start_s:-.5,time_s:.1,origin_in_context:true};
assert.equal(eventInWindow(crossing,0,12),true);
assert.equal(eventInWindow(crossing,.2,12),false);
assert.equal(eventInWindow({...crossing,origin_in_context:false},0,12),false);
assert.equal(eventInWindow({time_s:12},0,12),false);
console.log('Source clock and single-segment ownership of crossing passes: OK');
