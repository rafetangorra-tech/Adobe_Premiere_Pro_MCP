/**
 * Build the composite ExtendScript that applies a cut list to a sequence.
 *
 * Strategy: one round-trip script that razors at each cut's start + end on the
 * selected tracks, then ripple-removes the resulting sub-clip. Cuts are
 * processed in REVERSE chronological order so applying earlier cuts doesn't
 * shift the timestamps of later ones (under ripple delete).
 *
 * The bridge auto-prepends EXTENDSCRIPT_HELPERS, so __findSequence, JSON.stringify,
 * and the seconds/ticks helpers are all available. We rely on the QE DOM for
 * razor (qeTrack.razor(timecode)) and on the public DOM for clip.remove().
 */

import type { CutEntry } from './types.js';

export interface ApplyCutListInput {
  sequenceId?: string | undefined;
  cuts: CutEntry[];
  videoTrackIndices?: number[] | undefined;
  audioTrackIndices?: number[] | undefined;
  rippleDelete?: boolean | undefined;
  dryRun?: boolean | undefined;
}

/**
 * Returns the ExtendScript source that, when executed by the bridge, applies
 * the supplied cuts. Reading-only (dryRun) variant just reports what would
 * be cut and which clips would be affected.
 */
export function buildApplyCutListScript(input: ApplyCutListInput): string {
  // We only emit `remove` actions. `keep` entries are passed through for
  // completeness but become inverted ranges before reaching this layer.
  const removeCuts = input.cuts.filter((c) => c.action === 'remove');

  // Sort descending by startSec so later cuts apply first under ripple delete.
  const sortedDesc = [...removeCuts].sort((a, b) => b.startSec - a.startSec);

  const cutsLiteral = JSON.stringify(
    sortedDesc.map((c) => ({
      startSec: c.startSec,
      endSec: c.endSec,
      reason: c.reason || '',
      source: c.source || 'manual',
    }))
  );

  const sequenceLookup = input.sequenceId
    ? `__findSequence(${JSON.stringify(input.sequenceId)})`
    : 'app.project.activeSequence';

  const videoIndicesLiteral = JSON.stringify(input.videoTrackIndices ?? []);
  const audioIndicesLiteral = JSON.stringify(input.audioTrackIndices ?? []);
  const rippleLiteral = input.rippleDelete === false ? 'false' : 'true';
  const dryRunLiteral = input.dryRun ? 'true' : 'false';

  // The script. ExtendScript = ES3-ish — no const/let, no arrow funcs.
  return `
    try {
      app.enableQE();
      var sequence = ${sequenceLookup};
      if (!sequence) sequence = app.project.activeSequence;
      if (!sequence) return JSON.stringify({ success: false, error: "No sequence available" });

      if (app.project.activeSequence && app.project.activeSequence.sequenceID !== sequence.sequenceID) {
        app.project.openSequence(sequence.sequenceID);
      }
      var seq = app.project.activeSequence;
      if (!seq || seq.sequenceID !== sequence.sequenceID) {
        return JSON.stringify({ success: false, error: "Unable to activate target sequence" });
      }

      var qeSeq = qe.project.getActiveSequence();
      if (!qeSeq) return JSON.stringify({ success: false, error: "QE active sequence unavailable" });

      var fps = seq.timebase ? (254016000000 / parseInt(seq.timebase, 10)) : 30;

      function pad(n) { return n < 10 ? "0" + n : "" + n; }
      function tcOf(seconds) {
        var totalFrames = Math.round(seconds * fps);
        var hours = Math.floor(totalFrames / (fps * 3600));
        var mins = Math.floor((totalFrames % (fps * 3600)) / (fps * 60));
        var secs = Math.floor((totalFrames % (fps * 60)) / fps);
        var frames = Math.round(totalFrames % fps);
        return pad(hours) + ":" + pad(mins) + ":" + pad(secs) + ":" + pad(frames);
      }

      function buildIndices(count, requested) {
        if (!requested || requested.length === 0) {
          var all = [];
          for (var i = 0; i < count; i++) all.push(i);
          return all;
        }
        var ok = [];
        for (var j = 0; j < requested.length; j++) {
          if (requested[j] >= 0 && requested[j] < count) ok.push(requested[j]);
        }
        return ok;
      }

      var requestedVideo = ${videoIndicesLiteral};
      var requestedAudio = ${audioIndicesLiteral};
      var videoIndices = buildIndices(seq.videoTracks.numTracks, requestedVideo);
      var audioIndices = buildIndices(seq.audioTracks.numTracks, requestedAudio);

      var ripple = ${rippleLiteral};
      var dryRun = ${dryRunLiteral};
      var cuts = ${cutsLiteral};

      var EPS = 0.001; // 1ms tolerance for floating-point start/end matching

      // Find clips on a given track whose start lies within [cutStart-EPS, cutEnd-EPS).
      // After razoring at startSec and endSec, the clips between are exactly the
      // ones whose start is at or after cutStart and strictly before cutEnd.
      function clipsInRange(track, cutStart, cutEnd) {
        var found = [];
        for (var k = 0; k < track.clips.numItems; k++) {
          var cl = track.clips[k];
          var clStart = cl.start.seconds;
          var clEnd = cl.end.seconds;
          if (clStart >= cutStart - EPS && clEnd <= cutEnd + EPS) {
            found.push(cl);
          }
        }
        return found;
      }

      var applied = [];
      var skipped = [];
      var errors = [];

      for (var ci = 0; ci < cuts.length; ci++) {
        var cut = cuts[ci];
        var cutStart = cut.startSec;
        var cutEnd = cut.endSec;

        if (cutEnd <= cutStart + EPS) {
          skipped.push({ cut: cut, reason: "Zero-or-negative-length range" });
          continue;
        }
        if (cutStart < 0) {
          skipped.push({ cut: cut, reason: "Negative start time" });
          continue;
        }

        var startTC = tcOf(cutStart);
        var endTC = tcOf(cutEnd);

        var razoredVideo = [];
        var razoredAudio = [];
        var perCutRemoved = [];
        var perCutErrors = [];

        // 1. Razor on selected video tracks.
        for (var vi = 0; vi < videoIndices.length; vi++) {
          try {
            var qv = qeSeq.getVideoTrackAt(videoIndices[vi]);
            if (!qv) continue;
            qv.razor(startTC);
            qv.razor(endTC);
            razoredVideo.push(videoIndices[vi]);
          } catch (eRV) {
            perCutErrors.push({ stage: "razor_video", trackIndex: videoIndices[vi], error: String(eRV) });
          }
        }

        // 2. Razor on selected audio tracks.
        for (var ai = 0; ai < audioIndices.length; ai++) {
          try {
            var qa = qeSeq.getAudioTrackAt(audioIndices[ai]);
            if (!qa) continue;
            qa.razor(startTC);
            qa.razor(endTC);
            razoredAudio.push(audioIndices[ai]);
          } catch (eRA) {
            perCutErrors.push({ stage: "razor_audio", trackIndex: audioIndices[ai], error: String(eRA) });
          }
        }

        // 3. Collect (and optionally remove) clips inside the razored range.
        for (var v2 = 0; v2 < videoIndices.length; v2++) {
          var vTrack = seq.videoTracks[videoIndices[v2]];
          if (!vTrack) continue;
          var inV = clipsInRange(vTrack, cutStart, cutEnd);
          for (var iv = 0; iv < inV.length; iv++) {
            perCutRemoved.push({
              trackType: "video",
              trackIndex: videoIndices[v2],
              clipName: inV[iv].name,
              start: inV[iv].start.seconds,
              end: inV[iv].end.seconds
            });
          }
        }
        for (var a2 = 0; a2 < audioIndices.length; a2++) {
          var aTrack = seq.audioTracks[audioIndices[a2]];
          if (!aTrack) continue;
          var inA = clipsInRange(aTrack, cutStart, cutEnd);
          for (var ia = 0; ia < inA.length; ia++) {
            perCutRemoved.push({
              trackType: "audio",
              trackIndex: audioIndices[a2],
              clipName: inA[ia].name,
              start: inA[ia].start.seconds,
              end: inA[ia].end.seconds
            });
          }
        }

        // 4. Actually remove (skip if dryRun). Re-fetch the lists because we
        //    don't trust ExtendScript references to survive iteration mutations.
        if (!dryRun) {
          for (var v3 = 0; v3 < videoIndices.length; v3++) {
            var vt = seq.videoTracks[videoIndices[v3]];
            if (!vt) continue;
            // Iterate in reverse to keep clip indices stable as we remove.
            for (var n = vt.clips.numItems - 1; n >= 0; n--) {
              var cv = vt.clips[n];
              if (!cv) continue;
              if (cv.start.seconds >= cutStart - EPS && cv.end.seconds <= cutEnd + EPS) {
                try {
                  cv.remove(ripple, true);
                } catch (eRm) {
                  perCutErrors.push({ stage: "remove_video", trackIndex: videoIndices[v3], error: String(eRm) });
                }
              }
            }
          }
          for (var a3 = 0; a3 < audioIndices.length; a3++) {
            var at = seq.audioTracks[audioIndices[a3]];
            if (!at) continue;
            for (var m = at.clips.numItems - 1; m >= 0; m--) {
              var ca = at.clips[m];
              if (!ca) continue;
              if (ca.start.seconds >= cutStart - EPS && ca.end.seconds <= cutEnd + EPS) {
                try {
                  ca.remove(ripple, true);
                } catch (eRmA) {
                  perCutErrors.push({ stage: "remove_audio", trackIndex: audioIndices[a3], error: String(eRmA) });
                }
              }
            }
          }
        }

        applied.push({
          cut: cut,
          startTC: startTC,
          endTC: endTC,
          razoredVideoTracks: razoredVideo,
          razoredAudioTracks: razoredAudio,
          affectedClips: perCutRemoved,
          errors: perCutErrors
        });
        if (perCutErrors.length > 0) errors.push({ cut: cut, errors: perCutErrors });
      }

      return JSON.stringify({
        success: true,
        sequenceId: seq.sequenceID,
        sequenceName: seq.name,
        dryRun: dryRun,
        rippleDelete: ripple,
        applied: applied,
        skipped: skipped,
        errors: errors,
        summary: {
          requested: cuts.length,
          appliedCount: applied.length,
          skippedCount: skipped.length,
          errorCount: errors.length,
          totalSecondsCut: (function() {
            var s = 0;
            for (var x = 0; x < applied.length; x++) {
              s += applied[x].cut.endSec - applied[x].cut.startSec;
            }
            return s;
          })()
        }
      });
    } catch (e) {
      return JSON.stringify({ success: false, error: "apply_cut_list failed: " + e.toString() });
    }
  `;
}
