/**
 * bpm_server.ts — UDP BPM Microservice for Node.js
 *
 * realtime-bpm-analyzer is a browser/WebAudioAPI-only library and cannot
 * run in Node.js. This server implements a self-contained onset-based beat
 * detector that works directly on Float32 PCM chunks over UDP.
 *
 * Protocol:
 *   Python → Node  : raw Float32LE PCM bytes (CHUNK_SIZE * 4 bytes) on port 5005
 *   Node   → Python: 4-byte Float32LE BPM value back to sender (port 5005)
 *   Node   → Python: single 0x01 beat pulse to port 5006 on every detected beat
 *
 * Usage:
 *   npx ts-node bpm_server.ts
 *   — or compile: tsc bpm_server.ts && node bpm_server.js
 */

import * as dgram from 'dgram';
import type { RemoteInfo } from 'dgram';

// ------------------------------------------------------------------ //
//  Configuration
// ------------------------------------------------------------------ //
const PORT            = 5005;
const BEAT_PULSE_PORT = 5006;   // Python listens here for per-beat flashes
const SAMPLE_RATE     = 44100;

// Beat detector tuning
const ENERGY_WINDOW    = 43;   // ~1 second of history at 44100/1024 chunks
const ENERGY_THRESHOLD = 1.5;  // Beat fires when local energy > threshold * mean
const MIN_BPM          = 60;
const MAX_BPM          = 180;
const MIN_BEAT_GAP_MS  = (60 / MAX_BPM) * 1000;

// BPM estimator tuning
const BPM_HISTORY_SIZE = 8;    // Inter-beat intervals to average over
const BPM_BUFFER_SIZE  = 16;   // Smoothing buffer for final reported BPM

// ------------------------------------------------------------------ //
//  Beat Detector
// ------------------------------------------------------------------ //
class BeatDetector {
  private energyHistory: number[] = [];
  private lastBeatTime: number    = 0;
  private ibiBuffer: number[]     = [];   // Inter-beat intervals (ms)
  private bpmBuffer: number[]     = [];   // Smoothed BPM output
  private totalSamples: number    = 0;
  private bassEnergy(samples: Float32Array): number {
    let sum = 0;

    //simple low pass filter
    let filtered = 0;

    for (let i = 0; i < samples.length; i++) 
        {
        filtered = 0.95 * filtered + samples[i] * 0.05;
        sum += filtered * filtered;
    }

    return Math.sqrt(sum / samples.length);
  }

  /**
   * Feed a raw PCM chunk.
   * Returns { bpm, isBeat } where bpm is null until enough data is collected
   * and isBeat is true on the exact chunk where a beat onset was detected.
   */
  process(samples: Float32Array): { bpm: number | null; isBeat: boolean } {
    const energy = this.bassEnergy(samples);

    this.totalSamples += samples.length;

    this.energyHistory.push(energy);
    if (this.energyHistory.length > ENERGY_WINDOW) {
      this.energyHistory.shift();
    }

    if (this.energyHistory.length < ENERGY_WINDOW) {
      return { bpm: null, isBeat: false };
    }

    const meanEnergy = this.energyHistory.reduce((a,b)=>a+b,0) / ENERGY_WINDOW;

    const variance = this.energyHistory.reduce((sum,e)=>sum + Math.pow(e - meanEnergy,2),0) / ENERGY_WINDOW;
    const stdDev = Math.sqrt(variance);
    const threshold = meanEnergy + stdDev * 1.3;
    const isBeat     = energy > threshold && energy > meanEnergy * 1.2;
    const now        = (this.totalSamples / SAMPLE_RATE) * 1000;
    let   beatFired  = false;

    if (isBeat && now - this.lastBeatTime > MIN_BEAT_GAP_MS) {
      if (this.lastBeatTime > 0) {
        const ibi = now - this.lastBeatTime;
        this.ibiBuffer.push(ibi);

        let bpm =60000 / ibi;

        while (bpm < MIN_BPM) bpm *= 2;
        while (bpm > MAX_BPM) bpm /= 2;

        this.bpmBuffer.push(bpm);
        if (this.bpmBuffer.length > BPM_BUFFER_SIZE) {
          this.bpmBuffer.shift();
        }

      }
      this.lastBeatTime = now;
      beatFired = true;
    }

    return { bpm: this.estimateBpm(), isBeat: beatFired };
  }

  private rmsEnergy(samples: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / samples.length);
  }

  private estimateBpm(): number | null {
    if (this.ibiBuffer.length < 2) return null;

    const meanIbi = this.ibiBuffer.reduce((a, b) => a + b, 0) / this.ibiBuffer.length;
    const rawBpm  = 60_000 / meanIbi;

    // Fold doublings/halvings into a plausible BPM range
    let bpm = rawBpm;
    while (bpm < MIN_BPM) bpm *= 2;
    while (bpm > MAX_BPM) bpm /= 2;



    const smoothed = this.bpmBuffer.reduce((a, b) => a + b, 0) / this.bpmBuffer.length;
    return Math.round(smoothed * 10) / 10;  // 1 d.p.
  }
}

// ------------------------------------------------------------------ //
//  UDP Server
// ------------------------------------------------------------------ //
const detector = new BeatDetector();
const server   = dgram.createSocket('udp4');

// Print the latest BPM to the terminal every 5 seconds so the console
// doesn't flood with per-chunk output.
let latestBpm: number | null = null;
setInterval(() => {
  if (latestBpm !== null) {
    console.log(`[BPM Server] Current BPM: ${latestBpm.toFixed(1)}`);
  } else {
    console.log(`[BPM Server] Waiting for enough data to estimate BPM...`);
  }
}, 5000);

server.on('error', (err: Error) => {
  console.error(`[BPM Server] Fatal UDP error:\n${err.stack}`);
  server.close();
  process.exit(1);
});

server.on('message', (msg: Buffer, rinfo: RemoteInfo) => {
  // Validate: must be a multiple of 4 bytes (Float32)
  if (msg.byteLength === 0 || msg.byteLength % 4 !== 0) {
    console.warn(`[BPM Server] Unexpected packet size: ${msg.byteLength} bytes — ignoring`);
    return;
  }

  const samples          = new Float32Array(msg.buffer, msg.byteOffset, msg.byteLength / 4);
  const { bpm, isBeat }  = detector.process(samples);

  // Send BPM estimate back to the Python bpm_tracker on port 5005
  if (bpm !== null) {
    latestBpm = bpm;   // picked up by the 5-second interval logger
    const bpmResponse = Buffer.alloc(4);
    bpmResponse.writeFloatLE(bpm, 0);
    server.send(bpmResponse, rinfo.port, rinfo.address, (err) => {
      if (err) console.error(`[BPM Server] BPM send error: ${err.message}`);
    });
  }

  // Send a beat pulse to the visualizer on port 5006 so the green LED fires
  if (isBeat) {
    const beatPulse = Buffer.from([0x01]);
    server.send(beatPulse, BEAT_PULSE_PORT, rinfo.address, (err) => {
      if (err) console.error(`[BPM Server] Beat pulse send error: ${err.message}`);
    });
  }
});

server.on('listening', () => {
  const { address, port } = server.address();
  console.log(`[BPM Server] Listening on UDP ${address}:${port}`);
});

server.bind(PORT);

// Graceful shutdown
function shutdown(signal: string): void {
  console.log(`\n[BPM Server] Received ${signal} — shutting down.`);
  server.close(() => process.exit(0));
}

process.on('SIGINT',  () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));