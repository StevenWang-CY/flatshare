# Speech synthesis and recognition on the CPU (2026-10-27)

The voice servers hold about 3.4 GiB of the RTX 5060 Ti's memory (STT 2.4 GiB, TTS 1.0 GiB; docs/ARCHITECTURE.md),
which the KV pool could take, about 24,800 tokens more at the engine's 147,456 bytes a token, if the two servers ran on
the CPU. [`benchmarks/audio/cpu_audio.py`](../../audio/cpu_audio.py) measured what that would cost the calls: the box's
own TTS server and Whisper large-v3-turbo on the CPU, with 1, 2 and 4 requests at once.

Measured on Gray, an Intel Xeon Silver 4108 (two sockets of 8 cores, 1.8 GHz, 3.0 GHz turbo, AVX-512 with one FMA
unit), each server on the 8 cores of socket 0, the client on other cores. Phastform's Xeon E5-2609 v4 (8 cores a
socket, 1.7 GHz, no turbo, AVX2) is slower, so the box itself would do no better.

- **TTS.** Kokoro-FastAPI at the box's revision (ebd4ec9, kokoro 0.9.4, torch 2.8.0) with `USE_GPU=false` and 8
  threads, asked for 40 of the voice agent's first sentences (from
  [the first-sentence record](../2026-10-27-first-sentence)) as the agent asks for them: voice `af_heart`, raw 24 kHz
  PCM, streamed, after one request to warm it. `tts.jsonl` holds each request's first audio and total time and the
  audio's length.
- **STT.** faster-whisper 1.2.1 (CTranslate2 4.8.4) with large-v3-turbo's weights in INT8
  (`deepdml/faster-whisper-large-v3-turbo-ct2` at revision 4df90f75321148c3a29a9e2351b7ddf8f5b115a8, a conversion of
  `openai/whisper-large-v3-turbo`), greedy, without timestamps or VAD, on 20 of the caller corpus's lines, each
  transcribed whole as the worker's STT gets an utterance; with 8 threads (`stt-t8.jsonl`) and with 4
  (`stt-t4.jsonl`). The word error rate counts digits against the lines' spelled numbers ("4" for "four"), which make
  most of its errors.

| Requests at once | TTS first audio, p50 / p95 | TTS real-time factor, p50 | STT, 8 threads: transcript p50 / p95 | STT, 4 threads: p50 / p95 |
|---:|---:|---:|---:|---:|
| 1 | 991 / 1,347 ms | 0.42 | 9,045 / 9,220 ms | 16,347 / 16,461 ms |
| 2 | 2,736 / 4,420 ms | 0.75 | 18,299 / 20,873 ms | 18,259 / 18,722 ms |
| 4 | 3,988 / 7,353 ms | 1.28 | 30,505 / 33,655 ms | 31,100 / 34,411 ms |

The lines last 2.76 s at the median, and the transcripts' word error rate is 3.8% with either thread count. On the GPU
the TTS stage takes 211–242 ms and the voice turn's whole budget over voice alone is 250 ms (docs/SLO.md). The CPU's
first audio alone is 0.75 s later with one call and seconds later with two, and a transcript takes 9 s or more: every
utterance pays the encoder's pass over 30 s of padded audio, which turbo does not shorten. Neither server can move to
these CPUs without breaking the SLO, so their GPU memory stays theirs; the KV cache in FP8 is the cheaper way to a
larger pool.

To measure again (the TTS server started on the CPU on a socket, as `cpu_audio.py`'s docstring says):

```sh
python benchmarks/audio/cpu_audio.py tts --tts unix:PATH --sentences SENTENCES --out DIR --rounds 6
python benchmarks/audio/cpu_audio.py stt --model MODEL_DIR --corpus ~/flatshare/data/callers --out DIR --threads 8 --clips 20
```
