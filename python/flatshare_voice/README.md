# flatshare-voice

Flatshare's realtime voice plane on LiveKit Agents 1.8.8, for a reader who runs or changes the voice agent or the
measurement callers. It holds the voice agent the deployment runs, with the lifecycle events and per-turn stage times it
reports, and the callers that measure voice-to-voice latency from waveforms.
[ARCHITECTURE.md §3.4](../../docs/ARCHITECTURE.md#34-the-voice-worker) places the worker in the deployment, and
[§6.2](../../docs/ARCHITECTURE.md#62-the-voice-pipeline-in-measurement) in the measurements.

| Module | Contents |
|---|---|
| `flatshare_voice.worker` | The voice agent (`flatshare-voice-worker start`): Silero VAD, LiveKit's local end-of-turn model, and STT, LLM and TTS over the voice plane's sockets |
| `flatshare_voice.events` | A session's LiveKit signals as lifecycle events and turn lines, and their delivery to `/v1/events` |
| `flatshare_voice.bots` | The measurement callers (`flatshare-voice bots`) |
| `flatshare_voice.callers` | The caller corpus (`flatshare-voice render-callers`) |
| `flatshare_voice.audio` | A real-time microphone track, the agent's speech spans and a recording of both, on one clock |
| `flatshare_voice.protocol` | The agent's name and the headers it sends the control plane |
| `flatshare_voice.plugins` | The stock OpenAI STT, LLM and TTS plugins over TCP or a Unix socket |

## The voice agent

`flatshare-voice-worker start` registers the agent `flatshare-voice` with the LiveKit server and serves each dispatched
room with one `AgentSession`. On cyan it runs in the voice-plane namespace as `flatshare-voice-worker.service`, from
`~/flatshare/venvs/voice`. The environment configures it, and a socket that is not set, or a value its setting does not
take, stops it before it registers, with a journal line that names the setting.

| Variable | Meaning |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | The LiveKit server, from `~/.config/flatshare/livekit.env` |
| `FLATSHARE_LLM_UDS` | The control plane's socket (`gateway.sock`), or the engine's tunnel (`phastform-engine.sock`), the bypass |
| `FLATSHARE_STT_UDS`, `FLATSHARE_TTS_UDS` | The tunnels to the STT and TTS servers |
| `FLATSHARE_IDLE_PROCESSES` | Pre-started job processes (default 2) |
| `FLATSHARE_LLM_TIMEOUT` | Seconds before an LLM request fails (default 30). A failed request is retried once, where LiveKit retries three times |
| `FLATSHARE_TURN_DETECTION` | `model` (default) or `vad` |
| `FLATSHARE_PREEMPTIVE_GENERATION` | `true` (default) or `false` |
| `FLATSHARE_TURN_HINTS` | `true` announces each turn to the engine when the user stops speaking, so that its scheduler holds agent prefill back ([`hints.py`](src/flatshare_voice/hints.py)). Serve mode's default is the serving profile's, bench mode's `false` |
| `FLATSHARE_EVENTS` | `on` (default): lifecycle events through `FLATSHARE_LLM_UDS`, every one in bench mode and `session_closed` alone in serve mode; `off` |
| `FLATSHARE_LLM_MODEL` | The model the agent's requests name (default `Qwen/Qwen3-4B-Instruct-2507-FP8`) |
| `FLATSHARE_VOICE_MODE` | `bench` (default): every call taken; or `serve`: calls admitted by capacity |
| `FLATSHARE_SERVING_PROFILE` | Serve mode: a serving profile's `serving.json`, whose `voice` table names the calls admitted at once and the audio configuration |
| `FLATSHARE_AUDIO` | `synthetic` (bench mode's default) or `microphone` (serve mode's) |
| `FLATSHARE_ENDPOINTING_MIN_DELAY` | Seconds, 0 to 3, after the end of speech before a turn may end, over the audio configuration's value |
| `FLATSHARE_VAD_MIN_SILENCE` | Seconds of silence, above 0 and at most 3, that end speech for Silero VAD (default 0.55) |
| `FLATSHARE_VOICE_STATUS` | Serve mode: the file the worker writes its state to every second |
| `FLATSHARE_VOICE_DRAIN` | Serve mode: while this file exists the worker takes no new call |
| `FLATSHARE_NODE_BINDING` | `on` (serve mode's default): each call binds its STT and TTS to its GPU node ([Nodes](#nodes)), refused with the bypass; `off` (bench mode's) |

Experiments end a turn a fixed delay after the end of speech (`FLATSHARE_TURN_DETECTION=vad`, the `vad-turns` variant),
the same for every arm. LiveKit's end-of-turn model waited its 3 s maximum on about one synthesized turn in six, a mode
in voice-to-voice latency unrelated to serving. Preemptive generation, which starts the reply from the final transcript
before the turn's end is confirmed, is one of E2's factors.

### Bench and serve modes

Bench mode, the default and what every E2 point ran, takes every job LiveKit dispatches, its load always 0, so that
other tenants' CPU load on a shared host cannot drain the worker. Serve mode
([`admission.py`](src/flatshare_voice/admission.py)), which the `serve` variant sets, admits calls by capacity:

- **The limit** is the serving profile's `voice.max_calls`, two on the box. The worker's load is its calls in progress
  and being accepted over that limit, with LiveKit's load threshold at 1, so a full worker is offered no job. Its
  request handler also refuses a job at the limit, since LiveKit's own count of a request not yet launched is a guess.
- **The status** goes to the server as soon as a call's process closes, not only every 2.5 s. At the SDK's pace the
  24-hour soak found a full worker held full for up to 3 s after a call ended.
- **The engine.** Every second the worker asks the gateway's `/ready`, or on the bypass the engine's `/health`, and
  takes no call while the engine does not serve. A gateway without `/ready` leaves admission to capacity alone.
- **A drain.** While the drain file exists, which a release makes before a switch that would cut calls, the worker takes
  no new call.
- **Its state** goes to the status file every second, with the calls and why it takes no new one.
  `deploy/bin/serving-check SERVING_JSON --voice FILE` holds it to the profile.

### Audio configurations

`synthetic`, bench mode's default, is the configuration every E2 point ran: Silero's and LiveKit's defaults,
interruptions by VAD and no AEC warm-up. `microphone`, serve mode's, is for real callers, and no benchmark measured it:

- **The end of a turn** is the end-of-turn model's, since people pause inside a sentence, and `vad` is refused with it.
  Endpointing is pinned to 0.5 s at least and 3 s at most, LiveKit's values with VAD turns, which every E2 point ran.
- **Speech and interruptions.** Silero needs an activation of 0.6 and 0.1 s of speech to count speech, and an
  interruption 0.6 s of it, so that a click is no barge-in. An interruption no transcript follows within 2 s is taken
  for a false one, and the agent resumes.
- **Echo and noise.** LiveKit's 3 s AEC warm-up stays on, so that a loudspeaker's echo does not cut the first reply.
  Noise suppression is the caller's client's, since LiveKit's needs LiveKit Cloud.

`FLATSHARE_AUDIO` chooses either in either mode, so a soak of the serving configuration with the measurement callers
runs serve mode with `synthetic`.

### Nodes

Where the gateway fronts several GPU nodes it keeps each session on one, and the call's STT and TTS belong there too.
With node binding on ([`nodes.py`](src/flatshare_voice/nodes.py)), each call:

1. **Asks for its node** before its session starts, and binds its STT and TTS to the node's sockets the answer names,
   and its LLM to the gateway with its session id.
2. **On a node failure** ends the reply under way and refuses every request until the caller speaks again, so that a cut
   reply is never continued or retried over the audio already played.
3. **At the caller's next speech** asks again, and goes on with the whole conversation, kept in the worker, on the node
   the gateway names. The KV is only a cache, which the new node prefills again.
4. **At its end**, whatever ends it, ends its placement, so that its call slot is free at once.

A gateway with one upstream answers 404, and the call keeps its configured sockets. In bench mode nothing is asked. What
the worker assumes of the gateway, and its tests' fake gateway serves
([the gateway's side](../../crates/flatshare-gateway/README.md#nodes)):

| Step | The worker sends | It reads the answer as |
|---|---|---|
| Place | `POST /v1/sessions` on the gateway's socket as the voice tenant, with `{"session_id": "voice-RM_...", "class": "realtime", "reason": ...}`: `start`, `node_failed` with `failed_node`, or `unplaced`. It waits at most 1 s | `200`: the node and its `stt` and `tts` sockets, taken only in the worker's own socket directory and ending in `.sock`. `404` or `405`: no nodes. `503` with `no_call_slot`, `no_healthy_node` or `placement_full`: no node now, on which serve mode refuses the job. Anything else: no node yet, asked again at the next request |
| End | `DELETE /v1/sessions/voice-RM_...`, also for a job serve mode placed and then refused or failed to accept. It waits at most 1 s | `200` frees the call slot; a failure is logged, and the placement idles out |
| LLM | Every request through the gateway with `X-Session-ID` and `X-Flatshare-Class: realtime` | `502 upstream_unavailable`, `504 upstream_timeout`, `503 no_healthy_node`, or a stream error naming the first two: the node failed. Any other error fails the turn as without nodes |

Placement is sticky while the node's whole pipeline is healthy. `failed_node` is the worker's report, and the worker
never chooses a node.

## Lifecycle events and turn lines

Each session tells the control plane when it will next need the GPU
([PROTOCOL.md](../../docs/PROTOCOL.md#lifecycle-events)), from source `voice-worker`. In bench mode it sends every
event, for E2's lifecycle policy, and in serve mode `session_closed` alone, since the arrangement to deploy's gateway
acts on no other. Each comes from one LiveKit hook:

| Event | LiveKit 1.8.8 hook | When it fires |
|---|---|---|
| `user_speech_started` | `user_state_changed` to `speaking` | At Silero's start of speech, 0.10–0.16 s after the onset on cyan |
| `user_turn_ended` | `conversation_item_added` with the caller's message | Once the end of the turn is decided and the reply scheduled |
| `playback_started` | the audio output's `playback_started` | As the reply's first audio frame goes to the published track |
| `playback_ended` | the audio output's `playback_finished`, not interrupted | Once the reply has played out; `played_ms` is all of it |
| `interrupted` | the audio output's `playback_finished`, interrupted | When a barge-in interrupts the reply; `played_ms` is what the caller heard |
| `session_closed` | `close` | When the session closes, after the caller has left |

- **Delivery.** A hook only queues its event, and one task per session posts the queue as a JSON array to
  `POST /v1/events` on `FLATSHARE_LLM_UDS`, with a 1 s timeout. At the job's end it sends what is left, waiting at most
  2 s.
- **Losses.** A full queue of 64 drops its oldest event, and a batch that fails or is not answered 202 is lost, never
  retried, since a late timing signal would mislead. The first failure of an outage is logged, and so is recovery.
- **The switch.** `FLATSHARE_EVENTS=off` (the `no-events` variant) stops the events, so that E2 can measure arm 9 with
  and without them, and so does the bypass. A call bound to a node ends its placement either way.

**Turn lines.** Each reply logs one JSON line, logger `flatshare_voice.turns`, message `turn`, with `session_id`,
`turn`, `interrupted` and the stages LiveKit measured, in ms:

| Field | From | To |
|---|---|---|
| `transcript_ms` | the end of speech (VAD) | the final transcript |
| `end_of_turn_ms` | the end of speech | the end-of-turn decision |
| `llm_ttft_ms` | the LLM request | its first token |
| `llm_first_sentence_ms` | the LLM request | the first sentence handed to TTS |
| `turn_to_first_sentence_ms` | the end-of-turn decision | that sentence |
| `tts_first_audio_ms` | that sentence | its first audio |
| `e2e_ms` | the end of speech | the reply's first audio frame on the track, the agent's side of voice-to-voice |

The line ends with the reply's playout, as the room output reports it. `audio_ms` and `played_ms` are the audio handed
to the output and how much played. `stalls_ms` lists each time the output ran dry for 10 ms or more while the reply had
more to come, and `inventory_low_ms` the least audio ready ahead of the playhead once synthesis fell behind. A reply
whose output reported no playback has no stalls measured, which is not zero. At a session's end a line (message
`session ended`) gives its `turns`, `events_sent`, `events_dropped` and `events_failed`.

## Measurement callers

`flatshare-voice render-callers` renders the corpus once through a TTS server: 40 caller lines and 8 interruptions for a
restaurant's phone line, in six stock Kokoro voices, none of them the agent's, as 16 kHz WAVs. `flatshare-voice bots`
runs callers in the namespace, each in its own room. On each turn the caller plays a line and waits for the reply, and
voice-to-voice runs from the end of its speech to the agent's first audio above the speech threshold. On a share of
turns it interrupts after a random delay. Barge-in to silence runs from its onset until the agent's audio stops, and the
reply to the interruption is a second voice-to-voice sample. Every time comes from `time.monotonic()` in one process, so
no clock offset enters.

- **Warm-up turns.** `--warmup-turns N` opens each call with N turns, played but not measured, from a generator of their
  own, so that the measured turns are those of the same call without them.
- **Lines in order.** `--in-order` plays the corpus's lines in order rather than drawing them, as the endpointing sweep
  ([`sweep.sh`](../../benchmarks/endpointing/sweep.sh)) does.
- **Failed turns.** A turn fails when the agent does not answer within the 30 s reply timeout, or does not stop
  speaking. Its sample counts at the timeout, censored, so a reply that never came raises the percentiles instead of
  leaving them, and the turns after it are the same as in a call without it.

The run directory gets `run.json`, `turns.jsonl` and `summary.json`, which also gives the microphone loop's worst
lateness, a check that the caller kept up. What the callers measured is in
[BENCHMARKS.md](../../docs/BENCHMARKS.md#what-does-agent-load-do-to-a-voice-turn-on-the-same-gpu).

```sh
flatshare-voice render-callers ~/flatshare/data/callers --tts-url http://localhost/v1 \
  --uds $XDG_RUNTIME_DIR/flatshare/phastform-tts.sock
set -a; . ~/.config/flatshare/livekit.env; set +a
~/flatshare/src/deploy/bin/in-voicenet flatshare-voice bots --corpus ~/flatshare/data/callers \
  --callers 2 --turns 25 --barge-in 0.25 --out ~/flatshare/runs/<run>/bots
```

**Saving the calls' audio.** With `--save-audio` each caller keeps the frames it already handles and writes them once
its calls have ended, which changes nothing measured. `call-<i>.wav` is stereo at 24 kHz, the caller's speech on the
left and the agent's as received on the right, laid out on the caller's clock so that a real pause stays a pause.
`call-<i>.json` gives when each turn's events happened, in seconds into the WAV. A 6-minute call keeps about 32 MiB, and
[`benchmarks/e2/demo.py`](../../benchmarks/e2/README.md#the-demo-video) renders the demo video from two saved calls.

## Arriving calls (a soak's)

`flatshare-voice arrivals` runs the callers a soak needs ([`flatshare-bench soak`](../flatshare_bench/README.md#soaks)),
calls that arrive one at a time as a Poisson process of `--calls-per-hour`, each of `--turns` measured turns after
`--warmup-turns` (1), for `--duration` seconds. Each call is one of `bots`' callers in a room of its own. A call due
within 0.25 s of the last joins that much later, since LiveKit's dispatcher can drop the job of a room created
milliseconds after another. The arrival times come from the seed and the run's number, so a soak's runs are
reproducible.

Every attempt gets a line in `attempts.jsonl` as it ends, with when it was due and made, its room and the worker's
status file as it was made:

| `outcome` | What happened |
|---|---|
| `accepted` | An agent joined the room; `ended` says whether the call `completed`, was `dropped` by the agent, or was `cut` by the run's end |
| `refused` | No agent came within `--dispatch-timeout` (15 s), and `reason` says why (below) |
| `failed` | The caller could not join its room and publish its microphone, or failed on its own |
| `skipped` | Due while `--max-open` attempts (8) were open, and never made |
| `abandoned` | Hung up by a second signal before its outcome was known |

A refusal's `reason` comes from the worker's status: `capacity`, `placement` (the gateway placed the call on no node),
`engine`, `draining`, `no_agent` (a dispatch LiveKit lost or held back), `no_status` (a worker in bench mode) or
`worker_down`. At the end `summary.json` gives the attempts by outcome and reason and the measured turns' voice-to-voice
percentiles. SIGINT or SIGTERM ends a run early, each call with its current turn, and a second hangs every call up.

```sh
set -a; . ~/.config/flatshare/livekit.env; set +a
~/flatshare/src/deploy/bin/in-voicenet flatshare-voice arrivals --corpus ~/flatshare/data/callers \
  --calls-per-hour 20 --turns 10 --duration 3600 --out ~/flatshare/runs/<run>/calls
```

## Using the plugin factory

```python
from flatshare_voice import plugins
from flatshare_voice.protocol import room_session_id

socket_dir = f"{os.environ['XDG_RUNTIME_DIR']}/flatshare"
session_id = room_session_id(ctx.job.room.sid)  # "voice-RM_..."
llm = plugins.make_llm(
    "http://flatshare/v1", session_id, uds=f"{socket_dir}/gateway.sock"
)  # the control plane
stt = plugins.make_stt(
    "http://flatshare/v1", uds=f"{socket_dir}/phastform-stt.sock"
)  # the tunnel to the GPU node
tts = plugins.make_tts(
    "http://flatshare/v1", uds=f"{socket_dir}/phastform-tts.sock"
)  # raw 24 kHz PCM
```

The `phastform-*.sock` files are the tunnel's Unix-socket forwards (`deploy/cyan`). Create one LLM per session, since
the headers belong to the instance.

## Tests

The unit tests need no network, and CI runs them with `make test-python`. Each module in `tests/` holds one part, from
the lifecycle events (`test_events.py`) and the callers (`test_callers_and_bots.py`) to node binding against a fake
gateway (`test_nodes.py`).

```sh
uv run --extra test pytest -m "not integration"      # in python/flatshare_voice
```

The integration test runs on cyan in about 35 s. It needs Linux with unprivileged user namespaces, `unshare`, `nsenter`,
`setpriv`, `ip`, `ss` and `livekit-server` 1.13.10, and skips without them. In a private user and network namespace with
no route out (`tests/integration/plane.py`), it starts a loopback-only LiveKit server, an agent worker built on this
package's factory, and a bot. The mock STT, LLM and TTS run outside, reached only through their Unix sockets. The bot
plays a LibriSpeech utterance and barges in 1 s into the reply, while the mock LLM still streams a token every 100 ms.
The test asserts five things:

- every LLM request carries the session's headers;
- the barge-in closes the open stream within its bound;
- the lifecycle events arrive whole and in order;
- nothing listens outside loopback but the WebRTC clients' ICE sockets;
- no test process owns a socket on the host.

```sh
cd ~/flatshare/src/python/flatshare_voice
export UV_PROJECT_ENVIRONMENT=~/flatshare/venvs/voice
LIVEKIT_SERVER_BIN=~/flatshare/bin/livekit-server uv run --locked --extra test pytest \
  -m integration -v
```

## What the integration test measured

On cyan on 2026-10-22, over 10 runs, from the bot's barge-in onset to the mock seeing the client disconnect:

| Interruption handling | Median | Range | Bound in the test |
|---|---:|---:|---:|
| Stock (`resume_false_interruption=True`) | 2.19 s | 2.17–2.20 s | 3.0 s |
| `resume_false_interruption=False` | 0.58 s | 0.57–0.62 s | 1.0 s |

In both modes the agent's VAD saw the barge-in 0.10–0.16 s after its onset, and the agent's audio stopped at the bot
0.61–0.68 s after it.

- **Stock handling only pauses the audio** after 0.5 s of barge-in speech, and the LLM keeps streaming. The reply is
  interrupted when the barge-in's final transcript arrives, after 0.55 s of silence and the STT call, so the stream
  stays open for the whole barge-in plus about 0.9 s.
- **`resume_false_interruption=False`** interrupts the reply after 0.5 s of speech, but a false barge-in such as a cough
  then kills it, where stock handling would resume it after 2 s.
- **The disconnect.** Either way `SpeechHandle.interrupt()` cancels the reply's LLM task, and closing the unfinished
  stream closes its HTTP/1.1 connection rather than pooling it. That close is what the control plane sees, within 10 ms
  of the agent's LLM node reporting the cancellation.
- **Headers.** `livekit.plugins.openai.LLM(extra_headers=...)` sends the session's headers on every request of the
  instance, beside the plugin's own, whose `X-LiveKit-Session-ID` is a request id, not the conversation. Each request's
  timeout is the session's `llm_conn_options.timeout`, 10 s by default, which a cold prefill can outlast, so the worker
  sets 30 s (`FLATSHARE_LLM_TIMEOUT`).
- **The network.** libwebrtc gathers ICE candidates on every non-loopback interface whatever the server allows, and none
  with only `lo`, so the namespace adds a dummy interface with a /32 address. On the server `--bind 127.0.0.1` covers
  signalling only, and `rtc.tcp_port` must be 0.
- **LiveKit's defaults.** `AgentServer` puts its health server on every interface, which the worker binds to 127.0.0.1,
  and in production mode refuses jobs above 70% host CPU, which bench mode's zero load avoids. The worker and the test
  set `record=False` and `delete_room_on_close`, and turn off the 3 s AEC warm-up for the callers, who have no echo
  path. The end-of-turn model and Silero's weights ship in their wheels, so the worker starts offline.

The test audio, `tests/data/user_turn.wav` and `tests/data/barge_in.wav`, is LibriSpeech dev-clean utterances
1272-135031-0011 and 1272-135031-0012 (CC BY 4.0).
