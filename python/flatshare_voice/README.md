# flatshare-voice

Flatshare's realtime voice plane on LiveKit Agents 1.8.8 ([ARCHITECTURE §4.2](../../docs/ARCHITECTURE.md#42-the-voice-pipeline)), for a reader who runs or changes the voice agent or the measurement callers. The package holds:
- the voice agent the deployment runs;
- the lifecycle events it posts to the control plane, and its per-turn stage times;
- the measurement callers that read voice-to-voice latency from waveforms;
- the headers the agent sends to the control plane;
- a factory for the stock OpenAI STT, LLM and TTS plugins, pointed at OpenAI-compatible servers over TCP or a Unix socket.

| Module | Contents |
|---|---|
| `flatshare_voice.worker` | The voice agent (`flatshare-voice-worker start`): Silero VAD, LiveKit's local end-of-turn model, STT/LLM/TTS over the voice plane's sockets, LiveKit's default turn handling |
| `flatshare_voice.events` | `Lifecycle`: a session's LiveKit signals as lifecycle events, and a log line per turn with its stage times; `EventSender`: their delivery to `/v1/events` |
| `flatshare_voice.bots` | Measurement callers (`flatshare-voice bots`): voice-to-voice latency, barge-in to silence, and with `--save-audio` each call's audio |
| `flatshare_voice.callers` | The caller corpus (`flatshare-voice render-callers`): scripted lines in six stock Kokoro voices, 16 kHz WAVs |
| `flatshare_voice.audio` | A real-time microphone track, the agent's speech spans and a recording of both, on one clock |
| `flatshare_voice.protocol` | The agent name `flatshare-voice`, the `X-Session-ID` and `X-Flatshare-Class` names, `room_session_id(room_sid)`, `realtime_headers(session_id)` |
| `flatshare_voice.plugins` | `make_stt`, `make_llm`, `make_tts`, `openai_client` |

## The voice agent

`flatshare-voice-worker start` registers the agent `flatshare-voice` with the LiveKit server and serves each dispatched room with one `AgentSession`. It runs in the voice-plane namespace on cyan as `flatshare-voice-worker.service`, from `~/flatshare/venvs/voice`, which `deploy/cyan/bootstrap.sh` builds from `uv.lock`. The environment configures it:

| Variable | Meaning |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | The LiveKit server (from `~/.config/flatshare/livekit.env`) |
| `FLATSHARE_LLM_UDS` | The control plane's socket (`gateway.sock`), or the engine's tunnel (`phastform-engine.sock`) |
| `FLATSHARE_STT_UDS`, `FLATSHARE_TTS_UDS` | The tunnels to the STT and TTS servers |
| `FLATSHARE_IDLE_PROCESSES` | Pre-started job processes (default 2) |
| `FLATSHARE_LLM_TIMEOUT` | Seconds before an LLM request fails (default 30). A failed request is retried once, where LiveKit's default is three: a refused or reset connection gets a second try, and a timed-out request's retry starts after the callers' 30 s reply timeout has counted the turn, so it cannot hide a slow reply from E2 |
| `FLATSHARE_TURN_DETECTION` | `model` (default) or `vad` |
| `FLATSHARE_PREEMPTIVE_GENERATION` | `true` (default) or `false` |
| `FLATSHARE_EVENTS` | `on` (default): lifecycle events to the control plane through `FLATSHARE_LLM_UDS`, every one in bench mode and `session_closed` alone in serve mode ([Lifecycle events](#lifecycle-events-and-turn-lines)); `off` |
| `FLATSHARE_LLM_MODEL` | The model the agent's requests name (default the box's, `Qwen/Qwen3-4B-Instruct-2507-FP8`). An engine that serves another model answers a request for any other with 404: the `qwen3-14b` worker variant names Qwen3-14B-FP8, for the A100 kit's optional arm C' |
| `FLATSHARE_VOICE_MODE` | `bench` (default; the unit sets it): every call taken; or `serve`: call-capacity admission (below) |
| `FLATSHARE_SERVING_PROFILE` | Serve mode: a serving profile's `serving.json` (the box's, `deploy/phastform/serving.json`), whose `voice` table names the calls admitted at once and the audio configuration |
| `FLATSHARE_AUDIO` | `synthetic` (bench mode's default) or `microphone` (serve mode's, the profile's) |
| `FLATSHARE_ENDPOINTING_MIN_DELAY` | Seconds, 0 to 3, after the end of speech before a turn may end, over the audio configuration's (LiveKit's 0.5 by default): the endpointing sweep's ([`benchmarks/endpointing`](../../benchmarks/endpointing/sweep.sh)) |
| `FLATSHARE_VAD_MIN_SILENCE` | Seconds, above 0 and at most 3, of silence that end speech for Silero VAD, over the audio configuration's (Silero's 0.55 by default): the endpointing sweep's |
| `FLATSHARE_VOICE_STATUS` | Serve mode: the file the worker writes its state to every second (`%t/flatshare/voice-worker.json` in the unit) |
| `FLATSHARE_VOICE_DRAIN` | Serve mode: while this file exists the worker takes no new call (`%t/flatshare/voice-worker.drain`, which a release makes to drain the worker) |
| `FLATSHARE_NODE_BINDING` | `on`, serve mode's default: each call asks the gateway for its GPU node and binds its STT and TTS to that node's sockets ([Nodes](#nodes)); `off`, bench mode's default (the `node-binding` variant, `deploy/cyan/variants/node-binding`, turns it on there), and refused `on` where `FLATSHARE_LLM_UDS` bypasses the gateway |

The worker reads these whenever it is to serve (`start`, which its unit runs, or `dev`, `console` or `connect`), before it registers with LiveKit: a socket that is not set, or a value its setting does not take, stops it with a line that names the setting (`journalctl --user -u flatshare-voice-worker`), so that no call reaches a worker that would fail it.

Experiments change two settings. LiveKit's end-of-turn model waits up to the endpointing maximum (3 s) when it judges a line unfinished: on the synthesized callers that happened on about one turn in six, and put a 3 s mode in voice-to-voice latency unrelated to serving. Experiments therefore run with `FLATSHARE_TURN_DETECTION=vad`. That variant is `deploy/cyan/variants/vad-turns`: a turn ends a fixed delay after the end of speech (VAD plus LiveKit's 0.5 s minimum), the same for every arm. LiveKit's preemptive generation, on by default, starts the reply from the final transcript before the end of the turn is confirmed; `FLATSHARE_PREEMPTIVE_GENERATION=false` (the variant `deploy/cyan/variants/no-preemptive-generation`, which `flatshare-bench e2 run` links for an arm that asks for it) turns it off, one of E2's factors.

Everything else is LiveKit's default except two settings of the synthetic callers' audio configuration, whose callers are the measurement bots in the namespace: interruptions are detected by VAD (what production mode picks anyway; adaptive detection needs LiveKit Inference), and there is no AEC warm-up, which would discard a barge-in on the first reply of each call although the callers have no echo path.

### Bench and serve modes

Bench mode, the default and what every measurement ran, takes every job LiveKit dispatches, its load always 0, so that other tenants' CPU load on a shared host cannot drain the worker. Serve mode ([`admission.py`](src/flatshare_voice/admission.py)), which the `serve` variant of the worker's unit sets (`deploy/cyan/variants/serve`), admits calls by capacity:

- **The limit.** The calls admitted at once are the serving profile's `voice.max_calls`, two on the box: the measured envelope's. The worker's load is its calls in progress and being accepted over that number, with LiveKit's load threshold at 1 (a plain float, which dev mode holds too), so a full worker is marked full and the server offers it no job, and its request handler refuses a job request once the calls in progress and being accepted reach it: LiveKit counts a request not yet launched at a guessed share of the load, so two that arrive together could both pass its own check, where this count is exact. A refused request may go to another worker (`terminate=False`); with none, the room gets no agent.
- **The engine.** Every second it asks the LLM's socket whether the engine serves: the gateway's `/ready`, which answers 200 only while it does, or, on the bypass, the engine's own `/health`. While it does not, or does not answer within a second, the worker takes no call, and its load is 1. A gateway that serves no `/ready` (404, 405) leaves admission to capacity alone, said once in the journal.
- **A drain.** While the file `FLATSHARE_VOICE_DRAIN` names exists, the worker takes no call; the calls in progress go on. A release makes the file before a switch that would cut calls, waits for them to end and restarts the worker, whose start removes it (`deploy/bin/flatshare-release`).
- **Its state** goes to the file `FLATSHARE_VOICE_STATUS` names every second, whole: its pid, mode, the calls it admits, its audio configuration and turn detection, the calls in progress, whether it takes new ones and why not, and how many it has refused. `deploy/bin/serving-check SERVING_JSON --voice FILE` holds it to the profile.

### Audio configurations

`synthetic`, bench mode's default, is the configuration every E2 point ran: Silero's and LiveKit's defaults, interruptions by VAD, no AEC warm-up. `microphone`, serve mode's (the profile's), is for real callers and was not measured by any benchmark, whose callers are synthetic: LiveKit's end-of-turn model decides the end of a turn, as people pause inside a sentence, and `FLATSHARE_TURN_DETECTION=vad` is refused with it (unlink the `vad-turns` variant to serve); Silero needs an activation of 0.6 (against 0.5) and 0.1 s of speech (against 0.05) to count speech, and an interruption 0.6 s of it (against 0.5), so that a click or a door is no barge-in, and an interruption that no transcript follows within 2 s is taken for a false one, and the agent resumes (LiveKit's default); LiveKit's 3 s AEC warm-up stays on, so that a loudspeaker's echo of the agent does not cut its first reply. Endpointing is LiveKit's default, 0.5 s at least and 3 s at most. Noise suppression is the caller's client's, as a browser's WebRTC applies it: LiveKit's noise cancellation needs LiveKit Cloud, which the namespace cannot reach; the room input's gain control is LiveKit's default. `FLATSHARE_AUDIO` chooses either in either mode: a soak of the serving configuration with the measurement callers runs serve mode with `synthetic`.

### Nodes

Where the gateway fronts several GPU nodes, it places each session on one and keeps it there: the session's KV is in that node's engine, and the call's STT and TTS belong on the same node. With node binding on ([`nodes.py`](src/flatshare_voice/nodes.py); serve mode's default, `FLATSHARE_NODE_BINDING=on` in bench mode), each call:

1. **Asks for its node** before its session starts, on the gateway's socket (`FLATSHARE_LLM_UDS`), and binds its STT and TTS to the node's sockets the answer names, and its LLM to the gateway with its session id, which routes it to the session's node. The plugins keep their clients for the call; only where their requests go changes.
2. **On a node failure**, ends the reply under way and takes no request until the caller speaks again. A failure is a transport error or a timeout on the node's STT or TTS socket, before the answer or while its audio comes, or the gateway's answer to the call's LLM request, or its stream's error event, naming one (below). The session's speech is interrupted where the caller's audio stands, and every request of the call, a retry of the cut sentence or reply among them, is refused before it leaves the worker: a reply cut by a node failure is never continued, retried or followed by a new generation over the audio already played.
3. **At the caller's next speech** (Silero's start of speech, at least half a second before the turn's transcription), asks again, and its STT, TTS and LLM go to the node the gateway names from that turn on. The conversation's transcript is the session's chat context, kept in the worker, so the next reply runs on the new node with all of it; the KV is only a cache, which the new node prefills again.
4. **At its end**, whatever ends it and whether or not its lifecycle events are on, ends its placement, so that its call slot is free for the next call at once.

Without nodes nothing changes: a gateway with one upstream has no placement route and answers 404, and the call is served on the configured sockets, as before, without another request. In bench mode, as every measurement ran, nothing is asked at all.

**The contract with the gateway** (crates/flatshare-gateway, `[[nodes]]`; what the worker assumes, and its tests' fake gateway serves):

- **The request.** `POST /v1/sessions` on the gateway's Unix socket, as the voice tenant, without a key, with the headers `X-Session-ID: voice-<room SID>` and `X-Flatshare-Class: realtime`, and the body `{"session_id": "voice-RM_...", "class": "realtime", "reason": "start"}`; `reason` is `start` at a call's start, `node_failed` after a failure, with `"failed_node": "NAME"`, the node the call ran on, and `unplaced` when an earlier placement did not answer. The worker waits for it at most 1 s.
- **The answer.** `200` with `{"session_id": "voice-RM_...", "class": "realtime", "node": "NAME", "stt": "unix:/run/user/UID/flatshare/NAME-stt.sock", "tts": "unix:/run/user/UID/flatshare/NAME-tts.sock"}`: the node's name (`[A-Za-z0-9._-]{1,64}`) and its speech sockets as the gateway's `[[nodes]]` configure them, which the worker takes only in the directory of its own configured sockets (on cyan the tunnels', `flatshare-tunnel@NAME`'s; a file name alone is taken there too); an answer for another session, a socket elsewhere, or one that does not end in `.sock`, places the call nowhere. `404` or `405`: a gateway without nodes; the call keeps its configured sockets for good. `503` of type `no_call_slot` (every available realtime node at its `max_calls`), `no_healthy_node` (no available realtime node) or `placement_full`: no node takes the call now. In serve mode the worker asks before it accepts a call's job (`nodes.admit`, with the call counted against `max_calls` meanwhile) and refuses the job on any of these, or when the gateway does not answer within the second; the call's own placement at its start then finds the same node. A job it refuses after asking, or whose accept fails, has its placement ended (below). Anything else, or a timeout, after the call has started: no node for now; the call asks again at its next request.
- **The end.** `DELETE /v1/sessions/voice-RM_...` on the gateway's socket, as the voice tenant, waiting at most 1 s, when the call's job ends (`nodes.end`, a shutdown callback of the job), whatever ends it and whether or not its events are on; `200` (`{"session_id": ..., "ended": true}`, or `false` for a session without a placement) frees its call slot at once. Serve mode's admission sends it too for a job it placed that does not run after all: one it refuses after asking (the gateway refused the call, or answered after the second), and one whose accept fails (LiveKit 1.8.8's `AssignmentTimeoutError`, when the server assigns the job to no worker within 7.5 s of the accept), before the job may go to another worker. A gateway without nodes answered `404` to the placement, and gets nothing at the call's end. A failure is logged, never raised, and the placement then idles out after the gateway's `idle_s`.
- **Placement.** Sticky: a session asking again gets its node while that node is healthy, its whole pipeline (engine, STT and TTS); a session whose node failed is placed anew on its next request, this one or an LLM request, on a node whose pipeline is healthy. `failed_node` is the worker's report, which the gateway may weigh with its own health checks; the worker never chooses a node.
- **The LLM.** Every LLM request of the call goes to the gateway with `X-Session-ID` and `X-Flatshare-Class: realtime`, as without nodes, and the gateway sends it to the session's node. An answer of `502` with the error code `upstream_unavailable`, `504` with `upstream_timeout`, or `503` with `no_healthy_node` (`{"error": {"code": ...}}`, whose `type` is the status's OpenAI class), or a stream whose error event's `type` names `upstream_unavailable` or `upstream_timeout` (the gateway's `data: {"error": {...}}` before it breaks a stream off), says that the session's node failed. Any other error (`429`, a full admission point's `503`, a drain's) says nothing of the node, and fails the turn as it does without nodes.

## Lifecycle events and turn lines

Each session tells the control plane when it will next need the GPU ([PROTOCOL.md](../../docs/PROTOCOL.md#lifecycle-events)), with events of the session `voice-<room SID>` from source `voice-worker`: in bench mode all of them, for E2's lifecycle policy, and in serve mode `session_closed` alone, since the serving arrangement's gateway runs no admission policy and acts on no other. Each comes from one LiveKit hook:

| Event | LiveKit 1.8.8 hook | When it fires |
|---|---|---|
| `user_speech_started` | `user_state_changed` to `speaking` | At Silero VAD's start of speech, once it has heard 0.05 s: 0.10–0.16 s after the onset on cyan |
| `user_turn_ended` | `conversation_item_added` with the caller's message | Once the end of the turn is decided and its reply scheduled, as the agent turns `thinking`. The reply's LLM request starts then, or at the final transcript under preemptive generation |
| `playback_started` | the audio output's `playback_started` | As the reply's first audio frame goes to the published track |
| `playback_ended` | the audio output's `playback_finished`, not interrupted | Once the reply has played out; `played_ms` is all of it |
| `interrupted` | the audio output's `playback_finished`, interrupted | When a barge-in interrupts the reply: on its final transcript with LiveKit's default handling, which pauses the audio after 0.5 s of speech first, or after those 0.5 s with `resume_false_interruption=False`. `played_ms` is what the caller heard |
| `session_closed` | `close` | When the session closes, after the caller has left |

`metrics_collected` would give the end of the turn as well, but 1.8.8 deprecates it, and warns on every subscription, in favour of each message's `metrics`, which also hold the stage times below. A reply interrupted before any of its audio was made has no playback to report; the control plane sees its stream close.

**Delivery.** The events go to `POST /v1/events` over `FLATSHARE_LLM_UDS`, the control plane's socket, where the voice tenant needs no key. A hook only queues its event, and one background task per session posts what is queued as a JSON array, with a 1 s timeout: events queued while a POST is under way go out together in the next. Nothing reaches the session from there. A full queue (64 events) drops its oldest event, and a POST that fails, times out or is not answered 202 loses its batch; a lost batch is not retried, because a late timing signal would mislead. The first failure of an outage is logged as a warning, and recovery as information. `event_id` is the session, the sender's nonce and a sequence number (`voice-RM_a1B2c3-5f2a91c3-7`), so that a sender made anew for the session, as when a worker rejoins its room, repeats no id the control plane still remembers; `at_ms` is the wall clock when the hook ran. When the job shuts down it sends what is left, waiting at most 2 s.

**The switch.** `FLATSHARE_EVENTS=off` stops the events (the turn lines stay), so that E2 can measure arm 9 with and without them: `deploy/cyan/variants/no-events`, which `flatshare-bench e2 run` links for an arm with `[voice] events = false`. They stop as well when `FLATSHARE_LLM_UDS` is an engine's tunnel (`*-engine.sock`, the bypass), which takes none. A call bound to a node ends its placement by its own request, events or none ([Nodes](#nodes)).

**Turn lines.** Each reply logs one JSON line, logger `flatshare_voice.turns`, message `turn`, with `session_id`, `turn` (from 0 in each session), `interrupted` and, in ms, the stages LiveKit measured (a stage it did not measure is left out):

| Field | From | To |
|---|---|---|
| `transcript_ms` | the end of speech (VAD) | the final transcript |
| `end_of_turn_ms` | the end of speech | the end-of-turn decision |
| `llm_ttft_ms` | the LLM request | its first token |
| `llm_first_sentence_ms` | the LLM request | the first sentence handed to TTS |
| `turn_to_first_sentence_ms` | the end-of-turn decision | that sentence: `e2e_ms` less the playback latency, `tts_first_audio_ms` and `end_of_turn_ms`, because TTS starts only once the reply is scheduled |
| `tts_first_audio_ms` | that sentence | its first audio |
| `e2e_ms` | the end of speech | the reply's first audio frame on the track: the agent's side of voice-to-voice |

The line ends with the reply's playout, from a pass-through the worker puts at the top of the session's audio output (`events.Playout`), which notes when each frame of the reply reaches the output and the stretches LiveKit's room output reports as played (`playback_progressed`):

| Field | Meaning |
|---|---|
| `audio_ms`, `played_ms` | The audio the reply handed to the output, and how much of it played (less after a barge-in, or a pause that discarded what the output held) |
| `stalls_ms` | Each time the output ran out of audio while the reply had more to come, 10 ms (one WebRTC frame) or longer: the gaps between stretches of playback that no pause of the session's explains, the gap between two segments of a reply among them. Empty when the audio never ran out |
| `inventory_low_ms` | The least audio ready ahead of the playhead once synthesis had fallen behind it, just before each frame that arrived later than the frame before it lasted; absent when synthesis stayed ahead of playback, as it does when the next sentence's audio is ready before the last one has played. The start of a reply is no such moment: its first frame plays as it arrives |

A reply whose output reported no stretch of playback has only `audio_ms` and `played_ms`: its stalls and inventory are unmeasured, not zero.

When a session ends, one line (logger `flatshare_voice.events`, message `session ended`) gives its `turns` and, with events on, `events_sent`, `events_dropped` (pushed out of the queue, or unsent at the end) and `events_failed` (sent but not accepted). `flatshare-bench e2 run` reads both kinds of line from the worker's journal.

## Measurement callers

`flatshare-voice render-callers` renders the corpus once through a TTS server:
- 40 caller lines and 8 interruptions for a restaurant's phone line;
- six stock Kokoro voices, none of them the agent's;
- 16 kHz mono WAVs with 0.2 s of silence at each end, plus `manifest.json`.

`flatshare-voice bots` runs callers inside the namespace, each in its own room. The join token dispatches the agent. Each turn:
1. The caller plays a line.
2. It waits for the agent's reply: voice-to-voice runs from the end of its speech to the agent's first audio above the speech threshold.
3. On a share of turns it interrupts after a random delay: barge-in to silence runs from the interruption's speech onset until the agent's audio stops, and the reply to the interruption is a second voice-to-voice sample.

Every time comes from `time.monotonic()` in one process, so no clock offset enters. The run directory gets `run.json`, `turns.jsonl` and `summary.json`, which includes the microphone loop's worst lateness as a check that the caller kept up.

With `--warmup-turns N` each call opens with N warm-up turns, played but not measured: a line and its reply, never interrupted, drawing their lines and pauses from a generator of their own, so that the measured turns after them are those of the same call without them. They are numbered first in `turns.jsonl`, with `"warmup": true`, and `summary.json` leaves them out.

With `--in-order` each caller plays the corpus's turn lines in order rather than drawing each, caller i's measured turns from line i × TURNS on: one caller with as many turns as the corpus has lines plays every line once. The endpointing sweep ([`benchmarks/endpointing/sweep.sh`](../../benchmarks/endpointing/sweep.sh)) plays them so, and reads `spoke_during_turn`, a turn whose answer began before its line had ended, as a pause inside the line taken for the turn's end.

A turn fails when the agent does not answer within the reply timeout (30 s), or does not stop speaking. Its line in `turns.jsonl` has an `error` and keeps what the turn measured before it. A turn draws its interruption with its line, so a failed turn leaves the turns after it as they were: a point with failures plays the same calls as the others. In `summary.json` a sample the timeout cut short counts at the timeout, censored (`censored` gives how many), never as missing: a turn that was never answered raises the percentiles instead of leaving them.

```bash
flatshare-voice render-callers ~/flatshare/data/callers --tts-url http://localhost/v1 --uds $XDG_RUNTIME_DIR/flatshare/phastform-tts.sock
set -a; . ~/.config/flatshare/livekit.env; set +a
~/flatshare/src/deploy/bin/in-voicenet flatshare-voice bots --corpus ~/flatshare/data/callers \
  --callers 2 --turns 25 --barge-in 0.25 --out ~/flatshare/runs/<run>/bots
```

**Saving the calls' audio.** With `--save-audio` each caller also keeps the frames it already handles, what it sends and the agent's audio as received, each with the time it handled it, and once every call has ended it writes two files into the run directory:
- `call-<i>.wav`: stereo 16-bit at 24 kHz, the agent's rate. On the left is what the caller sent, upsampled from 16 kHz; on the right the agent's audio as received. Both are laid out on the caller's monotonic clock from the first frame of either: frames go one after another, as their sender made them, each as early as the frames after it allow, so the jitter of their arrival leaves no gap and a real pause stays a pause.
- `call-<i>.json`: when the WAV starts, on the monotonic and the wall clock (`start`), its `rate`, `channels` and length (`seconds`), and each of the caller's turns as `turns.jsonl` has it, with `at`: when its events happened, in seconds into the WAV. They are the line's `start` and `speech_end`, the `reply` (the agent's first audio), for a barge-in the interruption's onset (`barge_in`) and `barge_in_speech_end`, the agent's `silence` and its `barge_in_reply`, and the turn's `end`, when the agent's audio stopped for good. A turn that failed keeps the times it got to.

Recording does not change what is measured: each frame is appended where a caller already handles it (a copy of 480 bytes for each of the agent's), and the files are written once the calls have ended. A 6-minute call keeps about 32 MiB (90 KiB a second of call), and writing it takes about 115 MiB more for a moment, one call at a time. [`benchmarks/e2/demo.py`](../../benchmarks/e2/README.md#the-demo-video) renders the demo video from two saved calls.

The measured numbers are in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md#voice-on-the-shared-gpu).

## Arriving calls (a soak's)

`flatshare-voice arrivals` runs the callers a soak needs ([`flatshare-bench soak`](../flatshare_bench/README.md#soaks)): calls that arrive one at a time, a Poisson process of `--calls-per-hour`, each of `--turns` measured turns after `--warmup-turns` (1 by default: a serving worker's AEC warm-up ignores an interruption of its first reply) with the callers' think times, for `--duration` seconds. Each call is one of `bots`' callers in a room of its own, its turns measured exactly as E2's are; a call due within `bots`' join stagger (0.25 s) of the last one joins that much later, since LiveKit's dispatcher can drop the job of a room created within milliseconds of another's. The arrival times come from the seed and the run's number (`--run`), each call's lines and pauses from the seed and its number (`--first-call` numbers a run's calls after the soak's earlier ones).

Every attempt gets a line in `attempts.jsonl` as it ends, with when it was due and made, its room and the voice worker's status as it was made (`worker`: the status file the worker writes every second in serve mode, `FLATSHARE_VOICE_STATUS`):

| `outcome` | What happened |
|---|---|
| `accepted` | An agent joined the room (`dispatch_ms` after the join, listening `listening_ms` after it). `ended` says how the call went on: `completed` (every turn played), `dropped` (`reason`: the agent never listened, left, or the room went away) or `cut` (the run ended: `SIGINT`, `SIGTERM`, or `hung_up` by a second signal); `turns` and `errors` count its turns |
| `refused` | No agent came within `--dispatch-timeout` (15 s; LiveKit dispatches a room once). `reason`, from the worker's status as the attempt was made and as it gave up (`worker_after`): `capacity` (full, "N of M calls", or a job request it refused meanwhile), `placement` (a job request it refused meanwhile with room for the call, binding calls to GPU nodes: the gateway placed the call on no node), `engine` (the gateway says the engine does not serve), `draining` (a release drains it), `no_agent` (it admitted calls and refused none: LiveKit lost the dispatch), `no_status` (no status file: a worker in bench mode) or `worker_down` (a status too old for a running worker) |
| `failed` | The caller could not join its room and publish its microphone (`reason` `connect`, LiveKit's error in `detail`), or a fault of its own (`error`) |
| `skipped` | Due while `--max-open` attempts (8) were open: the load generator's bound, never made |
| `abandoned` | Hung up by a second signal before its outcome was known |

`turns.jsonl` gets each turn's line as `bots` writes it, with the run's number (`run`) and when the turn ended on the wall clock (`at`); `run.json` the configuration (no secret) and the run's start and end; `summary.json`, at the end, the attempts by outcome and reason, how the accepted calls ended, and the measured turns with their voice-to-voice percentiles and replies (`flatshare_stats.replies`, as `bots`' summary has them). No call arrives after `--duration`, and the calls in progress play their turns to the end. SIGINT or SIGTERM ends a run early: no call arrives after it, and each call in progress ends with its current turn; a second one hangs every call up at once. Either way the run writes its summary and exits 0.

```bash
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

The `phastform-*.sock` files are the tunnel's Unix-socket forwards (`deploy/cyan`). Create one LLM per session: the headers belong to the instance.

## Tests

Unit tests need no network. CI runs them with `make test-python`:

```bash
uv run --extra test pytest -m "not integration"      # in python/flatshare_voice
```

`tests/` checks, grouped by what each part holds to:

- **Lifecycle events** (`test_events.py`): a stand-in session's hooks, driven with LiveKit's own event types, give the protocol's events, each checked against flatshare-proto's schema; a turn line holds its stage times; a failing hook never reaches LiveKit; delivery to a fake control plane on a Unix socket: order and batching, the ids (a new sender for the same session repeats none), a timeout, a refusal and recovery, the full-queue drop of the oldest events, a missing socket, and the end of a session, which sends its last events and logs its counts; a call placed and observed as the worker's entrypoint does it, which in serve mode posts `session_closed` alone where bench mode posts every event, and whose end ends its placement, events or none.
- **The callers** (`test_callers_and_bots.py`): the corpus, its 16 kHz clips and manifest, and resampling that keeps the speech band and rejects aliases; the summary, which pools V2V samples, censors the replies that never came and leaves the warm-up turns out; a call that opens with its warm-up turns and measures the same turns after them; a failed turn, after which the call plays the same lines and interruptions as answered; an unanswered interruption, which fails the turn and keeps its reply; the callers joining their rooms `JOIN_STAGGER_S` apart (`test_the_callers_join_their_rooms_a_stagger_apart`).
- **Arriving calls** (`test_arrivals.py`): the arrival times, a Poisson process of the rate (exponential gaps, memoryless) drawn from the seed and the run alone; every attempt's outcome against fake callers, a failed join, an agent that never came, never listened or left after two turns, a full worker's refusal by capacity, and the reasons the worker's status gives (full, the engine, a drain, none, stale, a lost dispatch); a call due while too many are open, skipped; calls numbered through a soak's runs; SIGINT, which ends each call with its turn and writes the summary, and a second one, which hangs every call up at once; the command line's settings. A call told to stop ends before its next turn, and one not told plays every turn as E2's do (`test_callers_and_bots.py`).
- **The saved calls** (`test_callers_and_bots.py`): the layout of a recorded track, its frames together and its pauses kept; the caller's track upsampled stretch by stretch exactly as a whole; two turns, one with a barge-in, through a fake microphone and agent track, whose WAV shows each line's end and each reply's onset where `call-<i>.json` puts them, and V2V as the callers measured it.
- **The worker** (`test_worker.py`): LiveKit's turn handling by default; the experiments' fixed endpointing, and preemptive generation turned off; events to the control plane's socket unless switched off, every one in bench mode and `session_closed` alone in serve mode; the box's model unless told another; a misspelled setting or a missing socket stopping the worker before it registers.
- **Nodes** (`test_nodes.py`): against a fake gateway that places sessions, two nodes' mock STT and TTS on the tunnels' socket names, and a node whose TTS breaks off halfway: a call's speech going to the node the gateway names; a gateway without nodes (404), and binding off, changing nothing, and ending no placement at the call's end; serve mode's admission ending the placement of a call whose accept fails (`AssignmentTimeoutError`) or that it refuses after asking, before the refusal (`test_admission.py`); a node failure that cuts the reply, after which the cut sentence is synthesized by no node again and no request leaves until the caller speaks, then placed anew with the failed node named, its STT and TTS on the new node; the LLM's requests with the session's headers, and the gateway's upstream failures (an answer's error code, a stream's error event, each as the gateway's fixture of its errors has it) fencing the call where its other errors do not; the whole conversation sent after the rebinding; placements that name a path, another session or no `.sock` refused; a placement that does not answer asked again at the first request; the session's speech interrupted once per failure, and the next speech placing the call; serve mode binding by default and bench mode when told, never with the bypass.
- **The plugins and the protocol** (`test_plugins.py`, `test_protocol.py`): the worker letting idle connections go before the servers do; the LLM streaming with the session's headers, STT, and TTS's raw PCM over a Unix socket; the header names as flatshare-proto has them; a room's session id scoped to the room instance; the realtime headers, a new dict each call, and unsafe session ids refused.
- **The namespace audit** (`test_socket_audit.py`): `ss`'s addresses and links parsed; the namespace's layout passing; public and wildcard binds failing, and the server barred from the dummy address; host problems flagged for the test's own processes only.

The integration test (the W0 barge-in and header findings) runs on cyan. It needs Linux with unprivileged user namespaces,
`unshare`, `nsenter`, `setpriv`, `ip` and `ss`, and `livekit-server` 1.13.10:

```bash
cd ~/flatshare/src/python/flatshare_voice
export UV_PROJECT_ENVIRONMENT=~/flatshare/venvs/voice
LIVEKIT_SERVER_BIN=~/flatshare/bin/livekit-server uv run --locked --extra test pytest -m integration -v
```

A run takes about 35 s and prints the measured latencies at the end. Logs and a JSON
record per room are kept under pytest's base temp directory (`--basetemp` to choose it).
The test skips when it is not on Linux or cannot find `livekit-server`.

What the test does:

- It creates a private user and network namespace that holds `lo` and a dummy interface.
  The namespace has no route out.
- Inside the namespace it starts a LiveKit server with a loopback-only config (the
  `LIVEKIT_CONFIG` template in `tests/integration/plane.py`, ports 27880/tcp and
  27882/udp), an agent worker, and one bot per case.
- The agent worker is `tests/integration/agent_app.py`. It uses this package's factory
  and Silero VAD.
- The mock STT, LLM and TTS run in the test process, outside the namespace. The agent
  reaches them only through their Unix sockets, as the deployed worker reaches the
  control plane and the tunnels.
- The bot plays a LibriSpeech utterance. Once the agent's reply audio starts, the bot
  waits 1 s, then barges in with a second utterance. Meanwhile the mock LLM is still
  streaming: one token per 100 ms, for 20 s.
- The test asserts five things:
  - every LLM request in the room carries `X-Session-ID: voice-<room SID>` and
    `X-Flatshare-Class: realtime`;
  - the stream that was open at the barge-in is closed by the client, within the bound;
  - the lifecycle events reach the mock LLM's socket, which takes them as the control
    plane's does, all of them and in order: the turn, the reply, the barge-in that
    interrupts it, within 1 s of the stream closing and with `played_ms` within 250 ms of
    what the bot heard, its turn, the reply to it, the session's end; and the worker logs
    turn lines;
  - nothing in the namespace listens outside loopback, except the WebRTC clients' ICE
    sockets on the dummy address;
  - none of the test's processes owns a TCP or UDP socket on the host.

Namespace commands, as the test runs them:

```bash
# holder: prints "ready", then keeps the namespace alive
unshare --user --map-root-user --net -- sh -c 'ip link set lo up && ip link add flatshare0 type dummy && ip link set flatshare0 addrgenmode none && ip addr add 10.255.0.1/32 dev flatshare0 && ip link set flatshare0 up && echo ready && exec sleep infinity'
# every other process
nsenter --target=<holder pid> --user --net --preserve-credentials -- <command>
```

## W0 findings (cyan, 2026-10-22)

**Per-session headers.**
- The mechanism is the 1.8.8 constructor option `livekit.plugins.openai.LLM(extra_headers=...)`.
  `LLM.chat()` passes it to `chat.completions.create` on every request of that instance.
- The plugin writes its own headers into the same dict:
  - `User-Agent: LiveKit Agents/1.8.8 ...`;
  - `X-LiveKit-Room-ID` (the room SID), `X-LiveKit-Job-ID` and `X-LiveKit-Agent-ID`;
  - a new `X-LiveKit-Session-ID: inference_...` on every request. Despite its name, that
    one is a request id, not the conversation.
- `api_key` goes out as `Authorization: Bearer <key>`. The worker sends the factory's
  placeholder key: the gateway's voice socket names its tenant, so it needs none.
- All three plugins accept `client=`. The factory passes an `openai.AsyncClient` whose
  `httpx2` transport is `AsyncHTTPTransport(uds=...)`.
- Request timeouts: the LLM plugin sets each request's timeout from the session's
  `llm_conn_options.timeout`, 10 s by default, for connect, read, write and pool. The mock
  saw `x-stainless-read-timeout: 10.0`. The client's own timeout covers only the prewarm
  `GET /models`. Cold prefills longer than 10 s need
  `AgentSession(conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(timeout=...)))`.

**A barge-in closes the LLM stream.** Measured from the bot's barge-in speech onset to the
mock seeing the client disconnect, over 10 runs:

| Interruption handling | Median | Range | Bound in the test |
|---|---|---|---|
| Stock (`resume_false_interruption=True`) | 2.19 s | 2.17–2.20 s | 3.0 s |
| `resume_false_interruption=False` | 0.58 s | 0.57–0.62 s | 1.0 s |

In both modes, the agent's VAD saw the barge-in 0.10–0.16 s after onset. The agent's
audio stopped at the bot 0.61–0.68 s after onset.

`AgentActivity` decides when to interrupt, from Silero VAD:

- **Stock.** After 0.5 s of barge-in speech (`interruption.min_duration`), it only pauses
  the audio output. The agent state turns `listening`, but the LLM keeps streaming.
- The reply is interrupted only when the barge-in turn's final transcript arrives. That
  takes VAD end of speech (0.55 s of silence) plus the STT call. The call chain is
  `on_final_transcript` → `_cancel_speech_pause()` → `SpeechHandle.interrupt()`.
- So the stream stays open for the whole barge-in utterance plus about 0.9 s. The test's
  barge-in has 1.31 s of speech.
- **`resume_false_interruption=False`.** After 0.5 s of barge-in speech,
  `_interrupt_by_audio_activity()` calls `SpeechHandle.interrupt()` right away.
- A false barge-in (a cough) then kills the reply. In stock mode the reply would resume
  after `false_interruption_timeout` (2 s).

From `SpeechHandle.interrupt()`, both modes follow the same chain:

1. The reply task (`_pipeline_reply_task`) cancels its LLM task with `cancel_and_wait`.
2. The `CancelledError` unwinds the default `llm_node` out of `async with llm.chat(...)`.
3. `LLMStream.aclose()` cancels the plugin's `_run`.
4. In `_run`, `async with stream:` calls `openai.AsyncStream.close()`, and that calls
   `httpx2.Response.aclose()`.
5. The response is unfinished, so the client closes the HTTP/1.1 connection instead of
   pooling it. That close is the disconnect the control plane sees.

The agent's LLM node reports the cancellation within 10 ms of the mock seeing the
disconnect.

**Network.**
- libwebrtc (livekit-rtc 1.1.20) gathers ICE host candidates on every non-loopback
  interface, whatever the server's rtc block says. On cyan's host network, one process
  with two connected rooms held 9 UDP sockets and 7 passive ICE-TCP `LISTEN` sockets on
  the host's public address.
- With only `lo` in a namespace, libwebrtc gathers no candidate at all: it ignores
  loopback adapters. `Room.connect` then fails with `wait_pc_connection timed out`.
- A dummy interface with a /32 address fixes it. Media flowed within 0.75 s, and the
  server still offers only 127.0.0.1 candidates.
- On the server, `--bind 127.0.0.1` covers signalling only. `rtc.tcp_port` must be 0,
  because the ICE-TCP listener ignores the IP filters. The single `rtc.udp_port` binds
  only the filtered IPs.
- `AgentServer` defaults matter:
  - `host=""` puts its health server on all interfaces;
  - in production mode, `num_idle_processes` defaults to the CPU count (48 on cyan);
  - in production mode, the worker refuses jobs above 70 % host CPU load.

**Other 1.8.8 defaults, and what the worker and the test set.**
- `aec_warmup_duration` defaults to 3.0 s, which ignores barge-ins during the first 3 s of
  the session's first agent turn. The callers have no echo path, so the worker and the test
  set `None`.
- `turn_detection` defaults to LiveKit Inference's `TurnDetector()`, and an unset
  `interruption.mode` means adaptive (LiveKit Inference) in dev mode and VAD in production
  mode. The worker runs the local `v1-mini` model (or `"vad"` with `FLATSHARE_TURN_DETECTION=vad`)
  and pins `"vad"` interruptions; the test pins `"vad"` for both. Nothing is downloaded:
  the `v1-mini` weights ship in the `livekit-local-inference` wheel and Silero's in its
  plugin's, so the worker starts with `HF_HUB_OFFLINE=1` and an empty Hugging Face cache
  (`deploy/cyan/bootstrap.sh` checks that both load offline). `download-files` serves the
  older text turn-detector plugin, which the package does not install.
- `record` follows the server's recording setting. The worker and the test pass
  `record=False`.
- When the user leaves, the job stays in the room until LiveKit's departure timeout,
  unless `RoomOptions(delete_room_on_close=True)` is set. The worker and the test set it.

Test audio: `tests/data/user_turn.wav` and `tests/data/barge_in.wav` are LibriSpeech
dev-clean utterances 1272-135031-0011 and 1272-135031-0012 (CC BY 4.0).
