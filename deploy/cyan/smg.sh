#!/usr/bin/env bash
# smg.sh: E2 arm 8b's router, SMG 1.13.0 as shipped with its priority scheduler on, inside the voice-plane namespace,
# where flatshare-smg.service runs it (in-voicenet). SMG takes only http:// workers (it refuses unix:, and ipc:// in its
# OpenAI mode) and listens on TCP only, so it sits between two of systemd's proxies, each on a listener that
# systemd-socket-activate makes: the namespace's 127.0.0.1:18000, SMG's worker, to the engine's tunnel socket, and
# $XDG_RUNTIME_DIR/flatshare/smg.sock, the gateway's upstream for the arm, to SMG's 127.0.0.1:18090. Every TCP listener
# of the arm is on the namespace's loopback, which no other user of cyan reaches. SMG's pooled upstream connections
# close after 55 s idle, before the engine's 60 s keep-alive, as the gateway's do, and its metrics listen on loopback
# (its default is every interface). The gateway sets X-SMG-Priority from each request's class, so the classes it names
# are admitted up to system. There is no --max-concurrent-requests: it sizes the scheduler's pool only until a worker is
# known, and at 8 SMG's built-in class reservations (system 32, interactive 128) do not fit, so its priority scheduler
# would not start ("falling back to legacy admission").
set -euo pipefail
socket_dir=${XDG_RUNTIME_DIR:?}/flatshare
proxy=/usr/lib/systemd/systemd-socket-proxyd
/usr/bin/systemd-socket-activate -l 127.0.0.1:18000 "$proxy" "$socket_dir/phastform-engine.sock" &
/usr/bin/systemd-socket-activate -l "$socket_dir/smg.sock" "$proxy" 127.0.0.1:18090 &
exec "$HOME/flatshare/venvs/smg/bin/smg" launch --host 127.0.0.1 --port 18090 \
  --worker-urls http://127.0.0.1:18000 --backend openai --upstream-pool-idle-timeout-secs 55 \
  --prometheus-host 127.0.0.1 --prometheus-port 18091 \
  --priority-scheduler-enabled --priority-scheduler-default-max-class system
