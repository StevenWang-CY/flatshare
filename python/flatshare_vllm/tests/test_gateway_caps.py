"""The gateway's reply caps and choices (crates/flatshare-gateway/src/bounds.rs) as vLLM 0.32.0 reads them: every body
of the gateway's fixture (crates/flatshare-gateway/tests/fixtures/reply-caps.json, which the gateway's own tests keep
equal to what it does) is parsed by vLLM's request model for its route, and the reply limit and the choices it would
schedule are derived as its serving layer derives them (vllm/entrypoints/openai/*/serving.py: get_max_tokens over the
route's own field, then to_sampling_params, or to_beam_search_params under beam search). A Messages body is read by
vLLM's AnthropicMessagesRequest, which requires max_tokens, and scheduled as the chat request vLLM makes of it.

A body the gateway forwards under a cap must be scheduled with the limit the gateway wrote, never more than the cap, and
with no more choices than max_choices; one it forwards without a cap must mean to vLLM what the client's did. A body it
refuses must be one vLLM would have scheduled over the cap or the choices, would refuse itself, or would ignore the
field of. The negative controls: the chat body that once passed the cap, with only max_output_tokens, runs to the
model's length without the gateway, and vLLM runs n = 8.

The request models come from the vLLM source checkout on PYTHONPATH, as for the scheduler suites (make test-make-room);
without it, as in make test-vllm, the module skips.
"""

import json
from pathlib import Path

import pytest

chat = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
completion = pytest.importorskip("vllm.entrypoints.openai.completion.protocol")
responses = pytest.importorskip("vllm.entrypoints.openai.responses.protocol")
anthropic = pytest.importorskip("vllm.entrypoints.anthropic.protocol")
api_utils = pytest.importorskip("vllm.entrypoints.serve.utils.api_utils")
exceptions = pytest.importorskip("vllm.exceptions")

# What vLLM raises for a body it refuses with 400: pydantic's ValidationError (a ValueError) and its own
# VLLMValidationError (a VLLMClientError), from the request models and SamplingParams.
REFUSED = (ValueError, exceptions.VLLMValidationError)

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "crates"
    / "flatshare-gateway"
    / "tests"
    / "fixtures"
    / "reply-caps.json"
)
MODELS = {
    "/v1/chat/completions": chat.ChatCompletionRequest,
    "/v1/completions": completion.CompletionRequest,
    "/v1/responses": responses.ResponsesRequest,
    "/v1/messages": anthropic.AnthropicMessagesRequest,
}
# A context longer than every limit in the fixture, and a short prompt: a body without a limit runs to the context.
MAX_MODEL_LEN = 32768
PROMPT_LEN = 8
UNLIMITED = MAX_MODEL_LEN - PROMPT_LEN


def cases() -> list[dict]:
    return json.loads(FIXTURE.read_text())["cases"]


def scheduled(route: str, body: str) -> tuple[int, int]:
    """The reply limit and the choices (or the beam's width) vLLM 0.32.0 schedules for ``body`` on ``route``, from a
    server without default sampling parameters. The body is read as FastAPI hands it to the model, json.loads and
    pydantic's validation, so a key given twice is its last value. Raises one of REFUSED where vLLM refuses the
    body."""
    request = MODELS[route].model_validate(json.loads(body))
    if route == "/v1/messages":
        # vLLM serves a Messages body as the chat request it makes of it, with both limits at its max_tokens
        # (vllm/entrypoints/anthropic/serving.py, to_chat_completion_request).
        chat_body = {
            "model": request.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": request.max_tokens,
            "max_completion_tokens": request.max_tokens,
        }
        return scheduled("/v1/chat/completions", json.dumps(chat_body))
    if route == "/v1/chat/completions":
        asked = (
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens
        )
    elif route == "/v1/completions":
        asked = request.max_tokens
    else:
        asked = request.max_output_tokens
    max_tokens = api_utils.get_max_tokens(MAX_MODEL_LEN, asked, PROMPT_LEN, {})
    if route == "/v1/responses":
        params = request.to_sampling_params(max_tokens, {})
        return params.max_tokens, params.n
    if request.use_beam_search:
        beam = request.to_beam_search_params(max_tokens, {})
        return beam.max_tokens, beam.beam_width
    params = request.to_sampling_params(max_tokens, {})
    return params.max_tokens, params.n


def name(case: dict) -> str:
    """A case's test id: its route, caps and body."""
    return f"{case['route']} cap={case['cap']} degrade={case['degrade']} {case['input']}"


def bound(case: dict) -> int | None:
    """The cap the gateway applied to a case: the lower of its class's and degrade's."""
    caps = [cap for cap in (case["cap"], case["degrade"]) if cap is not None]
    return min(caps) if caps else None


def test_the_fixture_has_every_kind_of_case() -> None:
    every = cases()
    assert len(every) > 200
    assert {case["route"] for case in every} == set(MODELS)
    assert {case.get("code") for case in every if case["status"] == 400} == {
        "duplicate_field",
        "invalid_reply_limit",
        "invalid_choices",
        "too_many_choices",
    }
    assert any(case["status"] == 200 and bound(case) is None for case in every)
    assert any(case["status"] == 200 and case["degrade"] is not None for case in every)


@pytest.mark.parametrize("case", [c for c in cases() if c["status"] == 200], ids=name)
def test_a_body_the_gateway_forwards_is_scheduled_within_its_bounds(case: dict) -> None:
    try:
        limit, choices = scheduled(case["route"], case["output"])
    except REFUSED:
        # vLLM refuses a body the gateway forwards only where it refuses the client's own (Completions' n is an int,
        # so it refuses n = null): the gateway never makes a body vLLM takes into one it does not.
        with pytest.raises(REFUSED):
            scheduled(case["route"], case["input"])
        return
    assert choices <= case["max_choices"]
    cap = bound(case)
    if cap is None:
        # Without a cap the gateway leaves the limits alone: vLLM reads the body as it would the client's.
        assert "limit" not in case
        assert (limit, choices) == scheduled(case["route"], case["input"])
    else:
        assert limit == case["limit"] <= cap


@pytest.mark.parametrize("case", [c for c in cases() if c["status"] == 400], ids=name)
def test_a_body_the_gateway_refuses_is_one_vllm_would_not_bound(case: dict) -> None:
    model = MODELS[case["route"]]
    if case["param"] not in model.model_fields:
        # A field the route's model does not have, which vLLM keeps as an extra and never reads (best_of in 0.32, or
        # another route's limit): the gateway is stricter than it needs to be, never looser.
        return
    try:
        limit, choices = scheduled(case["route"], case["input"])
    except REFUSED:
        return  # vLLM refuses it too
    assert limit > bound(case) or choices > case["max_choices"], case


def test_without_the_gateway_the_old_bypass_runs_uncapped_and_n_multiplies() -> None:
    every = cases()
    bypass = next(
        case
        for case in every
        if case["route"] == "/v1/chat/completions"
        and case["cap"] == 24
        and case["degrade"] is None
        and json.loads(case["input"]).keys() == {"model", "messages", "max_output_tokens"}
        and json.loads(case["input"])["max_output_tokens"] == 4000
    )
    # vLLM's chat model keeps max_output_tokens as an extra field, so the client's body runs to the context's end.
    assert scheduled(bypass["route"], bypass["input"]) == (UNLIMITED, 1)
    assert scheduled(bypass["route"], bypass["output"]) == (24, 1)
    # And it schedules every choice it is asked for.
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "n": 8})
    assert scheduled("/v1/chat/completions", body) == (UNLIMITED, 8)
    eight = next(
        case
        for case in every
        if case["route"] == "/v1/chat/completions" and '"n":1,"n":8' in case["input"]
    )
    assert eight["status"] == 400 and eight["code"] == "duplicate_field"
    assert scheduled(eight["route"], eight["input"])[1] == 8
