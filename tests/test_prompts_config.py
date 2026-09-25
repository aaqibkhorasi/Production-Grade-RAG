from pathlib import Path

import yaml

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"


def _load_prompts() -> dict:
    return yaml.safe_load(_PROMPTS_PATH.read_text())


def test_grounding_check_prompt_treats_self_declined_answers_as_not_grounded():
    # Regression guard: a draft answer that honestly says "I can't answer this"
    # contains no false claims, so a naive grounding check would call it
    # GROUNDED -- which then leaves stale, irrelevant citations attached to a
    # non-answer. The prompt must explicitly tell the model to treat a
    # self-declared decline as NOT_GROUNDED too.
    system_prompt = _load_prompts()["grounding_check"]["system"].lower()
    assert "not_grounded" in system_prompt
    assert "decline" in system_prompt or "cannot answer" in system_prompt


def test_generate_prompt_does_not_ask_model_to_self_report_sources():
    # Regression guard: an earlier version of this prompt asked the model to
    # "list the source documents you used," which it reliably answered with
    # fabricated, plausible-looking citation numbers not present in the
    # actual retrieved chunks. The real citation panel (built from retrieved
    # chunk metadata) is the only source list that should ever be shown.
    system_prompt = _load_prompts()["generate_answer"]["system"].lower()
    assert "list the source documents" not in system_prompt
    human_template = _load_prompts()["generate_answer"]["human_template"].lower()
    assert "list the source documents" not in human_template


def test_generate_prompt_defines_the_marker_it_is_told_to_emit():
    # The marker is how grounding_check detects a decline deterministically, so
    # the string the generator is instructed to emit and the string the pipeline
    # looks for must be the same one.
    prompts = _load_prompts()
    marker = prompts["insufficient_context_marker"].strip()
    assert marker in prompts["generate_answer"]["system"]
