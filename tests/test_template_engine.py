import pytest

from strif.string_template import (
    MissingVariableError,
    RecursionLimitError,
    TemplateSyntaxError,
    compile_template,
    render_template,
    replace_template,
    template_hash,
)


def test_render_strict_missing_raises():
    with pytest.raises(MissingVariableError):
        render_template("${name}", {})


def test_replace_missing_is_blank():
    assert replace_template("a${x}b", {}) == "ab"


def test_escape_is_decided_before_blocks():
    assert render_template("$${x}", {"x": "v"}) == "${x}"
    assert render_template("$${?x}", {"x": "v"}) == "${?x}"


def test_conditions_present_and_non_empty_and_nested():
    tpl = "${?a}A${?b}B${|}bno${/}${|}C${/}"
    assert render_template(tpl, {"a": 1, "b": 1}) == "AB"
    assert render_template(tpl, {"a": 1, "b": ""}) == "Abno"
    assert render_template(tpl, {"a": 0, "b": 1}) == "C"
    assert (
        render_template(
            tpl,
            {},
        )
        == "C"
    )
    # Missing variable inside a skipped branch is not an error.
    assert render_template("${?a}${x}${/}", {"a": ""}) == ""


def test_skipped_branch_missing_variable_is_ok_but_taken_branch_errors():
    with pytest.raises(MissingVariableError):
        render_template("${?a}${x}${|}ok${/}", {"a": 1})


def test_syntax_errors():
    with pytest.raises(TemplateSyntaxError):
        compile_template("${?a}x")
    with pytest.raises(TemplateSyntaxError):
        compile_template("${|}")
    with pytest.raises(TemplateSyntaxError):
        compile_template("${1bad}")


def test_recursion_opt_in_self_reference_hits_depth_limit():
    assert render_template("${x}", {"x": "${y}", "y": "v"}) == "${y}"
    assert render_template("${x}", {"x": "${y}", "y": "v"}, recurse=True) == "v"
    with pytest.raises(RecursionLimitError):
        render_template("${x}", {"x": "${x}"}, recurse=True)
    with pytest.raises(RecursionLimitError):
        render_template("${a}", {"a": "${b}", "b": "${a}"}, recurse=True)


def test_hash_changes_with_content():
    assert template_hash("a") == template_hash("a")
    assert template_hash("a") != template_hash("b")


def test_trace_events_are_in_token_order():
    events = []
    render_template(
        "${a} $${e} ${?a}[${?b}B${/}]${|}x${/}",
        {"a": 1, "b": 1},
        emit=events.append,
    )
    traced = [e for e in events if "pos" in e]
    positions = [e["pos"] for e in traced]
    assert positions == sorted(positions)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "variable"
    assert kinds.index("escape") < kinds.index("condition_open")
    opens = [e for e in events if e["type"] == "condition_open"]
    assert [e["level"] for e in opens] == [1, 2]
    closes = [e for e in events if e["type"] == "condition_close"]
    assert [e["level"] for e in closes] == [2, 1]
