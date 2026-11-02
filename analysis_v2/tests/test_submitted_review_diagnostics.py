from submitted_review_diagnostics import match_excerpts, poll_origins


def source(index, text):
    return dict(message_index=index, source_field="observation", text=text)


def test_joined_quotes_must_match_every_fragment():
    assert match_excerpts("first […] second", [source(1,"first"),source(2,"second")])["status"] == "joined_exact_fragments"
    assert match_excerpts("first […] invented", [source(1,"first")])["status"] == "unmatched"


def test_literal_pipe_is_kept_in_single_source_quote():
    quote = "find . | grep test | head -20"
    result = match_excerpts(quote, [source(1,quote)])
    assert result["status"] == "whole_exact"
    assert len(result["fragments"]) == 1


def test_overlapping_separator_and_literal_compiler_pipe():
    sources = [source(1,"warning\n    |\ncode"), source(2,"ok\ntest passed")]
    for quote in ["warning\n    | […] ok\ntest passed", "warning\n    | | ok\ntest passed"]:
        result = match_excerpts(quote,sources)
        assert result["status"] == "joined_exact_fragments"
        assert [p["message_index"] for p in result["fragments"]] == [1,2]


def test_separator_only_is_not_invented_evidence():
    result = match_excerpts(" |  | ", [source(1,"evidence")])
    assert result == dict(status="empty_separator_only",fragments=[])


def test_poll_link_only_when_prior_process_pending():
    first=dict(tool="execute_bash",action_index=2,extracted_command="cargo build",arguments='{"command":"cargo build"}',observation="The command has no new output; may wait longer")
    poll=dict(tool="execute_bash",action_index=4,extracted_command=None,arguments='{"command":""}',observation="completed")
    assert poll_origins([first,poll]) == {4:2}
    assert poll_origins([dict(first,observation="completed"),poll]) == {}
