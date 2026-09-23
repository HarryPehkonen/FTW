"""The canonical token counter (ftw_plan.md §3.3, §3.1 skill size cap).

Deliberately not tied to any one model's BPE: budgets and the 1,500-token
skill-body cap must mean the same thing regardless of which provider is
active. This is a documented, deterministic approximation, not an attempt
at exactness against any specific tokenizer.
"""

from ftw.tokens import count_tokens


class TestCountTokens:
    def test_empty_string_is_zero_tokens(self):
        assert count_tokens("") == 0

    def test_is_deterministic(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert count_tokens(text) == count_tokens(text)

    def test_counts_words_separately(self):
        assert count_tokens("hello world") == 2

    def test_punctuation_counts_as_its_own_token(self):
        assert count_tokens("hello, world!") > count_tokens("hello world")

    def test_whitespace_only_is_zero_tokens(self):
        assert count_tokens("   \n\t  ") == 0

    def test_longer_text_counts_more(self):
        short = "one two three"
        long = short * 10
        assert count_tokens(long) > count_tokens(short)

    def test_non_ascii_text_counts_without_crashing(self):
        assert count_tokens("héllo wörld — emdash test 你好") > 0
