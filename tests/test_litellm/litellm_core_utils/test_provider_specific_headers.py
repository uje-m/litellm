import pytest

from litellm.litellm_core_utils.get_provider_specific_headers import (
    ProviderSpecificHeaderUtils,
)
from litellm.types.utils import ProviderSpecificHeader


class TestProviderSpecificHeaderUtils:
    def test_get_provider_specific_headers_matching_provider(self):
        """Test that the method returns extra_headers when custom_llm_provider matches."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "openai",
            "extra_headers": {
                "Authorization": "Bearer token123",
                "Custom-Header": "value",
            },
        }
        custom_llm_provider = "openai"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )

        expected = {"Authorization": "Bearer token123", "Custom-Header": "value"}
        assert result == expected

    def test_get_provider_specific_headers_no_match_or_none(self):
        """Test that the method returns empty dict when provider doesn't match or is None."""
        # Test case 1: Provider doesn't match
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic",
            "extra_headers": {"Authorization": "Bearer token123"},
        }
        custom_llm_provider = "openai"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )
        assert result == {}

        # Test case 2: provider_specific_header is None
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            None, "openai"
        )
        assert result == {}

    def test_get_provider_specific_headers_multi_provider_anthropic_to_bedrock(self):
        """Test that anthropic headers work with bedrock provider (multi-provider support)."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic,bedrock,bedrock_converse,vertex_ai",
            "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"},
        }

        # Test bedrock provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "bedrock"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

        # Test anthropic provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "anthropic"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

        # Test bedrock_converse provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "bedrock_converse"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

        # Test vertex_ai provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "vertex_ai"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

    def test_get_provider_specific_headers_multi_provider_no_match(self):
        """Test that non-listed providers return empty dict with multi-provider list."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic,bedrock,vertex_ai",
            "extra_headers": {"anthropic-beta": "test"},
        }

        # Test provider not in list
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "openai"
        )
        assert result == {}

    def test_get_provider_specific_headers_with_spaces(self):
        """Test that comma-separated list with spaces is handled correctly."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic, bedrock, vertex_ai",
            "extra_headers": {"anthropic-beta": "test"},
        }

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "bedrock"
        )
        assert result == {"anthropic-beta": "test"}

    def test_get_provider_specific_headers_none_custom_llm_provider(self):
        """Test that None custom_llm_provider returns empty dict."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic",
            "extra_headers": {"anthropic-beta": "test"},
        }

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, None
        )
        assert result == {}


class TestMergeHeaders:
    """Tests for the merge_headers functionality that properly combines anthropic-beta headers."""

    def test_merge_headers_basic_no_overlap(self):
        """Test basic merge without overlapping keys."""
        existing = {"x-api-key": "key1"}
        new = {"x-custom": "value"}
        ProviderSpecificHeaderUtils.merge_headers(existing, new)
        assert existing == {"x-api-key": "key1", "x-custom": "value"}

    def test_merge_headers_non_anthropic_beta_overwrites(self):
        """Test that non-anthropic-beta headers still overwrite."""
        existing = {"x-api-version": "v1"}
        new = {"x-api-version": "v2"}
        ProviderSpecificHeaderUtils.merge_headers(existing, new)
        assert existing["x-api-version"] == "v2"

    def test_merge_headers_anthropic_beta_combines_unique_values(self):
        """Test that anthropic-beta headers are merged with unique values."""
        existing = {"anthropic-beta": "web-search-2025-03-05,interleaved-thinking-2025-05-14"}
        new = {"anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14"}
        ProviderSpecificHeaderUtils.merge_headers(existing, new)

        # Should have all 3 unique values (interleaved-thinking is deduplicated)
        parts = set(existing["anthropic-beta"].split(","))
        expected = {
            "web-search-2025-03-05",
            "interleaved-thinking-2025-05-14",
            "claude-code-20250219",
        }
        assert parts == expected

    def test_merge_headers_model_config_preserved_with_client_headers(self):
        """
        Test the real-world scenario: model config extra_headers are preserved
        when client sends anthropic-beta.

        This is the core bug fix - when LiteLLM model config has:
          extra_headers: {"anthropic-beta": "web-search-2025-03-05"}
        And client sends:
          anthropic-beta: "claude-code-20250219"

        The merged result should include BOTH values.
        """
        # Model config extra_headers (has web-search for Vertex AI support)
        existing = {"anthropic-beta": "context-1m-2025-08-07,web-search-2025-03-05"}
        # Client-sent headers (Claude Code SDK sends these)
        new = {"anthropic-beta": "claude-code-20250219,fine-grained-tool-streaming-2025-05-14"}

        ProviderSpecificHeaderUtils.merge_headers(existing, new)

        parts = set(existing["anthropic-beta"].split(","))
        expected = {
            "context-1m-2025-08-07",
            "web-search-2025-03-05",
            "claude-code-20250219",
            "fine-grained-tool-streaming-2025-05-14",
        }
        assert parts == expected

    def test_merge_headers_anthropic_beta_new_value_when_not_existing(self):
        """Test that anthropic-beta is added when not in existing headers."""
        existing = {"x-api-key": "key1"}
        new = {"anthropic-beta": "web-search-2025-03-05"}
        ProviderSpecificHeaderUtils.merge_headers(existing, new)
        assert existing == {
            "x-api-key": "key1",
            "anthropic-beta": "web-search-2025-03-05",
        }

    def test_merge_headers_handles_spaces_in_comma_separated(self):
        """Test that spaces around comma-separated values are trimmed."""
        existing = {"anthropic-beta": "beta1, beta2"}
        new = {"anthropic-beta": "beta2, beta3"}
        ProviderSpecificHeaderUtils.merge_headers(existing, new)

        parts = set(existing["anthropic-beta"].split(","))
        assert parts == {"beta1", "beta2", "beta3"}
