from typing import Dict, Optional

from litellm.types.utils import ProviderSpecificHeader

# Headers that should be merged with comma-separated values instead of overwritten
COMMA_MERGED_HEADERS = {"anthropic-beta"}


class ProviderSpecificHeaderUtils:
    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: Optional[ProviderSpecificHeader],
        custom_llm_provider: Optional[str],
    ) -> Dict:
        """
        Get the provider specific headers for the given custom llm provider.

        Supports comma-separated provider lists for headers that work across multiple providers.

        Returns:
            Dict: The provider specific headers for the given custom llm provider
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return {}

        stored_providers = provider_specific_header.get("custom_llm_provider", "")
        provider_list = [p.strip() for p in stored_providers.split(",")]

        if custom_llm_provider in provider_list:
            return provider_specific_header.get("extra_headers", {})

        return {}

    @staticmethod
    def merge_headers(existing_headers: Dict, new_headers: Dict) -> None:
        """
        Merge new_headers into existing_headers in-place.

        For headers in COMMA_MERGED_HEADERS (like 'anthropic-beta'), values are merged
        as comma-separated unique values instead of being overwritten.

        This ensures that model config extra_headers (e.g., 'web-search-2025-03-05') are
        preserved when client-sent headers (e.g., 'claude-code-20250219') are applied.

        Args:
            existing_headers: The target dict to merge into (modified in-place)
            new_headers: The source dict with headers to merge
        """
        for key, value in new_headers.items():
            if key in COMMA_MERGED_HEADERS and key in existing_headers:
                # Merge comma-separated values, keeping unique values only
                existing_value = existing_headers[key]
                existing_parts = {
                    part.strip()
                    for part in str(existing_value).split(",")
                    if part.strip()
                }
                new_parts = {
                    part.strip() for part in str(value).split(",") if part.strip()
                }
                # Combine all unique parts
                all_parts = existing_parts | new_parts
                existing_headers[key] = ",".join(sorted(all_parts))
            else:
                # Regular overwrite for non-merged headers
                existing_headers[key] = value
