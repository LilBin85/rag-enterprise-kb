"""身份提供器适配器。"""

from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider, IdentityProvider

__all__ = ["DemoTokenIdentityProvider", "IdentityProvider"]
