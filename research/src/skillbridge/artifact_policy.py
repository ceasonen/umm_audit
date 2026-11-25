"""Enforce the workspace prohibition on project artifact hashing.

Legacy callers fail before creating an algorithm object. This guard does not
modify Python's hashlib module or third-party/network protocol implementations.
Use explicit manifests, record comparison and numeric checks for this project.
"""


class _ForbiddenHashing:
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise RuntimeError(
            f"Project artifact hashing is disabled by workspace policy: {name}. "
            "Use explicit records and direct comparisons instead."
        )


forbidden_hashing = _ForbiddenHashing()
