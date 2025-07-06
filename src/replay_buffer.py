import random
from typing import Any, List, Tuple

class ReplayBuffer:
    """Fixed-size FIFO buffer storing (state, action, reward, next_state, done) tuples."""

    def __init__(self, capacity: int):
        assert capacity > 0, "Replay buffer capacity must be positive."
        self.capacity = capacity
        self._buffer: List[Tuple[Any, ...]] = []
        self._pos: int = 0

    # -------------------------------------------------------------------------
    def push(self, *transition: Any) -> None:
        """Add a transition to the buffer (overwrites oldest when full)."""
        if len(self._buffer) < self.capacity:
            self._buffer.append(None)  # type: ignore
        self._buffer[self._pos] = transition  # type: ignore
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        assert batch_size > 0, "batch_size must be > 0"
        assert len(self._buffer) >= batch_size, (
            f"Not enough elements in buffer to sample {batch_size}, current size {len(self._buffer)}"
        )
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer) 