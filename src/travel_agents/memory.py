from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


@contextmanager
def open_checkpointer(database_url: str = "") -> Iterator[BaseCheckpointSaver]:
    """Postgres checkpoints when a URL is set (conversations survive restarts), otherwise in-memory."""
    if not database_url:
        yield InMemorySaver()
        return
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(database_url) as saver:
        saver.setup()  # idempotent: creates the checkpoint tables on first use
        yield saver
