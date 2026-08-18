import asyncio
import signal
from contextlib import suppress

from k_market_ai.core.config import get_settings
from k_market_ai.rag.infrastructure.runtime import WorkerRagRuntime


async def run() -> None:
    runtime = WorkerRagRuntime.create(get_settings())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for event in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(event, stop.set)

    await runtime.open()
    try:
        while not stop.is_set():
            processed = await runtime.handler.process_next()
            if not processed:
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=2)
    finally:
        await runtime.close()


def main() -> None:
    asyncio.run(run())
