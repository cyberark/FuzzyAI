
import asyncio
import logging
from typing import Any, Callable, Coroutine, Generic, TypeVar, Union

T = TypeVar("T")
logger = logging.getLogger(__name__)

class AsyncBatch(Generic[T]):

    class LastItem: pass

    class FlushMarker:
        def __init__(self) -> None:
            self.event = asyncio.Event()

    def __init__(self, drain: Callable[[list[T]], Coroutine[Any, Any, bool]], batch_size: int) -> None:
        super().__init__()

        self.drain = drain
        self.batch_size = batch_size

        self.items_q: asyncio.Queue[Union[T, AsyncBatch.LastItem, AsyncBatch.FlushMarker]] = asyncio.Queue()

        self.handler_task = asyncio.create_task(self._handler_task())
        self.done: bool = False
    
    async def stop(self) -> None:
        logger.debug(f"Stopping batch, {self.items_q.qsize()} items left in queue.")
        self.done = True # no new items will be queued after this
        self.items_q.put_nowait(AsyncBatch.LastItem()) # puts last item - will make handler loop finish
        await self.handler_task

    def put(self, item: T) -> None:
        if not self.done:
            self.items_q.put_nowait(item)

    async def flush(self) -> None:
        if self.done: # if the batch was stopped, then it will be flushed automatically when the handler task completes
            await self.handler_task
            return
        marker = AsyncBatch.FlushMarker()
        self.items_q.put_nowait(marker)
        await marker.event.wait()
    
    async def _handler_task(self) -> None:
        batch: list[T] = []

        while True:
            item = await self.items_q.get()
            
            if isinstance(item, AsyncBatch.LastItem): # LastItem signals end of processing, flush and finish task
                if batch:
                    await self.drain(batch)
                return

            if isinstance(item, AsyncBatch.FlushMarker): # FlushMarker signals explicit flushing request, flush and continue to next item
                if batch:
                    await self.drain(batch)
                    batch = []

                item.event.set() # notify flusher that flush is done
                continue

            # append item to batch
            batch.append(item)
            logger.debug("Batch size: %s", len(batch))
            
            # if batch size is big enough to be sent, send it and reset batch and timeout
            if len(batch) >= self.batch_size:
                if await self.drain(batch):
                    batch = []
                continue