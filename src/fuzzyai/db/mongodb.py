from typing import Any, Generic, List, Optional, Type, TypeVar, Union

from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_tornado import MotorClient
from pydantic import BaseModel

from fuzzyai.consts import ID_FIELD, MONGO_OPERATOR_GT

T = TypeVar('T', bound='MongoDocument')

class MongoDocument(BaseModel):
    @classmethod
    def from_dict(cls: Type[T], **kwargs: Any) -> T:
        raise NotImplementedError("This method must be implemented by the child class.")
    
    @classmethod
    def new(cls: Type[T], *args: Any) -> T:
        raise NotImplementedError("This method must be implemented by the child class.")

class MongoDBHandler(Generic[T]):
    def __init__(self, client: Union[AsyncIOMotorClient, MotorClient], database: str, collection: str, model_type: Type[T]) -> None:
        self._collection = client[database][collection]
        self._page_size = 1000
        self._last_id: Optional[str] = None
        self._model_type: Type[T] = model_type

    async def retrieve(self) -> List[T]:
        filter: dict[str, Any] = {ID_FIELD: {MONGO_OPERATOR_GT: self._last_id}} if self._last_id else {}
        result = await self._collection.find(filter).limit(self._page_size).to_list(length=None)
        self._last_id = result[-1][ID_FIELD] if result else None
        return [self._model_type.from_dict(**raw_result) for raw_result in result]

    async def retrieve_by_property(self, key: str, value: Any) -> List[T]:
        filter = {key: value}
        result = await self._collection.find(filter).to_list(length=None)
        return [self._model_type.from_dict(**raw_result) for raw_result in result]
    
    async def retrieve_all(self) -> List[T]:
        return [self._model_type.from_dict(**x) for x in await self._collection.find().to_list(length=None)]
    
    async def store(self, items: List[BaseModel]) -> None:
        await self._collection.insert_many([item.model_dump() for item in items])

    async def store_one(self, item: BaseModel) -> None:
        await self._collection.insert_one(item.model_dump())
    
    def sync_store(self, items: List[BaseModel]) -> None:
        self._collection.insert_many([item.model_dump() for item in items])

    def sync_store_one(self, item: BaseModel) -> None:
        self._collection.insert_one(item.model_dump())