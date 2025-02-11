import abc
from typing import Any, Callable, Generic, Optional, TypeVar, Union, cast, overload

KeyT = TypeVar("KeyT")
ValT = TypeVar("ValT")

T = TypeVar("T")


class FlavorManager(Generic[KeyT, ValT]):
    """
    Manages "flavoring" of classes, typically in order to assign implementations of some base class to different
    "keys" (aka flavors), and then to use these implementations without calling them explicitly.

    The flavors must be hashable (to serve as a dict's keys).

    for example, suppose you have these classes:

    ```
    class Base:
        def foo(bar: str):
            ...

    class Hello(Base):
        def foo(bar: str):
            return f"hello {bar}"

    class Hi(Base):
        def foo(bar: str):
            return f"hi {bar}"
    ```

    You want to create an instance according to a string parameter - 'hello' for `Hello`, 'hi' for `Hi`.

    Naive way to do it is with a switch-case:
    ```

    from base import Base, Hello, Hi

    def base_factory(flavor: str) -> Base:
        if flavor == 'hello':
            return Hello()
        elif flavor == 'hi':
            return Hi()
        raise ValueError(f"unknown flavor {flavor}")

    def main():
        a: Base = base_factory("hello")
        print(a.foo("a"))

        b: Base = base_factory("hi")
        print(b.foo("b"))
    ```

    The problem with that approch is not only it is cumbersome (each new implementation needs to be added to the if-else),
    but also this couples the implementations to the user (because you need to directly import it and call it)

    This is where FlavorManager comes to the rescue. you can create a manager and add each class with it's flavor:
    ```
    from typing import Type

    from agent_shared.flavor_manager import FlavorManager

    class Base:
        def foo(bar: str):
            ...

    base_flavor_manager: FlavorManager[str, Type[Base]] = FlavorManager()

    @base_flavor_manager.flavor("hello")
    class Hello(Base):
        def foo(bar: str):
            return f"hello {bar}"

    @base_flavor_manager.flavor("hi")
    class Hi(Base):
        def foo(bar: str):
            return f"hi {bar}"
    ```
    now you can implement your code in this way:
    ```

    from base import Base, base_flavor_manager

    def base_factory(flavor: str) -> Base:
        return base_flavor_manager[flavor]()

    def main():
        a: Base = base_factory("hello")
        print(a.foo("a"))

        b: Base = base_factory("hi")
        print(b.foo("b"))
    ```

    Notice that you don't refer to the implementation classes in this code - hence decoupling has been achieved.
    In addition, future implementations don't need to be added to your code - adding them to the flavor manager does that automatically.
    """

    def __init__(self) -> None:
        self._flavors: dict[KeyT, ValT] = {}

    def __setitem__(self, key: KeyT, value: ValT) -> None:
        self._flavors[key] = value

    def __getitem__(self, key: KeyT) -> ValT:
        return self._flavors[key]

    @overload
    def get(self, flavor: KeyT, default: T) -> Union[ValT, T]:
        ...

    @overload
    def get(self, flavor: KeyT) -> Optional[ValT]:
        ...

    def get(self, flavor: KeyT, default: Optional[T] = None) -> Optional[Union[ValT, T]]:
        """
        Returns the value of the given flavor. Similar api as dict's get()
        """
        return self._flavors.get(flavor, default)

    @overload
    def flavor(self, flavor: KeyT) -> Callable[[T], T]:
        ...

    @overload
    def flavor(self, flavor: KeyT, value: ValT) -> None:
        ...

    def flavor(self, flavor: KeyT, value: Optional[ValT] = None) -> Optional[Callable[[T], T]]:
        """
        adds a flavor to the flavor manager.
        Two use cases for this method exists: either supply the value of the flavor immediately, or supply only the flavor.
        The second form is designed to be used as a decorator for classes or functions:

        ```
        @my_manager.flavor(1337)
        class MyClass(MyBaseClass):
            ...
        ```
        """
        if value is not None:
            self[flavor] = value
            return None

        def decorator(value: T) -> T:
            self[flavor] = cast(ValT, value)
            return value

        return decorator
    
    def flavor_of(self, value: ValT) -> KeyT:
        """
        returns the key of this specific value
        """
        for key, val in self._flavors.items():
            if val == value:
                return key
        raise ValueError(f"Value {value} not found in flavors")
    
class TypedFlavorManager(FlavorManager[KeyT, ValT]):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def kwargs_type_parameter_name(self) -> str:
        ...

    @overload
    def flavor(self, flavor: KeyT) -> Callable[[T], T]:
        ...

    @overload
    def flavor(self, flavor: KeyT, value: ValT) -> None:
        ...

    def flavor(self, flavor: KeyT, value: Optional[ValT] = None) -> Optional[Callable[[T], T]]:
        if value is not None:
            return super().flavor(flavor, value)

        fm_super = super()
        kwargs_type_parameter_name = self.kwargs_type_parameter_name()

        def decorator(cls: T) -> T:
            original_init = cls.__init__ # type: ignore

            def new_init(self, *args: Any, **kwargs: Any) -> None: # type: ignore
                kwargs[kwargs_type_parameter_name] = flavor
                original_init(self, *args, **kwargs)

            cls.__init__ = new_init  # type: ignore
            return fm_super.flavor(flavor)(cls) # type: ignore

        return decorator
    