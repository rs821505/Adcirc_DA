import inspect
from dataclasses import dataclass, field
from pprint import pprint


@dataclass(frozen=True, order=True)
class Comment:
    id: int
    text: str


def main():
    comment = Comment(1, "test 1")

    print(comment)

    pprint(inspect.getmembers(Comment, inspect.isfunction))


if __name__ == "__main__":
    main()
