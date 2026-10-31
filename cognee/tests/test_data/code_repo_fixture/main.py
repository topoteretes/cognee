"""Entry point of the fixture project."""

from inventory.store import InventoryStore


def main():
    store = InventoryStore()
    store.add_line(10.0, 2)
    store.add_line(4.5, 4)
    print(store.discounted_total(10))


if __name__ == "__main__":
    main()
