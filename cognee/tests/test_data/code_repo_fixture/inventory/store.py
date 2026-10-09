"""In-memory inventory store that depends on pricing."""

import requests

from inventory.pricing import apply_discount, compute_total


class InventoryStore:
    def __init__(self):
        self.lines = []

    def add_line(self, unit_price, quantity):
        self.lines.append((unit_price, quantity))

    def total(self):
        return compute_total(self.lines)

    def discounted_total(self, percent):
        return apply_discount(self.lines, percent)

    def sync(self, url):
        return requests.post(url, json={"total": self.total()})
