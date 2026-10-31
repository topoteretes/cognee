"""Pricing helpers.

Call chain (asserted by the e2e test):
    apply_discount -> compute_total -> line_total
"""

import math

TAX_RATE = 0.2


def line_total(unit_price, quantity):
    return unit_price * quantity


def compute_total(lines):
    subtotal = sum(line_total(price, qty) for price, qty in lines)
    return math.floor(subtotal * (1 + TAX_RATE) * 100) / 100


def apply_discount(lines, percent):
    total = compute_total(lines)
    return total * (1 - percent / 100)
