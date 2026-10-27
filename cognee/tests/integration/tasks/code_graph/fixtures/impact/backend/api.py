from fastapi import FastAPI
from models import Order

app = FastAPI()


@app.get("/orders")
def list_orders():
    return load_orders()


def load_orders():
    return Order.query.all()
