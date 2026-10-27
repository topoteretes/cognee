import { useOrder } from "../hooks/useOrder";
export function OrderCard() { const order = useOrder(); return <div>{order.name}</div>; }
