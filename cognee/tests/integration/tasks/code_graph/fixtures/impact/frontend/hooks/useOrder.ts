import { useState, useEffect } from "react";
export function useOrder() { const [order, setOrder] = useState({name: "demo"}); useEffect(() => { fetch("/orders").then(r => r.json()).then(setOrder); }, []); return order; }
