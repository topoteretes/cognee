# Common Extraction Schemas

## Contact Information
```json
{
  "name": "string",
  "email": "string",
  "phone": "string",
  "company": "string",
  "role": "string"
}
```

## Invoice Data
```json
{
  "invoice_number": "string",
  "date": "ISO-8601 date",
  "vendor": "string",
  "line_items": [{"description": "string", "quantity": "number", "unit_price": "number"}],
  "total": "number",
  "currency": "string"
}
```
