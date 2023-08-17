## PromethAI Memory Manager



### Description


Initial code lets you do three operations:

1. Add to memory
2. Retrieve from memory
3. Structure the data to schema and load to duckdb

#How to use

## Installation

```docker compose build promethai_mem   ```

## Run

```docker compose up promethai_mem   ```


## Usage

The fast API endpoint accepts prompts and PDF files and returns a JSON object with the generated text.

```curl                                                                    
    -X POST                                                                                             
    -F "prompt=The quick brown fox"                                                                     
    -F "file=@/path/to/file.pdf"                                                                       
    http://localhost:8000/upload/                                                                    
```

{
  "payload": {
    "user_id": "681",
    "session_id": "471",
    "model_speed": "slow",
    "prompt": "Temperature=Cold;Food Type=Ice Cream",
    "pdf_url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
  }
}