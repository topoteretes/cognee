from io import BytesIO

from langchain.document_loaders import PyPDFLoader

from level_2_pdf_vectorstore__dlt_contracts import Memory
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import re
import json
import logging
import os
import uvicorn
from fastapi import Request
import yaml
from fastapi import HTTPException
from fastapi import FastAPI, UploadFile, File
from typing import List
import requests
# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(debug=True)


from fastapi import Depends


class ImageResponse(BaseModel):
    success: bool
    message: str




@app.get("/", )
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Hello, World, I am alive!"}

@app.get("/health")
def health_check():
    """
    Health check endpoint that returns the server status.
    """
    return {"status": "OK"}






#curl -X POST -H "Content-Type: application/json" -d '{"data": "YourPayload"}' -F "files=@/path/to/your/pdf/file.pdf" http://127.0.0.1:8000/upload/


class Payload(BaseModel):
    payload: Dict[str, Any]

# @app.post("/upload/", response_model=dict)
# async def upload_pdf_and_payload(
#         payload: Payload,
#         # files: List[UploadFile] = File(...),
# ):
#     try:
#         # Process the payload
#         decoded_payload = payload.payload
#     # except:
#     #     pass
#     #
#     # return JSONResponse(content={"response": decoded_payload}, status_code=200)
#
#         # Download the remote PDF if URL is provided
#         if 'pdf_url' in decoded_payload:
#             pdf_response = requests.get(decoded_payload['pdf_url'])
#             pdf_content = pdf_response.content
#
#             logging.info("Downloaded PDF from URL")
#
#             # Create an in-memory file-like object for the PDF content
#             pdf_stream = BytesIO(pdf_content)
#
#             contents = pdf_stream.read()
#
#             tmp_location = os.path.join('/tmp', "tmp.pdf")
#             with open(tmp_location, 'wb') as tmp_file:
#                 tmp_file.write(contents)
#
#             logging.info("Wrote PDF from URL")
#
#             # Process the PDF using PyPDFLoader
#             loader = PyPDFLoader(tmp_location)
#             pages = loader.load_and_split()
#             logging.info(" PDF split into pages")
#             Memory_ = Memory(index_name="my-agent", user_id='555' )
#             await Memory_.async_init()
#             Memory_._add_episodic_memory(user_input="I want to get a schema for my data", content =pages)
#
#
#             # Run the buffer
#             response = Memory_._run_buffer(user_input="I want to get a schema for my data")
#             return JSONResponse(content={"response": response}, status_code=200)
#
#             #to do: add the user id to the payload
#             #to do add the raw pdf to payload
#             # bb = await Memory_._run_buffer(user_input=decoded_payload['prompt'])
#             # print(bb)
#
#
#     except Exception as e:
#
#         return {"error": str(e)}
#             # Here you can perform your processing on the PDF contents
#             # results.append({"filename": file.filename, "size": len(contents)})
#
#             # Append the in-memory file to the files list
#             # files.append(UploadFile(pdf_stream, filename="downloaded.pdf"))
#


def memory_factory(memory_type):
    load_dotenv()
    class Payload(BaseModel):
        payload: Dict[str, Any]
    @app.post("/{memory_type}/add-memory", response_model=dict)
    async def add_memory(
            payload: Payload,
            # files: List[UploadFile] = File(...),
    ):
        try:

            logging.info(" Init PDF processing")


            decoded_payload = payload.payload

            if 'pdf_url' in decoded_payload:
                pdf_response = requests.get(decoded_payload['pdf_url'])
                pdf_content = pdf_response.content

                logging.info("Downloaded PDF from URL")

                # Create an in-memory file-like object for the PDF content
                pdf_stream = BytesIO(pdf_content)

                contents = pdf_stream.read()

                tmp_location = os.path.join('/tmp', "tmp.pdf")
                with open(tmp_location, 'wb') as tmp_file:
                    tmp_file.write(contents)

                logging.info("Wrote PDF from URL")

                # Process the PDF using PyPDFLoader
                loader = PyPDFLoader(tmp_location)
                # pages = loader.load_and_split()
                logging.info(" PDF split into pages")

                Memory_ = Memory(user_id=decoded_payload['user_id'])

                await Memory_.async_init()

                memory_class = getattr(Memory_, f"_add_{memory_type}_memory", None)
                output= await memory_class(observation=str(loader), params =decoded_payload['params'])
                return JSONResponse(content={"response": output}, status_code=200)

        except Exception as e:

            return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)

    @app.post("/{memory_type}/fetch-memory", response_model=dict)
    async def fetch_memory(
            payload: Payload,
            # files: List[UploadFile] = File(...),
    ):
        try:

            decoded_payload = payload.payload

            Memory_ = Memory(user_id=decoded_payload['user_id'])

            await Memory_.async_init()

            memory_class = getattr(Memory_, f"_fetch_{memory_type}_memory", None)
            output = memory_class(observation=decoded_payload['prompt'])
            return JSONResponse(content={"response": output}, status_code=200)

        except Exception as e:

            return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)

    @app.post("/{memory_type}/delete-memory", response_model=dict)
    async def delete_memory(
            payload: Payload,
            # files: List[UploadFile] = File(...),
    ):
        try:

            decoded_payload = payload.payload

            Memory_ = Memory(user_id=decoded_payload['user_id'])

            await Memory_.async_init()

            memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
            output = memory_class(observation=decoded_payload['prompt'])
            return JSONResponse(content={"response": output}, status_code=200)

        except Exception as e:

            return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)

memory_list = ["episodic", "buffer", "semantic"]
for memory_type in memory_list:
    memory_factory(memory_type)



@app.get("/available-buffer-actions", response_model=dict)
async def available_buffer_actions(
        payload: Payload,
        # files: List[UploadFile] = File(...),
):
    try:

        decoded_payload = payload.payload

        Memory_ = Memory(user_id=decoded_payload['user_id'])

        await Memory_.async_init()

        # memory_class = getattr(Memory_, f"_delete_{memory_type}_memory", None)
        output = Memory_._available_operations()
        return JSONResponse(content={"response": output}, status_code=200)

    except Exception as e:

        return JSONResponse(content={"response": {"error": str(e)}}, status_code=503)


#
    #     # Process each uploaded PDF file
    #     results = []
    #     for file in files:
    #         contents = await file.read()
    #         tmp_location = os.path.join('/tmp', "tmp.pdf")
    #         with open(tmp_location, 'wb') as tmp_file:
    #             tmp_file.write(contents)
    #         loader = PyPDFLoader(tmp_location)
    #         pages = loader.load_and_split()
    #
    #         stm = ShortTermMemory(user_id=decoded_payload['user_id'])
    #         stm.episodic_buffer.main_buffer(prompt=decoded_payload['prompt'], pages=pages)
    #         # Here you can perform your processing on the PDF contents
    #         results.append({"filename": file.filename, "size": len(contents)})
    #
    #     return {"message": "Upload successful", "results": results}
    #
    # except Exception as e:
    #     return {"error": str(e)}


# @app.post("/clear-cache", response_model=dict)
# async def clear_cache(request_data: Payload) -> dict:
#     """
#     Endpoint to clear the cache.
#
#     Parameters:
#     request_data (Payload): The request data containing the user and session IDs.
#
#     Returns:
#     dict: A dictionary with a message indicating the cache was cleared.
#     """
#     json_payload = request_data.payload
#     agent = Agent()
#     agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
#     try:
#         agent.clear_cache()
#         return JSONResponse(content={"response": "Cache cleared"}, status_code=200)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
# @app.post("/correct-prompt-grammar", response_model=dict)
# async def prompt_to_correct_grammar(request_data: Payload) -> dict:
#     json_payload = request_data.payload
#     agent = Agent()
#     agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
#     logging.info("Correcting grammar %s", json_payload["prompt_source"])
#
#     output = agent.prompt_correction(json_payload["prompt_source"], model_speed= json_payload["model_speed"])
#     return JSONResponse(content={"response": {"result": json.loads(output)}})


# @app.post("/action-add-zapier-calendar-action", response_model=dict,dependencies=[Depends(auth)])
# async def action_add_zapier_calendar_action(
#     request: Request, request_data: Payload
# ) -> dict:
#     json_payload = request_data.payload
#     agent = Agent()
#     agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
#     # Extract the bearer token from the header
#     auth_header = request.headers.get("Authorization")
#     if auth_header:
#         bearer_token = auth_header.replace("Bearer ", "")
#     else:
#         bearer_token = None
#     outcome = agent.add_zapier_calendar_action(
#         prompt_base=json_payload["prompt_base"],
#         token=bearer_token,
#         model_speed=json_payload["model_speed"],
#     )
#     return JSONResponse(content={"response": outcome})



def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the API server using uvicorn.

    Parameters:
    host (str): The host for the server.
    port (int): The port for the server.
    """
    try:
        logger.info(f"Starting server at {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        # Here you could add any cleanup code or error recovery code.


if __name__ == "__main__":
    start_api_server()
