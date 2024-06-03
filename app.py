from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from modules.pmbl import PMBL

app = FastAPI(docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

pmbl = PMBL("./PMB-7b.Q6_K.gguf")  # Replace with the path to your model

@app.head("/")
@app.get("/")
def index() -> HTMLResponse:
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_input = data["user_input"]
        mode = data["mode"]
        history = pmbl.get_chat_history(mode, user_input)
        response_generator = pmbl.generate_response(user_input, history, mode)
        return StreamingResponse(response_generator, media_type="text/plain")
    except Exception as e:
        print(f"[SYSTEM] Error: {str(e)}")
        return {"error": str(e)}

@app.post("/sleep")
async def sleep():
    try:
        pmbl.sleep_mode()
        return {"message": "Sleep mode completed successfully"}
    except Exception as e:
        print(f"[SYSTEM] Error: {str(e)}")
        return {"error": str(e)}