from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from livestream import LiveStream


app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

stream: LiveStream

@app.get("/", response_class=HTMLResponse)
async def get_index(
    request: Request
):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def post_index(
    request: Request,
    camera: str = Form(...),
    ip_address_input: str = Form(None)
):
    global stream

    if camera == "webcam":
        stream = LiveStream(web=True, cam="web")
    elif camera == "phone" and ip_address_input:
        stream = LiveStream(web=True, cam="phone", ip=ip_address_input)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "camera": camera,
            "ip_address": ip_address_input
        }
    )

@app.get("/video")
async def video():
    if stream is None:
        return {"error": "Stream is not initialized"}
    return StreamingResponse(stream.generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=5000)
