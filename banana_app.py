import base64
import time
from datetime import datetime

from potassium import Potassium, Request, Response

from transformers import pipeline
import torch
import app as vits_app

app = Potassium("my_app")
print('start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    time1 = time.time()
    # vits_app.app.run(host='0.0.0.0', port=app.config.get("PORT", 23456), debug=app.config.get("DEBUG", False))  # 对外开放
    context = {
        "model": vits_app
    }
    time2 = time.time()
    print("init time:", time2 - time1)
    return context
    # vits_app.app.run(host='0.0.0.0', port=vits_app.app.config.get("PORT", 23456), debug=vits_app.app.config.get("DEBUG", False))
    # device = 0 if torch.cuda.is_available() else -1
    # model = pipeline('fill-mask', model='bert-base-uncased', device=device)
    #
    # context = {
    #     "model": model
    # }
    #
    # return context


# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    time1 = time.time()
    text = request.json.get("text")
    speaker_id = request.json.get("id")
    task = {"text": text,
            "id": int(speaker_id),
            "format": 'wav',
            "length": 1.0,
            "noise": 0.6,
            "noisew": 0.8,
            "sdp_ratio": 0.2,
            "max": 50,
            "lang": 'auto',
            "speaker_lang": 'zh'}

    audio = vits_app.tts.bert_vits2_infer(task)
    # with open("output.wav", "wb") as f:
    #     f.write(audio.getbuffer())
    file_byte = audio.getvalue()
    file_base64 = base64.b64encode(file_byte).decode("utf-8")
    # cuda_status = torch.cuda.is_available()
    # print("cuda_status:", cuda_status)
    time2 = time.time()
    print("handler time:", time2 - time1)

    return Response(
        json={"outputs": "success", "file": file_base64},
        status=200
    )


if __name__ == "__main__":
    app.serve()
