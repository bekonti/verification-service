from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import imageRepresental

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/verifyTwoPics")
async def verify_two_pics(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        result = await imageRepresental.verify(file1, file2)

        return JSONResponse(result["verified"])

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/verifyDocWithLiveness")
async def verify_doc_with_liveness(image: UploadFile = File(...), video: UploadFile = File(...)):
    try:
        result = await imageRepresental.verify_doc_with_liveness(image, video)

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/getRepresent")
async def get_represent(file1: UploadFile = File(...)):
    try:
        result = await imageRepresental.get_represent(file1)

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Exception while processing images: {str(e)}")
