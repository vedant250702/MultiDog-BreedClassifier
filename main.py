from fastapi import FastAPI
from fastapi import routing
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.routes import router
from fastapi.responses import FileResponse
import dotenv
import uvicorn
import os

dotenv.load_dotenv()

app=FastAPI()
app.include_router(router=router)
app.add_middleware(CORSMiddleware,
                    allow_origins=["*"],
                    allow_credentials=True,
                    allow_methods=["*"],  
                    allow_headers=["*"],)
app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

@app.get("/{full_path:path}")
def home():
    return FileResponse("dist/index.html") 


if __name__=="__main__":
    os.system("pip install -r requirements.txt")
    print("Running on http://localhost:5000")
    uvicorn.run(app,host="0.0.0.0",port=5000)