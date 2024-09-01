from fastapi import FastAPI
from app.api.routes import api_router
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

def create_app() -> FastAPI: 
    app = FastAPI()
    
    # Incluir as rotas da API
    app.include_router(api_router)
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Utilize a variável de ambiente para a configuração de host e port se definida
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host=host, port=port, reload=True)
