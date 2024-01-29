from pydantic import BaseModel


class GenerateImageModel(BaseModel):
    mode: int
    category: str
    content: str
    tasks_id: str
    user_id: int
    image_url: str
    version: str


class GenerateCancelModel(BaseModel):
    tasks_id: str
