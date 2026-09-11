from pydantic import BaseModel, ConfigDict, Field


class DocumentUploadMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    department: str = Field(min_length=1, max_length=200)
