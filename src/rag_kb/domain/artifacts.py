"""受控本地产物的领域契约。"""

from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict


class ArtifactType(StrEnum):
    REPORT = "report"
    WORK_ITEM_DRAFT = "work_item_draft"
    CHECKLIST = "checklist"
    NOTIFICATION_DRAFT = "notification_draft"


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    task_id: str
    tenant_id: str
    artifact_type: ArtifactType
    storage_key: str
    content_hash: str
    status: str
    created_at: AwareDatetime
