"""Library-wide NL search route -- thin wrapper around llm.library_qa.query_library."""

from fastapi import APIRouter, Depends

from app.api.deps import get_db
from app.llm.library_qa import query_library
from app.schemas.search import SearchRequest, SearchResponse
from app.services.llm import get_llm_client_or_503

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search_library(body: SearchRequest, db=Depends(get_db)):
    client, config = get_llm_client_or_503()
    result = query_library(client, config["model"], body.query, db)
    return SearchResponse(answer=result["answer"], matched_video_ids=result["matched_video_ids"])
