from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status

from app.models.user import UserDB
from app.api.deps import get_current_user_optional, get_current_user
from app.services.analysis_service import analysis_service
from app.schemas.analysis import AnalysisSummaryResponse, AnalysisResponse

router = APIRouter(prefix="/history", tags=["Analysis History"])


@router.get("", response_model=List[AnalysisSummaryResponse])
async def get_history_list(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None),
    predicted_class: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    sort: str = Query("desc", pattern="^(asc|desc)$"),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """
    Retrieve user's historical MRI analyses with filtering, search, and pagination.
    Returns empty list for unauthenticated requests to prevent public data exposure.
    """
    user_id = str(current_user.id) if current_user else "guest_researcher"
    items, total = await analysis_service.get_user_history(
        user_id=user_id,
        page=page,
        page_size=page_size,
        status=status,
        predicted_class=predicted_class,
        search=search,
        sort_order=sort
    )
    return items


@router.get("/stats")
async def get_history_stats(
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """Retrieve statistical breakdown of user's past analyses."""
    if not current_user:
        return {"total_analyses": 0, "class_breakdown": {}, "user_id": None}

    user_id = str(current_user.id)
    items, total = await analysis_service.get_user_history(
        user_id=user_id,
        page=1,
        page_size=1000
    )

    class_counts = {}
    for item in items:
        cls = item.predictedClass or "Unclassified"
        class_counts[cls] = class_counts.get(cls, 0) + 1

    return {
        "total_analyses": total,
        "class_breakdown": class_counts,
        "user_id": user_id
    }


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_history_item(
    analysis_id: str,
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """Retrieve single historical analysis by ID."""
    user_id = str(current_user.id) if current_user else "anonymous"
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
        is_admin=(current_user.role == "admin" if current_user else False)
    )
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session '{analysis_id}' was not found in user history."
        )
    return analysis


@router.delete("/{analysis_id}")
async def delete_history_item(
    analysis_id: str,
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """Delete an analysis session from history and remove associated storage."""
    user_id = str(current_user.id) if current_user else "guest_researcher"
    await analysis_service.delete_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
        is_admin=(current_user.role == "admin" if current_user else False)
    )
    return {"message": f"Analysis '{analysis_id}' successfully deleted from history."}
