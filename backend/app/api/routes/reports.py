import io
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from app.models.user import UserDB
from app.api.deps import get_current_user_or_default
from app.services.analysis_service import analysis_service
from app.services.report_service import report_service

router = APIRouter(prefix="/analyses", tags=["Reports"])


@router.get("/{analysis_id}/report")
async def get_analysis_report_summary(
    analysis_id: str,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """Retrieve metadata and download URL for an analysis research report."""
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id=current_user.id or "guest_researcher",
        is_admin=(current_user.role == "admin")
    )
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session '{analysis_id}' was not found."
        )

    return {
        "analysis_id": analysis.id,
        "filename": analysis.fileName,
        "format": "PDF",
        "download_url": f"/api/analyses/{analysis_id}/report/download",
        "generated_at": analysis.createdAt,
        "disclaimer": "Research use only. Not for clinical diagnostic use."
    }


@router.get("/{analysis_id}/report/download")
async def download_analysis_report_pdf(
    analysis_id: str,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """Download research report as a formatted PDF document."""
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id=current_user.id or "guest_researcher",
        is_admin=(current_user.role == "admin")
    )
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session '{analysis_id}' was not found."
        )

    pdf_bytes = report_service.generate_pdf_report(analysis)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=NeuroLens_Report_{analysis_id}.pdf"
        }
    )
