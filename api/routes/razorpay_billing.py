from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request, Depends
from pydantic import BaseModel

from utils.monetization import (
    apply_razorpay_webhook,
    create_razorpay_subscription,
    verify_razorpay_signature,
)
from utils.rate_limit import rate_limit_dependency

router = APIRouter()


class CheckoutRequest(BaseModel):
    org_id: str
    org_name: str = "Workspace"
    user_email: str


@router.post("/checkout/razorpay", dependencies=[Depends(rate_limit_dependency)])
async def create_checkout(req: CheckoutRequest):
    """Create a hosted Razorpay subscription checkout link for an organization."""
    try:
        return create_razorpay_subscription(
            org_id=req.org_id,
            org_name=req.org_name,
            user_email=req.user_email,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Checkout creation failed: {exc}")


@router.post("/webhooks/razorpay", dependencies=[Depends(rate_limit_dependency)])
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str = Header(default=""),
):
    """Verify Razorpay webhook signature and update org subscription state."""
    body = await request.body()
    if not verify_razorpay_signature(body, x_razorpay_signature):
        raise HTTPException(status_code=401, detail="Invalid Razorpay webhook signature.")

    try:
        payload = await request.json()
        result = apply_razorpay_webhook(payload)
        return {"ok": True, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Webhook handling failed: {exc}")
