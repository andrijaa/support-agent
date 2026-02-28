from typing import Optional
from pydantic import BaseModel


class SignalMessage(BaseModel):
    type: str
    room: Optional[str] = None
    client_id: Optional[str] = None
    sdp: Optional[str] = None
    candidate: Optional[str] = None
    sdp_mid: Optional[str] = None
    sdp_mline_index: Optional[int] = None
