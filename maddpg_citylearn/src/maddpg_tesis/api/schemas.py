from typing import List

from pydantic import BaseModel, Field


class AgentObservation(BaseModel):
    obs: List[float] = Field(..., description="Observación de un agente")


class MADDPGRequest(BaseModel):
    observations: List[AgentObservation] = Field(
        ..., description="Observaciones de todos los agentes"
    )


class MADDPGResponse(BaseModel):
    actions: List[List[float]] = Field(
        ..., description="Acciones continuas para cada agente"
    )
