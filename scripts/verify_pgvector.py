from pgvector.sqlalchemy import Vector
import pgvector
import importlib.metadata

print("pgvector version:", importlib.metadata.version("pgvector"))

from sqlalchemy import Column, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class T(Base):
    __tablename__ = "t"
    id = Column(Text, primary_key=True)
    embedding = Column(Vector(1536))


col = T.embedding
methods = [
    m
    for m in dir(col)
    if any(k in m.lower() for k in ("dist", "cosine", "l2", "inner", "neighbor"))
]
print("Vector column distance methods:", methods)

# Confirm the expression compiles
import sqlalchemy as sa

fake_vec = [0.0] * 1536
expr = col.cosine_distance(fake_vec)
print("cosine_distance expression type:", type(expr))
print("compiled:", str(expr.compile(compile_kwargs={"literal_binds": False})))
