from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Double
from datetime import datetime
from dotenv import load_dotenv
import os 

#SPIN UP POSTGRES CONTAINER BEFORE EXECUTING THIS FILE
#DATABASE SECRETS STORED IN ENV FILE, ADD VARIABLE DB_URL = "postgresql+psycopg2://USER-NAME:PASSWORD@HOSTNAME/DATA-BASE-NAME"
#TO GET THIS TO OPERATE
load_dotenv()
engine = create_engine(os.getenv("DB_URL"))

#REQUIRED CLASS FOR ORM TO FUNCTION
class Base(DeclarativeBase):
    pass

class vector(Base):
    __table__name__ = "vector"
#CLASS CORRESPONDING TO GIVEN NODE IN CLUSTER
class Label(Base):
     __table_name__ = "Labels"
     label_name: Mapped[str] = mapped_column(primary_key=True)
    
#GENERATE ORM
Base.metadata.create_all(engine)