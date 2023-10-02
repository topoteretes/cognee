import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from sqlalchemy.exc import OperationalError
from time import sleep
import sys
from dotenv import load_dotenv
load_dotenv()

# this is needed to import classes from other modules
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of your script and add it to sys.path
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)


# in seconds
MAX_RETRIES = 3
RETRY_DELAY = 5

username = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
database_name = os.getenv('POSTGRES_DB')
host = os.getenv('POSTGRES_HOST')



SQLALCHEMY_DATABASE_URL = f"postgresql://{username}:{password}@{host}:5432/{database_name}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_recycle=3600,  # recycle connections after 1 hour
    pool_pre_ping=True  # test the connection for liveness upon each checkout
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()     

def safe_db_operation(db_op, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        with get_db() as db:
            try:
                return db_op(db, *args, **kwargs)
            except OperationalError as e:
                db.rollback()
                if "server closed the connection unexpectedly" in str(e) and attempt < MAX_RETRIES - 1:
                    sleep(RETRY_DELAY)
                else:
                    raise