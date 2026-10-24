import os,tempfile
os.environ['LABELER_DATA']=tempfile.mkdtemp(prefix='xiangqi-tests-')
import pytest
from app import db
from app.config import DATA
@pytest.fixture(autouse=True)
def clean_db():
 db.init()
 with db.connection() as c:
  for t in ['runs','audit','items','jobs']:c.execute(f'DELETE FROM {t}')
 yield
