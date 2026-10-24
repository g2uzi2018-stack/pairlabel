import sys,json,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import numpy as np
from app.config import DATA,load_config,ROOT,CLASSES
from app.segmentation import Segmenter
from scripts.benchmark import score
from app.pipeline import preview
c=load_config();c.update(annotation_profile='xiangqi',annotation_instructions='',classes=copy.deepcopy(CLASSES));a=json.loads((DATA/'benchmark/generator/six-pieces.json').read_text());truth=json.loads((ROOT/'samples/six-pieces.truth.json').read_text());rgb=np.asarray(Image.open(ROOT/'samples/six-pieces.png').convert('RGB'));t=time.time();objects,issues=Segmenter().segment(rgb,a['prediction']['objects'],c);result={'prediction':{'objects':objects},'issues':issues,'score':score({'objects':objects},truth),'wall_seconds':time.time()-t};(DATA/'benchmark/sam2-six.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));preview(ROOT/'samples/six-pieces.png',result['prediction'],c['classes'],DATA/'benchmark/sam2-six.jpg');print(result['score'],issues,result['wall_seconds'],flush=True)
