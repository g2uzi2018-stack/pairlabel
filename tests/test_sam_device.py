import sys,types
import pytest
from app.segmentation import Segmenter,select_device

@pytest.mark.parametrize('available,expected',[(False,'cpu'),(True,'cuda')])
def test_auto_device_uses_cuda_only_when_available(available,expected):
 torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:available))
 assert select_device('auto',torch)==expected

def test_cpu_selection_does_not_need_a_cuda_runtime():
 assert select_device('cpu',types.SimpleNamespace())=='cpu'
 assert select_device('auto',types.SimpleNamespace())=='cpu'

def test_explicit_cuda_never_silently_runs_on_cpu():
 with pytest.raises(ValueError,match='请求 CUDA'):select_device('cuda',types.SimpleNamespace())
 with pytest.raises(ValueError,match='必须是'):select_device('other',types.SimpleNamespace())

def test_unload_releases_cuda_cache_once_and_clears_model_references(monkeypatch):
 calls=[]
 monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda:calls.append('empty'))))
 segmenter=Segmenter();segmenter.device='cuda';segmenter.predictor=object();segmenter.image_ref=object();segmenter.signature='weights:cuda_fp32'
 segmenter.unload();segmenter.unload()
 assert calls==['empty']
 assert segmenter.predictor is None and segmenter.image_ref is None and segmenter.signature is None and segmenter.device is None
