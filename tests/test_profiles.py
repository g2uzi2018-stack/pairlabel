"""Profile and prompt contracts runnable with the Python standard library."""
import copy
import json
import unittest
from unittest.mock import Mock,patch

from app import config
from app.config import DEFAULT,CLASSES,CLASS_PRESETS,validate
from app.profiles import is_xiangqi
from app.prompts import build_detection_prompt,build_crop_prompt
from app.taxonomy import normalize_taxonomy


class ProfileTests(unittest.TestCase):
 def test_general_default_and_legacy_preset_are_independent(self):
  self.assertEqual(DEFAULT['annotation_profile'],'general')
  self.assertEqual(DEFAULT['annotation_instructions'],'')
  self.assertEqual(DEFAULT['classes'],CLASS_PRESETS['general'])
  self.assertEqual([x['name'] for x in DEFAULT['classes']],['person','cup','bottle','mouse'])
  self.assertEqual(CLASS_PRESETS['xiangqi'],CLASSES)
  self.assertEqual(len(CLASSES),14)
  self.assertIsNot(DEFAULT['classes'],CLASS_PRESETS['general'])
  self.assertIsNot(DEFAULT['classes'][0],CLASS_PRESETS['general'][0])
  self.assertIsNot(CLASS_PRESETS['xiangqi'][0],CLASSES[0])

 def test_explicit_profile_overrides_metadata_in_both_directions(self):
  self.assertFalse(is_xiangqi({'annotation_profile':'general','classes':CLASSES}))
  self.assertTrue(is_xiangqi({'annotation_profile':'xiangqi','classes':CLASS_PRESETS['general']}))

 def test_legacy_requires_complete_known_metadata(self):
  self.assertTrue(is_xiangqi({'classes':CLASSES}))
  self.assertFalse(is_xiangqi({'classes':CLASS_PRESETS['general']}))
  self.assertFalse(is_xiangqi({'classes':[{'name':'red_king','label':'红帅'}]}))
  self.assertFalse(is_xiangqi({'classes':[]}))
  for key,value in [('piece',None),('color',None),('piece','cup'),('color','blue')]:
   classes=copy.deepcopy(CLASSES);classes[0][key]=value
   with self.subTest(key=key,value=value):self.assertFalse(is_xiangqi({'classes':classes}))

 def test_saved_config_migration_preserves_legacy_and_explicit_choice(self):
  for user,expected in [({'classes':CLASSES},'xiangqi'),({'classes':CLASS_PRESETS['general']},'general'),({'classes':CLASSES,'annotation_profile':'general'},'general')]:
   stored=Mock();stored.read_text.return_value=json.dumps(user)
   with self.subTest(expected=expected,user_profile=user.get('annotation_profile')):
    with patch.object(config,'CONFIG',stored):loaded=config.load_config()
    self.assertEqual(loaded['annotation_profile'],expected)
    self.assertEqual(loaded['classes'],user['classes'])
    self.assertEqual(loaded['annotation_instructions'],'')

 def test_arbitrary_classes_replace_preset_with_1_to_200_entries(self):
  for count in (1,17,200):
   c=copy.deepcopy(DEFAULT)
   c['classes']=[{'id':i,'name':f'arbitrary_{i}','label':f'自定义目标 {i}'} for i in range(count)]
   c['annotation_instructions']='只标注前景中可见的实体'
   with self.subTest(count=count):self.assertIs(validate(c),c)
  for classes in ([],[{'id':i,'name':str(i),'label':str(i)} for i in range(201)],{},[None]):
   c=copy.deepcopy(DEFAULT);c['classes']=classes
   with self.subTest(classes_type=type(classes).__name__,count=len(classes)):
    with self.assertRaises(ValueError):validate(c)

 def test_invalid_profiles_and_instructions_are_rejected(self):
  for value in ('','General','chess',None,True,[],{}):
   c=copy.deepcopy(DEFAULT);c['annotation_profile']=value
   with self.subTest(profile=value):
    with self.assertRaises(ValueError):validate(c)
  for value in (None,True,3,[],{},'字'*2001):
   c=copy.deepcopy(DEFAULT);c['annotation_instructions']=value
   with self.subTest(instructions_type=type(value).__name__):
    with self.assertRaises(ValueError):validate(c)
  c=copy.deepcopy(DEFAULT);c['annotation_instructions']='字'*2000
  self.assertIs(validate(c),c)

 def test_old_quality_and_model_validation_is_preserved(self):
  for key,value in [('iou_threshold',.09),('min_confidence',1.01),('max_attempts',0)]:
   c=copy.deepcopy(DEFAULT);c[key]=value
   with self.subTest(key=key):
    with self.assertRaises(ValueError):validate(c)
  for change in ('family','weights','image_tokens'):
   c=copy.deepcopy(DEFAULT);rt=c['runtime']
   if change=='image_tokens':rt[change]=100
   else:rt['reviewer'][change]=rt['generator'][change]
   with self.subTest(change=change):
    with self.assertRaises(ValueError):validate(c)
  legacy=copy.deepcopy(DEFAULT);legacy.pop('annotation_profile');legacy.pop('annotation_instructions');legacy['classes']=copy.deepcopy(CLASSES)
  self.assertIs(validate(legacy),legacy)
  self.assertTrue(is_xiangqi(legacy))

 def test_disabled_taxonomy_keeps_category_confidence_and_raw_evidence(self):
  prediction={'objects':[{'class_id':1,'text':'帥','color':'unknown','confidence':.98}]}
  out,changes=normalize_taxonomy(prediction,CLASSES,enabled=False)
  self.assertEqual(out,prediction);self.assertEqual(changes,[])
  self.assertIsNot(out,prediction);self.assertIsNot(out['objects'][0],prediction['objects'][0])
  legacy={'objects':[{'class_id':1,'text':'帥','confidence':.98}]}
  out,changes=normalize_taxonomy(legacy,CLASSES)
  self.assertEqual(out['objects'][0]['class_id'],0)
  self.assertEqual(out['objects'][0]['class_id_raw'],1)
  self.assertEqual(legacy['objects'][0]['class_id'],1)
  self.assertEqual(len(changes),1)


class PromptTests(unittest.TestCase):
 def test_general_prompts_have_no_builtin_chess_requirements(self):
  c=copy.deepcopy(DEFAULT)
  # Complete legacy metadata must still respect explicit general mode.
  c['classes']=[{'id':0,'name':'tool','label':'工具','piece':'king','color':'red'}]
  for prompt in [build_detection_prompt(c,0),build_detection_prompt(c,1),build_crop_prompt(c,3)]:
   for word in ('象棋','棋子','红黑','圆形','棋局','帅','将','读字','库存','color','"piece"'):
    with self.subTest(word=word,prompt=prompt[:25]):self.assertNotIn(word,prompt)
   for phrase in ('完整实体外缘','0–1000','左 < 右','上 < 下','0 到 1','JSON','独立','class_id','confidence','text','没有可读文字时填空字符串','不要求先读出文字'):
    self.assertIn(phrase,prompt)

 def test_arbitrary_categories_and_verbatim_instructions_reach_both_builders(self):
  c=copy.deepcopy(DEFAULT)
  c['classes']=[{'id':0,'name':'sprocket','label':'链轮'},{'id':1,'name':'sensor','label':'传感器'}]
  c['annotation_instructions']='仅标注托盘内目标。\n允许旋转，文字不可见也按外观识别。'
  before=copy.deepcopy(c)
  for prompt in [build_detection_prompt(c,0),build_detection_prompt(c,2),build_crop_prompt(c,5)]:
   for entry in c['classes']:self.assertIn(f'{entry["id"]}: {entry["label"]} ({entry["name"]})',prompt)
   self.assertIn(c['annotation_instructions'],prompt)
   self.assertNotIn('person',prompt)
  self.assertEqual(c,before)
  self.assertIn('unknown_regions',build_detection_prompt(c,0))
  self.assertIn('kind="unreadable"',build_crop_prompt(c,5))
  self.assertIn('PATCH 5',build_crop_prompt(c,5))

 def test_user_chess_words_are_retained_in_general_mode(self):
  c=copy.deepcopy(DEFAULT);c['classes']=[{'id':0,'name':'chess_box','label':'象棋盒'}]
  c['annotation_instructions']='只标注圆形红黑棋子收纳盒'
  prompt=build_detection_prompt(c,0)
  self.assertIn('象棋盒',prompt);self.assertIn(c['annotation_instructions'],prompt)
  self.assertNotIn('必须读清文字',prompt)

 def test_xiangqi_and_legacy_keep_color_glyph_and_unknown_instructions(self):
  for explicit in (True,False):
   c={'classes':CLASSES,'annotation_instructions':'只标注桌面上的实体'}
   if explicit:c['annotation_profile']='xiangqi'
   for prompt in [build_detection_prompt(c,1),build_crop_prompt(c,2)]:
    for phrase in ('炮/砲','帅/将','红黑','color','unknown','背面','只标注桌面上的实体','不要凭棋局规则'):
     self.assertIn(phrase,prompt)


if __name__=='__main__':unittest.main()
