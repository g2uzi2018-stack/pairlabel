#!/usr/bin/env python3
"""Synthetic fixtures with exact truth. These are not real-world accuracy evidence."""
import json,math,random
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'samples';OUT.mkdir(exist_ok=True)
FONT='/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'

def make(name,pieces,size=(900,1000),rotate=False):
 w,h=size;rng=np.random.default_rng(12);texture=np.full((h,w,3),[221,186,132],dtype=float)+rng.normal(0,1.7,(h,w,1));im=Image.fromarray(np.clip(texture,0,255).astype('uint8'));d=ImageDraw.Draw(im)
 left,top,dx,dy=90,95,90,90
 for y in range(10):d.line((left,top+y*dy,left+8*dx,top+y*dy),fill='#886a44',width=2)
 for x in range(9):
  if x in [0,8]:d.line((left+x*dx,top,left+x*dx,top+9*dy),fill='#886a44',width=2)
  else:
   d.line((left+x*dx,top,left+x*dx,top+4*dy),fill='#886a44',width=2);d.line((left+x*dx,top+5*dy,left+x*dx,top+9*dy),fill='#886a44',width=2)
 for y in [0,7]:
  d.line((left+3*dx,top+y*dy,left+5*dx,top+(y+2)*dy),fill='#886a44',width=2);d.line((left+5*dx,top+y*dy,left+3*dx,top+(y+2)*dy),fill='#886a44',width=2)
 d.text((220,top+4*dy+15),'楚 河',font=ImageFont.truetype(FONT,40),fill='#a37d4c');d.text((530,top+4*dy+15),'漢 界',font=ImageFont.truetype(FONT,40),fill='#a37d4c')
 truth=[];radius=37
 for cid,char,col,row in pieces:
  cx=left+col*dx;cy=top+row*dy
  d.ellipse((cx-radius+3,cy-radius+6,cx+radius+4,cy+radius+6),fill='#997c58')
  d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill='#f4dfae',outline='#ac8a57',width=3)
  color='#b72227' if cid<7 else '#26272a';d.ellipse((cx-radius+6,cy-radius+6,cx+radius-6,cy+radius-6),outline=color,width=2)
  glyph=Image.new('RGBA',(70,70));gd=ImageDraw.Draw(glyph);gd.text((35,32),char,font=ImageFont.truetype(FONT,46),anchor='mm',fill=color,stroke_width=0)
  if rotate:glyph=glyph.rotate(180 if cid>=7 else -12,resample=getattr(Image,'Resampling',Image).BICUBIC)
  im.paste(glyph,(cx-35,cy-35),glyph)
  truth.append({'class_id':cid,'bbox':[cx-radius,cy-radius,cx+radius,cy+radius],'confidence':1.0,'text':char})
 im.save(OUT/(name+'.png'));(OUT/(name+'.truth.json')).write_text(json.dumps({'objects':truth,'synthetic':True,'note':'Programmatically generated; real-world accuracy is not established.'},ensure_ascii=False,indent=2))

small=[(11,'車',1,0),(7,'將',4,0),(13,'卒',7,3),(6,'兵',1,6),(0,'帥',4,9),(5,'炮',7,7)]
make('six-pieces',small);make('rotated-pieces',small,rotate=True)
full=[]
for offset,row,chars in [(7,0,'車馬象士將士象馬車'),(0,9,'車馬相仕帥仕相馬車')]:
 kinds=[4,3,2,1,0,1,2,3,4]
 for x,(kind,char) in enumerate(zip(kinds,chars)):full.append((offset+kind,char,x,row))
for offset,row,char in [(7,2,'砲'),(0,7,'炮')]:
 for x in [1,7]:full.append((offset+5,char,x,row))
for offset,row,char in [(7,3,'卒'),(0,6,'兵')]:
 for x in [0,2,4,6,8]:full.append((offset+6,char,x,row))
make('full-board',full)
Image.new('RGB',(640,480),'#ebe9e4').save(OUT/'empty.png');(OUT/'empty.truth.json').write_text('{"objects":[],"synthetic":true}')
print(OUT)
