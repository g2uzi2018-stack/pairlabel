"""Profile selection, including tasks saved before profiles were introduced."""

def is_xiangqi(c):
 if 'annotation_profile' in c:return c['annotation_profile']=='xiangqi'
 # Legacy tasks carry explicit taxonomy metadata. Names and labels alone are
 # never sufficient, and an explicit general profile always takes precedence.
 classes=c.get('classes',[])
 return bool(classes) and all(isinstance(x,dict) and x.get('piece') in
  ('king','advisor','elephant','horse','rook','cannon','pawn') and
  x.get('color') in ('red','black') for x in classes)
