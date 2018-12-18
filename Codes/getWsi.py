
def getWsi(path): #imports a WSI
  import openslide
  wsi = openslide.OpenSlide(path)
  return wsi
