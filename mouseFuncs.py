import pyautogui as p

def pydrag(coords):
    p.moveTo(coords[0][0], coords[0][1])
    for c in coords[1:]:
        p.dragTo(c[0], c[1], .5, button='left')

def getPos():
    x, y = p.position()
    return (x,y)
