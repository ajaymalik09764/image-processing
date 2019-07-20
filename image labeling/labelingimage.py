import numpy as np
import cv2
cv2.namedWindow('image')
img=cv2.imread('img3.jpg')
img=cv2.resize(img,(800,600))
cv2.imshow('image',img)
p=img.copy()

point=[]

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        point.append([x,y])
        cv2.circle(p,(x,y),3,(255,0,0),-1)
cv2.setMouseCallback('image',draw_circle)

s=img.copy()
def rec(point):
    l1=[i[0] for i in point]
    l2=[i[1] for i in point]
    p1=(min(l1),min(l2))
    p2=(max(l1),max(l2))

    cv2.rectangle(s,p1,p2,(0,255,0),1)
    cv2.imwrite("labeledimage.png",s)

def mapp(h):
    #h=h.reshape((4,2))
    hnew=np.zeros((4,2),dtype = np.float32)

    add=h.sum(axis=1)
    hnew[0]=h[np.argmin(add)]
    hnew[3]=h[np.argmax(add)]

    diff=np.diff(h,axis = 1)
    hnew[1]=h[np.argmin(diff)]
    hnew[2]=h[np.argmax(diff)]

    return hnew


def crop(p):
    pts=np.float32([[0,0],[600,0],[0,800],[600,800]])  #map to 800*800 target window
    op=cv2.getPerspectiveTransform(p,pts)  #get the top or bird eye view effect
    dst=cv2.warpPerspective(img,op,(600,800))
    cv2.imwrite("crop.png",dst)


while(1):
    cv2.imshow('image',p)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        rec(point)
        cv2.imshow("image",s)
        p1=mapp(np.array(point).astype("float32"))
        crop(p1)
        break
    elif k== ord('q'):
        break
        
cv2.destroyAllWindows()