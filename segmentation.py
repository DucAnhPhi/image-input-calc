import cv2 as cv

class Segmentation:
    def get_contours(self, img):
        _, contours, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        return contours
    
    def get_bounding_rect(self, img, cnt):
        x, y, w, h = cv.boundingRect(cnt)

    def draw_contours(self, img):
        tmp = img.copy()
        cnt = self.get_contours(tmp)
        cv.drawContours(tmp, cnt, -1, (0,255,0), 3)
        return tmp

