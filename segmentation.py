import cv2 as cv
import numpy as np


class Segmentation:
    def filter_big_contour(self, img, cnt, maxThresh=0.3):
        maxArea = len(img) * len(img[0]) * maxThresh
        if cv.contourArea(cnt) >= maxArea:
            return False
        return True

    def not_outer_border(self, img, cnt):
        x, y, width, height = cv.boundingRect(cnt)
        if x == 0 and y == 0:
            if width == img.shape[1] and height == img.shape[0]:
                return False
        return True

    def filter_small_contour(self, img, cnt, minThresh=100):
        if cv.contourArea(cnt) <= minThresh:
            return False
        return True

    def get_centroids_of_contours(self, contours):
        moments = [cv.moments(cnt) for cnt in contours]
        centroids = [(int(m['m10']/m['m00']), int(m['m01']/m['m00']))
                     for m in moments]
        return centroids

    def get_mean_centroid(self, centroids):
        meanX = np.mean(np.array([c[0] for c in centroids]))
        meanY = np.mean(np.array([c[1] for c in centroids]))
        return (meanX, meanY)

    def get_mean_distance_to_mean_centroid(self, centroids, meanX, meanY):
        distances = [np.linalg.norm((c[0]-meanX, c[1]-meanY))
                     for c in centroids]
        return np.mean(np.array(distances))

    def filter_far_away_contours(self, filtered, thresh=1.8):
        centroids = self.get_centroids_of_contours(filtered)
        meanX, meanY = self.get_mean_centroid(centroids)
        meanD = self.get_mean_distance_to_mean_centroid(
            centroids, meanX, meanY)

        def check_too_far(cnt):
            m = cv.moments(cnt)
            centroidX, centroidY = (
                int(m['m10']/m['m00']), int(m['m01']/m['m00']))
            # euclidian distance
            d = np.linalg.norm((centroidX - meanX, centroidY - meanY))
            return d < meanD * thresh

        filtered = filter(check_too_far, filtered)
        return filtered

    def filter_nested_contour(self, img, contours, hierarchy, cnt, index, maxThresh=0.3):
        parentIndex = hierarchy[0][index][-1]
        if parentIndex == -1:
            return True
        else:
            pX, pY, pWidth, pHeight = cv.boundingRect(contours[parentIndex])
            cX, cY, cWidth, cHeight = cv.boundingRect(cnt)
            maxArea = len(img) * len(img[0]) * maxThresh
            if pWidth * pHeight < maxArea:
                if cX >= pX and cY >= pY:
                    if (cX + cWidth) <= (pX+pWidth) and (cY+cHeight) <= (pY+pHeight):
                        return False
            return True

    def filter_contours(self, img, contours, hierarchy):
        filtered = []
        for i in range(len(contours)):
            cnt = contours[i]
            valid = self.filter_big_contour(img, cnt)
            #valid = valid & self.filter_small_contour(img, cnt)
            valid = valid & self.filter_nested_contour(
                img, contours, hierarchy, cnt, i)
            if valid:
                filtered.append(cnt)
        # filtered = self.filter_far_away_contours(filtered)
        return filtered

    def resize_keep_ratio(self, img, size=75, interpolation=cv.INTER_AREA):
        # get height and width of given image
        h, w = img.shape[:2]
        c = None if len(img.shape) < 3 else img.shape[2]
        if h == w:
            return cv.resize(img, (size, size), interpolation)
        # get longest edge
        dif = max(h, w)
        # calculate offsets
        xOffset = int((dif-w)/2.)
        yOffset = int((dif-h)/2.)
        # generate mask with longest edge and offsets
        if c is None:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[yOffset:yOffset+h, xOffset:xOffset+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[yOffset:yOffset+h, xOffset:xOffset+w,
                 :] = img[:h, :w, :] = img[:h, :w, :]
        # return resized mask
        return cv.resize(mask, (size, size), interpolation)

    def get_subimage_from_contour(self, frame, cnt):
        x, y, cntWidth, cntHeight = cv.boundingRect(cnt)
        blankImg = np.zeros(
            shape=frame.shape, dtype=np.uint8)
        cv.drawContours(
            blankImg, [cnt], -1, (255, 255, 255), 1)
        cv.fillPoly(blankImg, pts=[cnt], color=(255, 255, 255))
        subImg = blankImg[y:y+cntHeight, x:x+cntWidth]
        subImg = self.resize_keep_ratio(subImg)
        return subImg
