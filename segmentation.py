import cv2 as cv
import numpy as np


class Segmentation:
    def filter_big_contour(self, img, cnt, maxThresh=0.3):
        maxArea = len(img) * len(img[0]) * maxThresh
        if cv.contourArea(cnt) >= maxArea:
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
            valid = valid & self.filter_small_contour(img, cnt)
            valid = valid & self.filter_nested_contour(
                img, contours, hierarchy, cnt, i)
            if valid:
                filtered.append(cnt)
        filtered = self.filter_far_away_contours(filtered)
        return filtered

    def approx_contour(self, cnt, precision=0.1):
        # approx contour shape to another shape with less # vertices depending on precision we specify - uses Douglas-Peucker algorithm.
        # epsilon is max distance from contour to approx. contour
        epsilon = precision * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        return approx


    def filtered_to_contourlist(self,filtered):
        filteredContours=[]
        for cnt in filtered:
            filteredContours.append(cnt)
        return filteredContours

    def get_properties_mincircle(self,filteredContours):
        xList = []
        yList = []
        rList = []

        for i in range(len(filteredContours)):

            (x, y), r = cv.minEnclosingCircle(filteredContours[i])

            xList.append(x)
            yList.append(y)
            rList.append(r)

        return xList,yList,rList
