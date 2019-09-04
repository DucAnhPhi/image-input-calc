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
        filtered= list(filtered)
        return filtered

    def filter_contours2(self, img, contours, hierarchy):
        filtered = []
        for i in range(len(contours)):
            cnt = contours[i]
            valid = self.filter_nested_contour(
                img, contours, hierarchy, cnt, i)
            if valid:
                filtered.append(cnt)
        filtered = self.filter_far_away_contours(filtered)

        filtered= list(filtered)
        return filtered


    def approx_contour(self, cnt, precision=0.1):
        # approx contour shape to another shape with less # vertices depending on precision we specify - uses Douglas-Peucker algorithm.
        # epsilon is max distance from contour to approx. contour
        epsilon = precision * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        return approx


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

    def resize_keep_ratio(self, img, size=75, interpolation=cv.INTER_AREA):
        h, w = img.shape[:2]
        c = None if len(img.shape) < 3 else img.shape[2]
        if h == w:
            return cv.resize(img, (size, size), interpolation)
        if h > w:
            dif = h
        else:
            dif = w
        x_pos = int((dif-w)/2.)
        y_pos = int((dif-h)/2.)
        if c is None:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w,
                 :] = img[:h, :w, :] = img[:h, :w, :]
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

    def get_subimage_list_from_contour_list(self, frame, contourList):
        subimageList = []
        for i in range(len(contourList)):
            subimageList.append(self.get_subimage_from_contour(frame, contourList[i]))
        return subimageList

    def get_subimage_list_list_from_contour_list_list(self, frame, contourListList):
        subimageList = []
        for i in range(len(contourListList)):
            subimageList.append(self.get_subimage_list_from_contour_list(frame, contourListList[i]))
        return subimageList



    def print_subimage_list_Images(self, frame, subimageList, name="Image_"):

        for i in range(len(subimageList)):

            cv.imwrite((name + str(i) + ".png"), self.get_subimage_from_contour(frame, subimageList[i][0]))
            print("Saved the file")

    def print_lineList_images(self, frame, lineList):

        for i in range(len(lineList)):

            name= str("Line_" + str(i) + "_Symbol_")
            self.print_subimage_list_Images(frame, lineList[i], name)


    def print_subimage_list_list_Images(self, frame, orderedLineList, name="TrainingSamples/Image_"):

        for i in range(len(orderedLineList)):
            for j in range(len(orderedLineList[i])):
                cv.imwrite((name + str(i) + "_" + str(j) + ".png"), self.get_subimage_from_contour(frame, orderedLineList[i][j]))
                print("Saved the file")