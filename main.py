# Investigation of the work of background subtraction methods and frame annotation
# options: darkening with background blurring and highlighting foreground objects.


import cv2
import numpy as np


def ob_highlighting(frame,foreground):

    blur = cv2.blur(frame,(15,15))
    contrast = cv2.convertScaleAbs(blur, alpha=0.3, beta=10)

    res = contrast.copy()
    non_zero_pixels = np.count_nonzero(foreground)

    if non_zero_pixels > 5000:
        res[foreground > 0 ] = frame[foreground > 0 ]

    return res, non_zero_pixels

def main():

    capture = cv2.VideoCapture('test4.avi')
    # initializing subtractors
    mogSub = cv2.bgsegm.createBackgroundSubtractorMOG(9000)
    mog2Sub = cv2.createBackgroundSubtractorMOG2(9000, 100, True)
    gmgSub = cv2.bgsegm.createBackgroundSubtractorGMG(400, 0.8)
    knnSub = cv2.createBackgroundSubtractorKNN(9000, 100, True)
    cntSub = cv2.bgsegm.createBackgroundSubtractorCNT(5, True)

    frameCount = 0

    Text_Color = (255, 255, 255)

    while(capture.isOpened()):
        ret, frame = capture.read()

        if not ret:
            break

        frameCount += 1
        (H, W) = frame.shape[:2]
        # applying subtractors to the current frame
        fg_mog = mogSub.apply(frame)
        fg_mog2 = mog2Sub.apply(frame)
        fg_gmg = gmgSub.apply(frame)
        fg_knn = knnSub.apply(frame)
        fg_cnt = cntSub.apply(frame)

        fg_mog2[fg_mog2 < 250 ] = 0
        fg_knn[fg_knn < 250 ] = 0

        # noise filtering and pixel overlap
        kernel = np.ones((10,10),np.uint8)
        fg_gmg = cv2.erode(fg_gmg,kernel,iterations = 1)
        fg_gmg = cv2.dilate(fg_gmg,(15,15),iterations = 2)

        conv_kernel = np.array([[0,    0,    1/12, 0,    0],
                                [0,    0,    2/12, 0,    0],
                                [1/12, 2/12, 0,    2/12, 1/12],
                                [0,    0,    2/12, 0,    0],
                                [0,    0,    1/12, 0,    0]])

        fg_mog = cv2.filter2D(src=fg_mog, ddepth=-1, kernel=conv_kernel)
        fg_mog2 = cv2.filter2D(src=fg_mog2, ddepth=-1, kernel=conv_kernel)
        fg_gmg = cv2.filter2D(src=fg_gmg, ddepth=-1, kernel=conv_kernel)
        fg_knn = cv2.filter2D(src=fg_knn, ddepth=-1, kernel=conv_kernel)
        fg_cnt = cv2.filter2D(src=fg_cnt, ddepth=-1, kernel=conv_kernel)

        kernel = np.ones((3,3),np.uint8)
        fg_mog = cv2.erode(fg_mog,kernel,iterations = 1)
        fg_mog = cv2.dilate(fg_mog,(10,10),iterations = 2)

        kernel = np.ones((3,3),np.uint8)
        fg_mog2 = cv2.erode(fg_mog2,kernel,iterations = 1)
        fg_mog2 = cv2.dilate(fg_mog2,(10,10),iterations = 2)

        kernel = np.ones((9,9),np.uint8)
        fg_knn = cv2.erode(fg_knn,kernel,iterations = 1)
        fg_knn = cv2.dilate(fg_knn,(15,15),iterations = 2)

        kernel = np.ones((3,3),np.uint8)
        fg_cnt = cv2.erode(fg_cnt,kernel,iterations = 1)
        fg_cnt = cv2.dilate(fg_cnt,(10,10),iterations = 2)

        Text_Position = (30, 30)
        Text_Size = 0.7

        # Frame annotation and counting of non-zero pixels
        bo_frame_mog, non_zero_pixels_MOG = ob_highlighting(frame,fg_mog)
        bo_frame_mog2, non_zero_pixels_MOG2 = ob_highlighting(frame,fg_mog2)
        bo_frame_gmg, non_zero_pixels_GMG = ob_highlighting(frame,fg_gmg)
        bo_frame_knn, non_zero_pixels_KNN = ob_highlighting(frame,fg_knn)
        bo_frame_cnt, non_zero_pixels_CNT = ob_highlighting(frame,fg_cnt)

        # output information about the current frame
        cv2.putText(bo_frame_mog,  'Fr_N: %d, Pix: %d' % (frameCount, non_zero_pixels_MOG),  Text_Position, cv2.FONT_ITALIC, Text_Size, Text_Color, 1, cv2.LINE_4)
        cv2.putText(bo_frame_mog2, 'Fr_N: %d, Pix: %d' % (frameCount, non_zero_pixels_MOG2), Text_Position, cv2.FONT_ITALIC, Text_Size, Text_Color, 1, cv2.LINE_4)
        cv2.putText(bo_frame_gmg,  'Fr_N: %d, Pix: %d' % (frameCount, non_zero_pixels_GMG),  Text_Position, cv2.FONT_ITALIC, Text_Size, Text_Color, 1, cv2.LINE_4)
        cv2.putText(bo_frame_knn,  'Fr_N: %d, Pix: %d' % (frameCount, non_zero_pixels_KNN),  Text_Position, cv2.FONT_ITALIC, Text_Size, Text_Color, 1, cv2.LINE_4)
        cv2.putText(bo_frame_cnt,  'Fr_N: %d, Pix: %d' % (frameCount, non_zero_pixels_CNT),  Text_Position, cv2.FONT_ITALIC, Text_Size, Text_Color, 1, cv2.LINE_4)

        cv2.imshow('ORIG', frame)
        cv2.imshow('MOG', bo_frame_mog)
        cv2.imshow('MOG2', bo_frame_mog2)
        cv2.imshow('GMG', bo_frame_gmg)
        cv2.imshow('KNN', bo_frame_knn)
        cv2.imshow('CNT', bo_frame_cnt)

        cv2.moveWindow('ORIG', 0, 0)
        cv2.moveWindow('MOG', W, 0)
        cv2.moveWindow('MOG2',  2*W, 0)
        cv2.moveWindow('GMG', 0, 110+H)
        cv2.moveWindow('KNN', W, 110+H)
        cv2.moveWindow('CNT', 2*W, 110+H)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
