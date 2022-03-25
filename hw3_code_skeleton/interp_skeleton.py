import cv2
import sys
import numpy as np
import pickle
import numpy as np
import os

BLUR_OCC = 3


def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def find_holes(flow):
    '''
    Find a mask of holes in a given flow matrix
    Determine it is a hole if a vector length is too long: >10^9, of it contains NAN, of INF
    :param flow: an dense optical flow matrix of shape [h,w,2], containing a vector [ux,uy] for each pixel
    :return: a mask annotated 0=hole, 1=no hole
    '''
    holes=None
    h = flow.shape[0]
    w = flow.shape[1]
    alter_holes = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            if flow[i][j][0] > np.power(10, 9) or flow[i][j][1] > np.power(10,9):
                alter_holes[i][j] = 0
            elif np.isnan(flow[i][j][0]) or np.isnan(flow[i][j][1]):
                alter_holes[i][j] = 0
            elif np.isinf(flow[i][j][0] or np.isinf(flow[i][j][1])):
                alter_holes[i][j] = 0
            else:
                alter_holes[i][j] = 1
    holes = alter_holes
    # to be completed ...
    return holes


def holefill(flow, holes):
    '''
    fill holes in order: row then column, until fill in all the holes in the flow
    :param flow: matrix of dense optical flow, it has shape [h,w,2]
    :param holes: a binary mask that annotate the location of a hole, 0=hole, 1=no hole
    :return: flow: updated flow
    '''
    h,w,_ = flow.shape
    has_hole=1
    while has_hole==1:
        # to be completed ...
        # ===== loop all pixel in x, then in y
        foo = 1
        for y in range(0, h):
            for x in range(0,w):
                # to be completed ...
                # Case 1 Four corner points
                avg_u = 0
                avg_v = 0
                count = 0
                if holes[y][x] == 0 and y == 0 and x == 0:
                    if holes[y][x+1] == 1:
                        avg_u += flow[y][x+1][0]
                        avg_v += flow[y][x+1][1]
                        count += 1
                    if holes[y+1][x] == 1:
                        avg_u += flow[y+1][x][0]
                        avg_v += flow[y+1][x][1]
                        count += 1
                    if holes[y+1][x+1] == 1:
                        avg_u += flow[y+1][x + 1][0]
                        avg_v += flow[y+1][x + 1][1]
                        count += 1
                    if count > 0:
                        flow[y][x][0] = avg_u / count
                        flow[y][x][1] = avg_v / count
                        holes[y][x] = 1
                        foo=0

                avg_u = 0
                avg_v = 0
                count = 0
                if holes[y][x] == 0 and y == 0 and x == w - 1:
                    if holes[y][x-1] == 1:
                        avg_u += flow[y][x - 1][0]
                        avg_v += flow[y][x - 1][1]
                        count += 1
                    if holes[y+1][x-1] == 1:
                        avg_u += flow[y+1][x - 1][0]
                        avg_v += flow[y+1][x - 1][1]
                        count += 1
                    if holes[y+1][x] == 1:
                        avg_u += flow[y+1][x][0]
                        avg_v += flow[y+1][x][1]
                        count += 1
                    if count > 0:
                        flow[y][x][0] = avg_u / count
                        flow[y][x][1] = avg_v / count
                        holes[y][x] = 1
                        foo=0

                avg_u = 0
                avg_v = 0
                count = 0
                if holes[y][x] == 0 and y == h - 1 and x == 0:
                    if holes[y-1][x] == 1:
                        avg_u += flow[y-1][x][0]
                        avg_v += flow[y-1][x][1]
                        count += 1
                    if holes[y-1][x+1] == 1:
                        avg_u += flow[y-1][x + 1][0]
                        avg_v += flow[y-1][x + 1][1]
                        count += 1
                    if holes[y][x+1] == 1:
                        avg_u += flow[y][x + 1][0]
                        avg_v += flow[y][x + 1][1]
                        count += 1
                    if count > 0:
                        flow[y][x][0] = avg_u / count
                        flow[y][x][1] = avg_v / count
                        holes[y][x] = 1
                        foo=0

                avg_u = 0
                avg_v = 0
                count = 0
                if holes[y][x] == 0 and y == h - 1 and x == w - 1:
                    if holes[y][x-1] == 1:
                        avg_u += flow[y][x - 1][0]
                        avg_v += flow[y][x - 1][1]
                        count += 1
                    if holes[y-1][x-1] == 1:
                        avg_u += flow[y-1][x - 1][0]
                        avg_v += flow[y-1][x - 1][1]
                        count += 1
                    if holes[y-1][x] == 1:
                        avg_u += flow[y-1][x][0]
                        avg_v += flow[y-1][x][1]
                        count += 1
                    if count > 0:
                        flow[y][x][0] = avg_u / count
                        flow[y][x][1] = avg_v / count
                        holes[y][x] = 1
                        foo=0

                # Case Two: ALONG THE EDGE
                # TOP ROW
                avg_u = 0
                avg_v = 0
                count = 0
                if holes[y][x] == 0 and y == 0 and x != 0 and x != w-1:
                    if holes[y][x-1] == 1:
                        avg_u += flow[y][x - 1][0]
                        avg_v += flow[y][x - 1][1]
                        count += 1
                    if holes[y][x+1] == 1:
                        avg_u += flow[y][x + 1][0]
                        avg_v += flow[y][x + 1][1]
                        count += 1
                    if holes[y+1][x-1] == 1:
                        avg_u += flow[y+1][x-1][0]
                        avg_v += flow[y+1][x-1][1]
                        count += 1
                    if holes[y+1][x] == 1:
                        avg_u += flow[y+1][x][0]
                        avg_v += flow[y+1][x][1]
                        count += 1
                    if holes[y+1][x+1] == 1:
                        avg_u += flow[y+1][x+1][0]
                        avg_v += flow[y+1][x+1][1]
                        count += 1
                    if count > 0:
                        flow[y][x][0] = avg_u / count
                        flow[y][x][1] = avg_v / count
                        holes[y][x] = 1
                        foo=0
                # LAST ROW
                avg_u = 0
                avg_v = 0
                count = 0
                if holes[y][x] == 0 and y == h - 1 and x != 0 and x != w-1:
                    if holes[y][x-1] == 1:
                        avg_u += flow[y][x - 1][0]
                        avg_v += flow[y][x - 1][1]
                        count += 1
                    if holes[y][x+1] == 1:
                        avg_u += flow[y][x + 1][0]
                        avg_v += flow[y][x + 1][1]
                        count += 1
                    if holes[y-1][x] == 1:
                        avg_u += flow[y-1][x][0]
                        avg_v += flow[y-1][x][1]
                        count += 1
                    if holes[y-1][x-1] == 1:
                        avg_u += flow[y-1][x-1][0]
                        avg_v += flow[y-1][x-1][1]
                        count += 1
                    if holes[y-1][x+1] == 1:
                        avg_u += flow[y-1][x+1][0]
                        avg_v += flow[y-1][x+1][1]
                        count += 1
                    if count > 0:
                        flow[y][x][0] = avg_u / count
                        flow[y][x][1] = avg_v / count
                        holes[y][x] = 1
                        foo=0
                # LEFT COLUMN
                avg_u = 0
                avg_v = 0
                count = 0
                if holes[y][x] == 0 and x == 0 and y != 0 and y != h - 1:
                    if holes[y-1][x] == 1:
                        avg_u += flow[y-1][x][0]
                        avg_v += flow[y-1][x][1]
                        count += 1
                    if holes[y-1][x+1] == 1:
                        avg_u += flow[y-1][x + 1][0]
                        avg_v += flow[y-1][x + 1][1]
                        count += 1
                    if holes[y+1][x] == 1:
                        avg_u += flow[y+1][x][0]
                        avg_v += flow[y+1][x][1]
                        count += 1
                    if holes[y+1][x+1] == 1:
                        avg_u += flow[y+1][x + 1][0]
                        avg_v += flow[y+1][x + 1][1]
                        count += 1
                    if holes[y][x+1] == 1:
                        avg_u += flow[y][x + 1][0]
                        avg_v += flow[y][x + 1][1]
                        count += 1
                    if count > 0:
                        flow[y][x][0] = avg_u / count
                        flow[y][x][1] = avg_v / count
                        holes[y][x] = 1
                        foo=0
                # RIGHT COLUMN
                avg_u = 0
                avg_v = 0
                count = 0
                if holes[y][x] == 0 and x == w - 1 and y != 0 and y != h - 1:
                    if holes[y-1][x] == 1:
                        avg_u += flow[y-1][x][0]
                        avg_v += flow[y-1][x][1]
                        count += 1
                    if holes[y-1][x-1] == 1:
                        avg_u += flow[y - 1][x-1][0]
                        avg_v += flow[y - 1][x-1][1]
                        count += 1
                    if holes[y+1][x] == 1:
                        avg_u += flow[y+1][x][0]
                        avg_v += flow[y+1][x][1]
                        count += 1
                    if holes[y+1][x-1] == 1:
                        avg_u += flow[y + 1][x - 1][0]
                        avg_v += flow[y + 1][x - 1][1]
                        count += 1
                    if holes[y][x-1] == 1:
                        avg_u += flow[y][x - 1][0]
                        avg_v += flow[y][x - 1][1]
                        count += 1
                    if count > 0:
                        flow[y][x][0] = avg_u / count
                        flow[y][x][1] = avg_v / count
                        holes[y][x] = 1
                        foo=0
                #CASE THREE: OTHRE LOCATIONS
                avg_u = 0
                avg_v = 0
                count = 0
                if holes[y][x] == 0 and x != 0 and y != 0 and x != w-1 and y != h-1:
                    if holes[y-1][x-1] == 1:
                        avg_u += flow[y-1][x - 1][0]
                        avg_v += flow[y-1][x - 1][1]
                        count += 1
                    if holes[y-1][x] == 1:
                        avg_u += flow[y-1][x][0]
                        avg_v += flow[y-1][x][1]
                        count += 1
                    if holes[y-1][x+1] == 1:
                        avg_u += flow[y-1][x + 1][0]
                        avg_v += flow[y-1][x + 1][1]
                        count += 1
                    if holes[y][x-1] == 1:
                        avg_u += flow[y][x - 1][0]
                        avg_v += flow[y][x - 1][1]
                        count += 1
                    if holes[y][x+1] == 1:
                        avg_u += flow[y][x + 1][0]
                        avg_v += flow[y][x + 1][1]
                        count += 1
                    if holes[y+1][x-1] == 1:
                        avg_u += flow[y+1][x - 1][0]
                        avg_v += flow[y+1][x - 1][1]
                        count += 1
                    if holes[y+1][x] == 1:
                        avg_u += flow[y+1][x][0]
                        avg_v += flow[y+1][x][1]
                        count += 1
                    if holes[y+1][x+1] == 1:
                        avg_u += flow[y+1][x + 1][0]
                        avg_v += flow[y+1][x + 1][1]
                        count += 1
                    if count > 0:
                        flow[y][x][0] = avg_u / count
                        flow[y][x][1] = avg_v / count
                        holes[y][x] = 1
                        foo=0
        if foo == 1:
            has_hole = 0
    return flow

def occlusions(flow0, frame0, frame1):
    '''
    Follow the step 3 in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.
    :param flow0: dense optical flow
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :return:
    '''
    height,width,_ = flow0.shape
    occ0 = np.zeros([height,width],dtype=np.float32)
    occ1 = np.zeros([height,width],dtype=np.float32)

    # ==================================================
    # ===== step 4/ warp flow field to target frame
    # ==================================================
    flow1 = interpflow(flow0, frame0, frame1, 1.0)
    pickle.dump(flow1, open('flow1.step4.data', 'wb'))
    # ====== score
    flow1       = pickle.load(open('flow1.step4.data', 'rb'))
    flow1_step4 = pickle.load(open('flow1.step4.sample', 'rb'))
    diff = np.sum(np.abs(flow1-flow1_step4))
    print('flow1_step4',diff)

    # for y in range(1,10):
    #     for x in range(1,10):
    #         print([flow1_step4[y][x][0], flow1_step4[y][x][1]])
    # for y in range(1, 3):
    #     for x in range(1,3):
    #         if flow1[y][x][0] != flow1_step4[y][x][0] or flow1[y][x][1] != flow1_step4[y][x][1]:
    #             print(y,x)
    #             print([flow1[y][x], flow1_step4[y][x]])
    #             print([flow1[y][x][0], flow1_step4[y][x][0], flow1[y][x][1], flow1_step4[y][x][1]])
    # ==================================================
    # ===== main part of step 5
    # ==================================================
    # to be completed...
    # for y in range(height):
    #     for x in range(width):
    #         u = flow0[y][x][0]
    #         v = flow0[y][x][1]
    #         x1 = int(u + x)
    #         y1 = int(v + y)
    #         u1 = flow1[y][x][0]
    #         v1 = flow1[y][x][1]
    #         if x1 >= width or y1 >= height:
    #             occ1[y][x] = 1
    #         elif np.sum(abs(flow0[y][x] - flow1[y1][x1])) > 0.5:
    #             occ1[y][x] = 1
    #         elif np.sum(np.abs(flow1[y][x])) > height + width:
    #             occ0[y][x] = 1
    # occ0t = pickle.load(open('occ0.step5.sample', 'rb'))  # load sample result
    # occ1t = pickle.load(open('occ1.step5.sample', 'rb'))  # load sample result

    for y in range(height):
        for x in range(width):
            locv = flow0[y][x][1] + y
            locu = flow0[y][x][0] + x
            if np.isnan(flow1[y][x][0]) or np.isnan(flow1[y][x][1]) or flow1[y][x][0] >= 1e10 or flow1[y][x][
                1] >= 1e10 or np.isinf(flow1[y][x][0]) or np.isinf(flow1[y][x][1]):
                occ1[y][x] = 1

            if locv <= -1.5 or locv >= height - 0.5 or locu <= -1.5 or locu >= width - 0.5:  # if out of bounds
                occ0[y][x] = 1
            else:
                y1 = int(locv)
                x1 = int(locu)
                if locv < 0:
                    y1 = 0
                if locu < 0:
                    x1 = 0
                # num is greater than 0
                if locv >= 0:
                    y1 = np.int32(locv + 0.5)
                if locu >= 0:
                    x1 = np.int32(locu + 0.5)
                if locv >= height:
                    y1 = height - 1
                if locu >= width:
                    x1 = width - 1
                if np.sum(np.abs(flow0[y][x] - flow1[y1][x1])) > 0.5:
                    occ0[y][x] = 1
            # if (int(x + x1) > width - 1) or (int(y + y1) > height - 1):
            #     occ1[y][x] = 1
            #
            # elif np.sum(np.abs(flow0[y][x] - flow1_step4[np.int32(y + y1)][np.int32(x + x1)])) > 0.5:
            #     occ1[y][x] = 1
            #
            # if np.sum(np.abs(flow1_step4[y][x])) > height + width:
            #     occ0[y][x] = 1

    return occ0,occ1


def interpflow(flow, frame0, frame1, t):
    '''
    Forward warping flow (from frame0 to frame1) to a position t in the middle of the 2 frames
    Follow the algorithm (1) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param flow: dense optical flow from frame0 to frame1
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param t: the intermiddite position in the middle of the 2 input frames
    :return: a warped flow
    '''
    #iflow = None
    iflow = np.zeros_like(flow)
    for i in range(iflow.shape[0]):
        for j in range(iflow.shape[1]):
            iflow[i][j] = 1e10

    # to be completed ...
    h, w, _ = iflow.shape
    #print(h,w)  h = 380 w = 420
    #Construct a map to store pixel location in frame t and its corresponding I1-I0 value and location [y,x] in frame 0
    map = []
    for y in range(h):
        map.append([])
        for x in range(w):
            map[y].append([])

    for y in range(h):
        for x in range(w):
            map[y][x].append(1e9)
            map[y][x].append([])

    #print(len(map))
    for y in range(h):
        for x in range(w):
            for yy in np.arange(-0.5, 0.51, 0.5):
                for xx in np.arange(-0.5, 0.51, 0.5):
                    ny = np.int32(y + t * flow[y][x][1] + yy + 0.5)
                    nx = np.int32(x + t * flow[y][x][0] + xx + 0.5)
                    # x1 = np.int32(x + xx + flow[y][x][0] + 0.5)
                    # y1 = np.int32(y + yy + flow[y][x][1] + 0.5)
                    #frame1_val = bilinear(frame1.astype(np.float32), x + xx + flow[y][x][0], y + yy + flow[y][x][1])
                    # if nx >= 0 and nx < w and ny >= 0 and ny < h and x1 >= 0 and x1 < w and y1 < h and y1 >=0:
                    #     if np.sum(np.absolute(frame1[y1][x1] - frame0[y][x])) < map[ny][nx][0]:
                    #         map[ny][nx][0] = np.sum(np.absolute(frame1[y1][x1] - frame0[y][x]))
                    #         map[ny][nx][1] = [y, x]
                    frame1_val = bilinear(frame1, x + xx + flow[y][x][0], y + yy + flow[y][x][1])
                    if nx >= 0 and nx < w and ny >= 0 and ny < h:
                        if np.sum(np.absolute(frame1_val - frame0[y][x])) < map[ny][nx][0]:
                            map[ny][nx][0] = np.sum(np.absolute(frame1_val - frame0[y][x]))
                            map[ny][nx][1] = [y, x]


    for ny in range(h):
        for nx in range(w):
            if map[ny][nx][1] != []:
                iflow[ny][nx] = flow[map[ny][nx][1][0]][map[ny][nx][1][1]]
    return iflow

def bilinear(frame, x, y):
    i = int(np.floor(x))
    j = int(np.floor(y))
    a = x - i
    b = y - j
    if i >= frame.shape[1] - 1 or j >= frame.shape[0] - 1:
        result = frame[frame.shape[0] - 1][frame.shape[1] - 1]
    else:
        if j >= 0 and i >= 0:
            result = ((1 - a) * (1 - b)) * (frame[j, i]) + (a * (1 - b)) * (frame[j, i + 1]) + (a * b) * (
            frame[j + 1, i + 1]) + ((1 - a) * b) * (frame[j + 1, i])
        else:
            result = 0
    return result

def warpimages(iflow, frame0, frame1, occ0, occ1, t):
    '''
    Compute the colors of the interpolated pixels by inverse-warping frame 0 and frame 1 to the postion t based on the
    forwarded-warped flow iflow at t
    Follow the algorithm (4) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
     for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param iflow: forwarded-warped (from flow0) at position t
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param occ0: occlusion mask of frame 0
    :param occ1: occlusion mask of frame 1
    :param t: interpolated position t
    :return: interpolated image at position t in the middle of the 2 input frames
    '''

    iframe = np.zeros_like(frame0).astype(np.float32)
    # to be completed ...
    h, w, _ = iframe.shape
    for y in range(h):
        for x in range(w):
            occ0[y][x] = np.round(occ0[y][x])
            occ1[y][x] = np.round(occ1[y][x])

    for y in range(h):
        for x in range(w):
            u0 = x - t * iflow[y][x][0]
            v0 = y - t * iflow[y][x][1]
            u1 = x + (1 - t) * iflow[y][x][0]
            v1 = y + (1 - t) * iflow[y][x][1]

            y0 = int(round(v0))
            x0 = int(round(u0))
            y1 = int(round(v1))
            x1 = int(round(u1))

            b0 = bilinear(frame0, u0, v0)
            b1 = bilinear(frame1, u1, v1)

            if y0 < 0 or y0 >= frame0.shape[0] or x0 < 0 or x0 >= frame0.shape[1]:
                iframe[y][x] = b1
                continue

            if y1 < 0 or y1 >= frame1.shape[0] or x1 < 0 or x1 >= frame1.shape[1]:
                iframe[y][x] = b0
                continue

            if occ0[y0][x0] == 0 and occ1[y1][x1] == 0:  # blend
                iframe[y][x] = (1 - t) * frame0[y0][x0] + t * frame1[y1][x1]

            elif occ0[y0][x0] == 1 and occ1[y1][x1] == 0:
                iframe[y][x] = b1

            elif occ1[y1][x1] == 1 and occ0[y0][x0] == 0:
                iframe[y][x] = b0
    return iframe

def blur(im):
    '''
    blur using a gaussian kernel [5,5] using opencv function: cv2.GaussianBlur, sigma=0
    :param im:
    :return updated im:
    '''
    # to be completed ...
    im = cv2.GaussianBlur(im, (5,5), 0)
    return im


def internp(frame0, frame1, t=0.5, flow0=None):
    '''
    :param frame0: beggining frame
    :param frame1: ending frame
    :return frame_t: an interpolated frame at time t
    '''
    print('==============================')
    print('===== interpolate an intermediate frame at t=',str(t))
    print('==============================')

    # ==================================================
    # ===== 1/ find the optical flow between the two given images: from frame0 to frame1,
    #  if there is no given flow0, run opencv function to extract it
    # ==================================================
    if flow0 is None:
        i1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        i2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        flow0 = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # ==================================================
    # ===== 2/ find holes in the flow
    # ==================================================
    holes0 = find_holes(flow0)
    pickle.dump(holes0,open('holes0.step2.data','wb'))  # save your intermediate result
    # ====== score
    holes0       = pickle.load(open('holes0.step2.data','rb')) # load your intermediate result
    holes0_step2 = pickle.load(open('holes0.step2.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes0-holes0_step2))
    print('holes0_step2',diff)

    # ==================================================
    # ===== 3/ fill in any hole using an outside-in strategy
    # ==================================================
    flow0 = holefill(flow0,holes0)
    pickle.dump(flow0, open('flow0.step3.data', 'wb')) # save your intermediate result
    # ====== score
    flow0       = pickle.load(open('flow0.step3.data', 'rb')) # load your intermediate result
    flow0_step3 = pickle.load(open('flow0.step3.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow0-flow0_step3))
    print('flow0_step3',diff)

    # ==================================================
    # ===== 5/ estimate occlusion mask
    # ==================================================
    occ0, occ1 = occlusions(flow0,frame0,frame1)
    pickle.dump(occ0, open('occ0.step5.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step5.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step5.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step5.data', 'rb')) # load your intermediate result
    occ0_step5  = pickle.load(open('occ0.step5.sample', 'rb')) # load sample result
    occ1_step5  = pickle.load(open('occ1.step5.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step5 - occ0))
    print('occ0_step5',diff)
    diff = np.sum(np.abs(occ1_step5 - occ1))
    print('occ1_step5',diff)

    # ==================================================
    # ===== step 6/ blur occlusion mask
    # ==================================================
    for iblur in range(0,BLUR_OCC):
        occ0 = blur(occ0)
        occ1 = blur(occ1)
    pickle.dump(occ0, open('occ0.step6.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step6.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step6.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step6.data', 'rb')) # load your intermediate result
    occ0_step6  = pickle.load(open('occ0.step6.sample', 'rb')) # load sample result
    occ1_step6  = pickle.load(open('occ1.step6.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step6 - occ0))
    print('occ0_step6',diff)
    diff = np.sum(np.abs(occ1_step6 - occ1))
    print('occ1_step6',diff)

    # ==================================================
    # ===== step 7/ forward-warp the flow to time t to get flow_t
    # ==================================================
    flow_t = interpflow(flow0, frame0, frame1, t)
    pickle.dump(flow_t, open('flow_t.step7.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step7.data', 'rb')) # load your intermediate result
    flow_t_step7 = pickle.load(open('flow_t.step7.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step7))
    print('flow_t_step7',diff)

    # ==================================================
    # ===== step 8/ find holes in the estimated flow_t
    # ==================================================
    holes1 = find_holes(flow_t)
    pickle.dump(holes1, open('holes1.step8.data', 'wb')) # save your intermediate result
    # ====== score
    holes1       = pickle.load(open('holes1.step8.data','rb')) # load your intermediate result
    holes1_step8 = pickle.load(open('holes1.step8.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes1-holes1_step8))
    print('holes1_step8',diff)

    # ===== fill in any hole in flow_t using an outside-in strategy
    flow_t = holefill(flow_t, holes1)
    pickle.dump(flow_t, open('flow_t.step8.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t = pickle.load(open('flow_t.step8.data', 'rb')) # load your intermediate result
    flow_t_step8 = pickle.load(open('flow_t.step8.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step8))
    print('flow_t_step8',diff)

    # ==================================================
    # ===== 9/ inverse-warp frame 0 and frame 1 to the target time t
    # ==================================================
    frame_t = warpimages(flow_t, frame0, frame1, occ0, occ1, t)
    pickle.dump(frame_t, open('frame_t.step9.data', 'wb')) # save your intermediate result
    # ====== score
    frame_t       = pickle.load(open('frame_t.step9.data', 'rb')) # load your intermediate result
    frame_t_step9 = pickle.load(open('frame_t.step9.sample', 'rb')) # load sample result
    diff = np.sqrt(np.mean(np.square(frame_t.astype(np.float32)-frame_t_step9.astype(np.float32))))
    print('frame_t',diff)

    return frame_t


if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2020, HW3: video frame interpolation')
    print('==================================================')

    # ===================================
    # example:
    # python interp_skeleton.py frame0.png frame1.png flow0.flo frame05.png
    # ===================================
    path_file_image_0 = sys.argv[1]
    path_file_image_1 = sys.argv[2]
    path_file_flow    = sys.argv[3]
    path_file_image_result = sys.argv[4]

    # ===== read 2 input images and flow
    frame0 = cv2.imread(path_file_image_0)
    frame1 = cv2.imread(path_file_image_1)
    flow0  = readFlowFile(path_file_flow)

    # ===== interpolate an intermediate frame at t, t in [0,1]
    frame_t= internp(frame0=frame0, frame1=frame1, t=0.5, flow0=flow0)
    cv2.imwrite(filename=path_file_image_result, img=(frame_t * 1.0).clip(0.0, 255.0).astype(np.uint8))
