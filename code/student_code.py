import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """
    N, l = np.shape(points_2d)
    thd = points_3d[0]
    twd = points_2d[0]
    A = np.asarray([[thd[0],thd[1],thd[2],1,0,0,0,0,-twd[0]*thd[0],-twd[0]*thd[1],-twd[0]*thd[2]],
                    [0,0,0,0,thd[0],thd[1],thd[2],1,-twd[1]*thd[0],-twd[1]*thd[1],-twd[1]*thd[2]]])
    B = np.asarray([[twd[0]], [twd[1]]])
    for i in range(1,N):
        thd = points_3d[i]
        twd = points_2d[i]
        a = np.asarray([[thd[0],thd[1],thd[2],1,0,0,0,0,-twd[0]*thd[0],-twd[0]*thd[1],-twd[0]*thd[2]],
                        [0,0,0,0,thd[0],thd[1],thd[2],1,-twd[1]*thd[0],-twd[1]*thd[1],-twd[1]*thd[2]]])
        b = np.asarray([[twd[0]], [twd[1]]])
        B = np.vstack((B,b))
        A = np.vstack((A,a))
    M = np.linalg.lstsq(A,B,rcond=-1)[0]
    M = np.vstack((M,np.asarray([[1]])))
    M = M.reshape(3,4)
    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    Q = M[:,:-1]
    m4 = M[:,3]
    Q = -np.linalg.inv(Q)
    cc = np.matmul(Q,m4)

    return cc

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    N,l = np.shape(points_a)

    au = np.mean(points_a[:,0])
    av = np.mean(points_a[:,1])
    bu = np.mean(points_b[:,0])
    bv = np.mean(points_b[:,1])
    sa = np.std(points_a[:,0])+np.std(points_a[:,1])
    sb = np.std(points_b[:,0])+np.std(points_b[:,1])
    sa /= 2
    sb /= 2

    a1 = np.array([[sa,0,0],[0,sa,0],[0,0,1]])
    b1 = np.array([[sb,0,0],[0,sb,0],[0,0,1]])
    a2 = np.array([[1,0,-au],[0,1,-av],[0,0,1]])
    b2 = np.array([[1,0,-bu],[0,1,-bv],[0,0,1]])
    Ta = np.matmul(a1,a2)
    Tb = np.matmul(b1,b2)


    a = points_a[0]
    a = np.append(a,np.asarray([1]))
    b = points_b[0]
    b = np.append(b,np.asarray([1]))
    a = np.matmul(Ta,a)
    b = np.matmul(Tb,b)

    C = np.asarray([[a[0]*b[0], a[1]*b[0], b[0], a[0]*b[1], a[1]*b[1], b[1], a[0], a[1]]])

    for i in range(1,N):
        a = points_a[i]
        a = np.append(a,np.asarray([1]))
        b = points_b[i]
        b = np.append(b,np.asarray([1]))
        a = np.matmul(Ta,a)
        b = np.matmul(Tb,b)

        c = np.asarray([[a[0]*b[0], a[1]*b[0], b[0], a[0]*b[1], a[1]*b[1], b[1], a[0], a[1]]])
        C = np.vstack((C,c))

    d = np.ones(N).T
    F = np.linalg.lstsq(C,d,rcond=-1)[0]
    F = np.append(F,np.asarray([-1]))
    F = F.reshape(3,3)
    u,e,v = np.linalg.svd(F)
    z = np.argsort(e)
    e[z[0]] = 0
    F = np.dot(u*e,v)
    F = np.matmul(np.matmul(Tb.T,F),Ta)

    return F

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """
    N, l = np.shape(matches_a)
    sample = 10
    c = 0
    M = 0
    eps = 0.05
    inlb = np.ones(10)
    best_F = np.ones((3,3))
    x1 = matches_a[:,0]
    y1 = matches_a[:,1]
    x2 = matches_b[:,0]
    y2 = matches_b[:,1]
    r = np.power((np.max(x2)- np.min(x2))*(np.max(y2)- np.min(y2))*(np.max(x1)- np.min(x1))*(np.max(y1)- np.min(y1)),0.25)*0.15
    while(M < 100 and c < 1000):
        c+=1
        m = 0
        ind = np.random.randint(N, size = sample)
        inl = ind
        F = estimate_fundamental_matrix(matches_a[ind, :], matches_b[ind, :])
        for i in range(0,N):
            if i in ind:
                continue
            a = matches_a[i,:]
            a = np.append(a,np.asarray([1]))
            b = matches_b[i,:]
            b = np.append(b,np.asarray([1]))
            dist = np.sqrt(np.square(a[0]-b[0])+ np.square(a[1]-b[1]))
            d = np.matmul(np.matmul(b.T,F),a)
            if (abs(d) < eps ):
                inl = np.append(inl,np.asarray([i]))
                if (dist < r):
                    m+= 1
        if(m > M):
            M = m
            best_F = F
            inlb = inl
        #print(c)
    inlb = np.sort(inlb)
    print(M)

    # Placeholder values

    inliers_a = matches_a[inlb, :]
    inliers_b = matches_b[inlb, :]



    return best_F, inliers_a, inliers_b
