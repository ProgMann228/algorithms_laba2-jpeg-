import numpy as np

def ZigZag(block,N):
    zigzag=[]
    k=0
    i,j=0,0
    diag_start_i = i
    diag_start_j = j

    while len(zigzag) < N * N:
       # print(f"indexes",i,j)
        zigzag.append(block[i][j])
        if k%2==0:
          #  print("четный")
            if(i==diag_start_j and j==diag_start_i):
              #  print("конец диаг")
                if j < (N-1):
                    j+=1
                else:
                    i+=1
                k += 1
                diag_start_i = i
                diag_start_j = j
            else:
                i=i-1 if i>0 else 0
                j+=1

        else:
           # print("нечетный")
            if(i==diag_start_j and j==diag_start_i):
              #  print("конец диаг")
                if i<(N-1):
                    i+=1
                else:
                    j+=1
                k += 1
                diag_start_i = i
                diag_start_j = j
            else:
                i+=1
                j=j-1 if i>0 else 0

    return zigzag


def iZigZag(zig_list,N):
    block = np.zeros((N, N), dtype=int)
    k=0
    i,j=0,0
    diag_start_i = i
    diag_start_j = j
    a=0
    for a in range(len(zig_list)):
        if i >= N or j >= N:
            break
        block[i][j]=zig_list[a]
        if k%2==0:
            if(i==diag_start_j and j==diag_start_i):
                if j < (N-1):
                    j+=1
                else:
                    i+=1
                k += 1
                diag_start_i = i
                diag_start_j = j
            else:
                i=i-1 if i>0 else 0
                j+=1

        else:
            if(i==diag_start_j and j==diag_start_i):
                if i<(N-1):
                    i+=1
                else:
                    j+=1
                k += 1
                diag_start_i = i
                diag_start_j = j
            else:
                i+=1
                j=j-1 if i>0 else 0

    return block

"""
block = np.array([
    [ 0,  1,  5,  6, 14, 15, 27, 28],
    [ 2,  4,  7, 13, 16, 26, 29, 42],
    [ 3,  8, 12, 17, 25, 30, 41, 43],
    [ 9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63]
])

rez=ZigZag(block,8)
print(rez)
"""