import icp
import numpy as np 
import PIL
import sys

def get_coordinates(path):
    points = []
    with PIL.Image.open(path) as im:
        w, h = im.size
        for x in range(w):
            for y in range(h):
                p = im.getpixel((x, y))
                if (p == 0): # black pixel signifies walls
                    points.append([x, y])

    return points

# returns two arrays containing count elements each
# sampled from a and b respectively
def sample_arrays(a, b, count):
    a_indices = np.random.choice(a.shape[0], count, replace=False)
    b_indices = np.random.choice(b.shape[0], count, replace=False)

    return a[a_indices], b[b_indices]

def compare_maps(path1, path2):
    points1 = get_coordinates(path1)
    points2 = get_coordinates(path2)

    n_samples = int(min(len(points1), len(points2))  * 0.7) # use 70% of the smallest map for sampling
    sp1, sp2 = sample_arrays(np.asarray(points1), np.asarray(points2), n_samples)

    T, distances, iters = icp.icp(sp1, sp2)
    score = np.sqrt(np.mean(distances ** 2))
    print("Map score: " + str(score) + " (took " + str(iters) + "iterations)")

if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("usage: ./" +sys.argv[0] + "<source map> <dest map>")
        exit(0)

    compare_maps(sys.argv[1], sys.argv[2])
