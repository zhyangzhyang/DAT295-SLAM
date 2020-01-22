import numpy as np
# map 1 RAW


# map 2 RAW
#gmapping_2 = np.asarray([[0, 0],
 #           [-0.07, 2.41],
  #          [1.76, 3.78],
   #         [2.74, 4.05],
    #        [3.10, 0.52],
     #       [0.03, 0.35]])


gmapping_1 = np.asarray([[0, 0],
            [-0.16, 0.88],
            [2.27, 2.04],
            [2.46, 0.38],
            [-0.04, 0.18]])            

#ground_truth_2 = np.asarray([[0.51, 0.49],
#            [0.63, 2.93],
#            [2.63, 4.10],
#            [3.60, 4.08],
#            [3.58, 0.69],
#            [0.60, 0.74]])

ground_truth_1 = np.asarray([[0.61,0.46],
            [0.6,2.33],
            [3.03,2.35],
            [3.08,0.51],
            [0.61,0.51]])            

def offset_data(data, x_offset, y_offset):
    return np.add(data, np.asarray([x_offset, y_offset]))

def get_score(data, ground_truth):
    offset_dataset = offset_data(data, ground_truth[0][0], ground_truth[0][1])

    res = np.subtract(ground_truth, offset_dataset)
    return np.sqrt(np.sum(res ** 2, axis=1)).sum()

print(get_score(gmapping_1, ground_truth_1))