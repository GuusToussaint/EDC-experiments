import numpy as np
import os

def create_korns_t1():
    # Create a dataset with 5000 samples and 25 features
    # with random features ranging from -10 to 10
    X = (np.random.rand(5000, 25) - 0.5) * 20

    D1 = lambda x: np.sum([
        1.57*x[0],
        -39.34*x[1],
        2.13*x[2],
        46.59*x[3],
        11.54*x[4],
    ])
    D2 = lambda x: np.sum([
        -1.57*x[0],
        39.34*x[1],
        -2.13*x[2],
        -46.59*x[3],
        -11.54*x[4],
    ])

    Y = np.array([np.argmax([D1(x), D2(x)]) for x in X])

    return X, Y

def create_korns_t2():
    # Create a dataset with 5000 samples and 25 features
    # with random features ranging from -10 to 10
    X = (np.random.rand(5000, 25) - 0.5) * 20

    D1 = lambda x: np.sum([
        5,16*x[0],
        -19.83*x[1],
        19.83*x[2],
        29.31*x[3],
        5.29*x[4],
    ])
    D2 = lambda x: np.sum([
        -5,16*x[0],
        19.83*x[1],
        -0.93*x[2],
        -29.31*x[3],
        5.29*x[4],
    ])

    Y = np.array([np.argmax([D1(x), D2(x)]) for x in X])

    return X, Y
    
def create_korns_t3():
    # Create a dataset with 5000 samples and 25 features
    # with random features ranging from -10 to 10
    X = (np.random.rand(5000, 25) - 0.5) * 20

    D1 = lambda x: np.sum([
        -34.16*x[0],
        2.19*x[1],
        -12.73*x[2],
        5.62*x[3],
        -16.36*x[4],
    ])
    D2 = lambda x: np.sum([
        34.16*x[0],
        -2.19*x[1],
        12.73*x[2],
        -5.62*x[3],
        16.36*x[4],
    ])

    Y = np.array([np.argmax([D1(x), D2(x)]) for x in X])

    return X, Y

if __name__ == "__main__":
    output_file = os.path.join(
        "/data1/toussaintg1/EDC-datasets/korns"
    )

    X, Y = create_korns_t3()

    print(f"Fraction of class 0: {np.sum(Y == 0) / len(Y)}")

    # Write to a csv file
    np.savetxt(
        os.path.join(
            output_file,
            "t3.csv",
        ),
        np.concatenate(
            [X, Y.reshape(-1, 1)],
            axis=1
        ),
        delimiter=","
    )