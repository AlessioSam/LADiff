import matplotlib.pyplot as plt

def plot_skeleton(skel):
    # skel size = 21, 3
    skel = skel.detach().cpu().numpy()
    # 1. plot skeleton
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('skeleton')
    ax.scatter(skel[:, 0], skel[:, 1], skel[:, 2], c='r', marker='o')
    for i in range(0, 20):
        ax.plot([skel[i, 0], skel[i+1, 0]], [skel[i, 1], skel[i+1, 1]], [skel[i, 2], skel[i+1, 2]], c='b')
    plt.show()

    # save skeleton
    fig.savefig('skeleton.png')