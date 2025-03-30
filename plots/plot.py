import matplotlib.pyplot as plt
import numpy as np
def plot_grouping():
    ks = ["Advancement", "Novelty", "Soundness", "Clarity", "Compliance"]
    xs = range(1, 6)
    xs_psv1 = (np.array(xs) - 0.3).tolist()
    xs_psv2 = (np.array(xs) - 0.0).tolist()
    xs_psv3 = (np.array(xs) + 0.3).tolist()
    # xs_psv4 = (np.array(xs) + 0.3).tolist()
    print(xs_psv1)

    model_1 = [0.306, 0.296, 0.218, 0.183, 0.479]
    model_2 = [0.132, 0.227, 0.176, 0.112, 0.006]
    model_3 = [0.277, 0.423, 0.245, 0.287, 0.077]
    model_4 = [0.180, 0.153, 0.194, 0.194, 0.008]

    bar_width = 0.3
    plt.figure(figsize=(9, 5))
    plt.bar(xs_psv1, height=model_1, width=bar_width, color='deepskyblue', label="GPT-4o")
    plt.bar(xs_psv2, height=model_2, width=bar_width, color='darkcyan', label="Llama-3.1-8B")
    plt.bar(xs_psv3, height=model_3, width=bar_width, color="orange", label="Llama-3.1-70B")
    # plt.bar(xs_psv4, height=model_4, width=bar_width, color="darkmagenta", label="Mixtral-8x7B-v0.1")
    plt.xticks(xs, ks, fontproperties='Times New Roman', fontsize=22)
    plt.yticks(fontproperties='Times New Roman', fontsize=22)
    plt.ylim(0, 0.5)
    # plt.xlabel("Compression ratio", fontdict={"size":24, "family": 'Times New Roman'})
    plt.ylabel(r"F1", fontsize=22, family='Times New Roman')
    plt.legend(bbox_to_anchor=(0.97, -0.14), ncol=3, prop={"family": 'Times New Roman', "size": 20})
    plt.subplots_adjust(top=0.97, bottom=0.25, right=0.98, left=0.11)
    # plt.show()
    plt.savefig('grouping.png', dpi=1024)


if __name__=="__main__":
    plot_grouping()