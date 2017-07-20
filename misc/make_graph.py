import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


class LCGraph(object):
    """Learning Curve Graph
    """

    def __init__(self, title, sizes, ylim=None):
        """
        Args:
            title: graph titile
            sizes: list of sum which is number of train size
            ylim: tuple (ymin, ymax) Defines minimum and maximum yvalues plotted.
        """
        self.title = title
        self.sizes = sizes
        self.ylim = ylim
        self.train_lines = []
        self.validation_lines = []

    def add_line_pair(self, label, vals_tr, vals_vd):

        """Add new line in a graph

        Args:
            label: label of a line pair
            vals_tr: values of score each steps on training
            vals_vd: values of score each steps on validation
        """
        if len(self.sizes) != len(vals_tr) or len(self.sizes) != len(vals_vd):
            msg = "Invalid data points, {} {} {}".format(len(self.sizes), len(vals_tr), len(vals_vd))
            raise Exception(msg)

        self.train_lines.append(("(t) " + label, vals_tr))
        self.validation_lines.append(("(v) " + label, vals_vd))
        return self

    def save(self, filename):
        types = ["-", "--", "-.", ":", ".", ",", "o", "^",
                 "v", "<", ">", "s", "+", "x", "D", "d",
                 "1", "2", "3", "4", "h", "H", "p", "|", "_"]
        plt.figure(figsize=(25, 16))
        plt.title(self.title)
        if self.ylim is not None:
            plt.ylim(*self.ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        for (label, vals), t in zip(self.train_lines, types):
            plt.plot(self.sizes, vals, t, color="r", label=label)

        for (label, vals), t in zip(self.validation_lines, types):
            plt.plot(self.sizes, vals, t, color="g", label=label)

        plt.legend(loc="best")
        plt.savefig(filename)


def main():
    parser = argparse.ArgumentParser(description="Graph Plotter")
    parser.add_argument("log", type=str, help="filename of log like lc.json")
    parser.add_argument("tags", type=str, help="filename of tags like tags.txt")
    args = parser.parse_args()

    with open(args.tags, "r", encoding="utf-8") as fd:
        tags = fd.read().split("\n")

    dest_dir = os.path.dirname(args.log)
    with open(args.log, "r", encoding="utf-8") as fd:
        res = json.loads(fd.read())

    num_iter = len(res["training_fscore"])
    sizes = np.cumsum([sum([res["training_{}_support".format(tag)][i] for tag in tags])
                       for i in range(num_iter)])

    LCGraph("learning curve of chunk level fscore", sizes)\
        .add_line_pair("", res["training_fscore"], res["validation_fscore"])\
        .save(os.path.join(dest_dir, "lc_f.png"))

    LCGraph("learning curve of accuracy", sizes)\
        .add_line_pair("", res["training_accuracy"], res["validation_accuracy"])\
        .save(os.path.join(dest_dir, "lc_acc.png"))

    x = LCGraph("learning curve of fcore each classes", sizes)
    macro_tr = [np.mean([res["training_{}_fscore".format(tag)][i] for tag in tags])
                for i in range(num_iter)]
    macro_vd = [np.mean([res["validation_{}_fscore".format(tag)][i] for tag in tags])
                for i in range(num_iter)]
    x.add_line_pair("macro", macro_tr, macro_vd)
    for tag in tags:
        key_training = "_".join(["training", tag, "fscore"])
        key_validation = "_".join(["validation", tag, "fscore"])
        x.add_line_pair(str(tag), res[key_training], res[key_validation])
    x.save(os.path.join(dest_dir, "lc_fs.png"))

if __name__ == "__main__":
    main()

